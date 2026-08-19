from __future__ import annotations

import hashlib
import logging
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from config import SegmentationKnowledgeConfig
from data.corpus import SequenceChunk, is_tangut

from .base import FeatureMatrix, KnowledgeFeatureProvider


LOGGER = logging.getLogger(__name__)

_TAGS = ("B", "I", "E", "S")
_DICT_NAMES = (
    "B2", "B3", "B4", "B5P",
    "I3", "I4", "I5P",
    "E2", "E3", "E4", "E5P",
    "rel_seen_B", "rel_seen_I", "rel_seen_E",
    "rel_unseen_B", "rel_unseen_I", "rel_unseen_E",
    "has_yi", "has_yin", "has_book_title",
)
_GAP_NAMES = (
    "L_freq", "L_assoc", "L_ent_prev", "L_ent_cur",
    "R_freq", "R_assoc", "R_ent_cur", "R_ent_next",
)
_B_INDEX = {"2": 0, "3": 1, "4": 2, "5P": 3}
_I_INDEX = {"3": 4, "4": 5, "5P": 6}
_E_INDEX = {"2": 7, "3": 8, "4": 9, "5P": 10}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _length_bin(length: int) -> str:
    return str(length) if length in {2, 3, 4} else "5P"


def _entropy(probabilities: dict[str, float]) -> float:
    value = -sum(
        probability * math.log(probability)
        for probability in probabilities.values()
        if probability > 0
    )
    return value / math.log(len(_TAGS))


class _StringTrie:
    """供辞书匹配和人工标注重叠审计共用的轻量字符串Trie。"""

    _TERMINALS = "\0"

    def __init__(self) -> None:
        self.root: dict[str, Any] = {}

    def insert(self, text: str, payload: Any) -> None:
        node = self.root
        for character in text:
            node = node.setdefault(character, {})
        node.setdefault(self._TERMINALS, []).append(payload)

    def matches(self, text: str):
        for start in range(len(text)):
            node = self.root
            for end in range(start, len(text)):
                node = node.get(text[end])
                if node is None:
                    break
                for payload in node.get(self._TERMINALS, ()):
                    yield start, end + 1, payload


class _FrozenCRFAdapter:
    """不导入隔壁工程包，直接只读恢复其CRF、辞书和gap状态。"""

    def __init__(self, config: SegmentationKnowledgeConfig) -> None:
        self.config = config
        self.model: Any | None = None
        self.word_set: set[str] = set()
        self.lexicon = _StringTrie()
        self.reliability: dict[str, dict[str, Any]] = {}
        self.class_priors: dict[str, float] = {}
        self.bigram_log_frequency: dict[tuple[str, str], float] = {}
        self.bigram_association: dict[tuple[str, str], float] = {}
        self.left_entropy: dict[str, float] = {}
        self.right_entropy: dict[str, float] = {}
        self.gap_metric = "unknown"
        self.resource_hashes: dict[str, str] = {}
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        if not self.config.model_path or not self.config.lexicon_path or not self.config.gap_path:
            raise ValueError("E4分词资源路径不完整")
        try:
            import joblib
        except ImportError as error:
            raise RuntimeError("E4加载CRF分词器需要joblib/sklearn-crfsuite环境") from error

        payload = joblib.load(self.config.model_path)
        if not isinstance(payload, dict) or "model" not in payload:
            raise ValueError("E4 CRF_model.joblib格式不符合xixia_seg推理模型")
        self.model = payload["model"]
        self.word_set = set(payload.get("word_set", ()))
        if not hasattr(self.model, "predict_marginals_single"):
            raise ValueError("E4分词CRF不支持BIES边缘概率输出")

        with self.config.lexicon_path.open("rb") as stream:
            lexicon_state = pickle.load(stream)
        for word, info in lexicon_state.get("trie_entries", ()):
            self.lexicon.insert(word, (word, info))
        self.reliability = dict(lexicon_state.get("reliability", {}))
        self.class_priors = dict(lexicon_state.get("class_priors", {}))

        with self.config.gap_path.open("rb") as stream:
            gap_state = pickle.load(stream)
        self._restore_gap(gap_state)
        self.resource_hashes = {
            "model": _file_sha256(self.config.model_path),
            "lexicon": _file_sha256(self.config.lexicon_path),
            "gap": _file_sha256(self.config.gap_path),
        }
        self.loaded = True

    @staticmethod
    def _neighbor_entropy(neighbors: dict[str, dict[str, int]]) -> dict[str, float]:
        output: dict[str, float] = {}
        for character, counts in neighbors.items():
            total = sum(counts.values())
            if total:
                output[character] = -sum(
                    (count / total) * math.log(count / total)
                    for count in counts.values()
                    if count
                )
        return output

    def _restore_gap(self, state: dict[str, Any]) -> None:
        metric = str(state.get("bigram_metric", "dpmi"))
        bigram = {
            tuple(key.split("|")): int(value)
            for key, value in state.get("bigram_freq", {}).items()
        }
        unigram = Counter(state.get("unigram_freq", {}))
        total = int(state.get("total_bigrams", sum(bigram.values())))
        self.bigram_log_frequency = {
            pair: math.log1p(count) for pair, count in bigram.items()
        }
        association: dict[tuple[str, str], float] = {}
        for pair, count in bigram.items():
            left, right = pair
            left_count = max(unigram.get(left, 1), 1)
            right_count = max(unigram.get(right, 1), 1)
            if metric == "dice":
                value = 2.0 * count / (left_count + right_count)
            elif metric == "t_score":
                expected = left_count * right_count / max(total, 1)
                value = (count - expected) / math.sqrt(count) if count else 0.0
            else:
                discounted = count - 0.5
                value = (
                    max(
                        0.0,
                        math.log(discounted)
                        - math.log(left_count)
                        - math.log(right_count)
                        + math.log(max(total, 1)),
                    )
                    if discounted > 0
                    else 0.0
                )
            association[pair] = value
        self.bigram_association = association
        # xixia_seg命名：left_adj是“后接字”，right_adj是“前接字”。
        self.right_entropy = self._neighbor_entropy(state.get("left_adj", {}))
        self.left_entropy = self._neighbor_entropy(state.get("right_adj", {}))
        self.gap_metric = metric

    def _lexicon_vectors(self, text: str) -> list[list[float]]:
        rows = [[0.0] * len(_DICT_NAMES) for _ in text]
        seen_begin = [-1.0] * len(text)
        seen_inside = [-1.0] * len(text)
        seen_end = [-1.0] * len(text)
        unseen_begin = [-1.0] * len(text)
        unseen_inside = [-1.0] * len(text)
        unseen_end = [-1.0] * len(text)
        for start, end, payload in self.lexicon.matches(text):
            word, info = payload
            length = end - start
            bucket = _length_bin(length)
            rows[start][_B_INDEX[bucket]] = 1.0
            rows[end - 1][_E_INDEX[bucket]] = 1.0
            if bucket != "2":
                for position in range(start + 1, end - 1):
                    rows[position][_I_INDEX[bucket]] = 1.0
            for position in range(start, end):
                rows[position][17] = max(rows[position][17], float(bool(info.get("has_yi"))))
                rows[position][18] = max(rows[position][18], float(bool(info.get("has_yin"))))
                rows[position][19] = max(
                    rows[position][19], float(bool(info.get("has_book_title")))
                )
            reliability = self.reliability.get(word)
            if reliability is not None:
                score = float(reliability.get("value", 0.0))
                observed = int(reliability.get("occ", 0)) > 0
            else:
                score = float(self.class_priors.get(bucket, 0.5))
                observed = False
            begin, inside, finish = (
                (seen_begin, seen_inside, seen_end)
                if observed
                else (unseen_begin, unseen_inside, unseen_end)
            )
            begin[start] = max(begin[start], score)
            finish[end - 1] = max(finish[end - 1], score)
            for position in range(start, end):
                inside[position] = max(inside[position], score)
        for position in range(len(text)):
            for index, value in (
                (11, seen_begin[position]),
                (12, seen_inside[position]),
                (13, seen_end[position]),
                (14, unseen_begin[position]),
                (15, unseen_inside[position]),
                (16, unseen_end[position]),
            ):
                if value >= 0:
                    rows[position][index] = value
        return rows

    def _gap_vectors(self, text: str) -> list[list[float]]:
        rows = [[0.0] * len(_GAP_NAMES) for _ in text]
        for position, character in enumerate(text):
            if position:
                pair = (text[position - 1], character)
                rows[position][0] = self.bigram_log_frequency.get(pair, 0.0)
                rows[position][1] = self.bigram_association.get(pair, 0.0)
                rows[position][2] = self.right_entropy.get(text[position - 1], 0.0)
                rows[position][3] = self.left_entropy.get(character, 0.0)
            if position + 1 < len(text):
                pair = (character, text[position + 1])
                rows[position][4] = self.bigram_log_frequency.get(pair, 0.0)
                rows[position][5] = self.bigram_association.get(pair, 0.0)
                rows[position][6] = self.right_entropy.get(character, 0.0)
                rows[position][7] = self.left_entropy.get(text[position + 1], 0.0)
        return rows

    def _features(self, text: str) -> list[dict[str, str | float]]:
        lexicon = self._lexicon_vectors(text)
        gap = self._gap_vectors(text)
        output: list[dict[str, str | float]] = []
        for position, character in enumerate(text):
            features: dict[str, str | float] = {
                "c0": character,
                "c-1": text[position - 1] if position else "<BOS>",
                "c+1": text[position + 1] if position + 1 < len(text) else "<EOS>",
                "c-2": text[position - 2] if position > 1 else "<BOS2>",
                "c+2": text[position + 2] if position + 2 < len(text) else "<EOS2>",
                "is_digit": str(character.isdigit()),
                "is_alpha": str(character.isalpha()),
                "is_punct": str(not character.isalnum() and not character.isspace()),
                "in_dict": str(character in self.word_set),
            }
            if position:
                features["c-1_c0"] = text[position - 1] + character
            if position + 1 < len(text):
                features["c0_c+1"] = character + text[position + 1]
            for name, value in zip(_DICT_NAMES, lexicon[position]):
                if value:
                    features[f"dict:{name}"] = float(value)
            for name, value in zip(_GAP_NAMES, gap[position]):
                if value:
                    features[f"gap:{name}"] = float(value)
            output.append(features)
        return output

    @staticmethod
    def _word_lengths(tags: list[str]) -> list[int]:
        lengths = [1] * len(tags)
        position = 0
        while position < len(tags):
            if tags[position] != "B":
                position += 1
                continue
            end = position + 1
            while end < len(tags) and tags[end] == "I":
                end += 1
            if end < len(tags) and tags[end] == "E":
                length = end - position + 1
                for index in range(position, end + 1):
                    lengths[index] = length
                position = end + 1
            else:
                position += 1
        return lengths

    def predict(self, text: str) -> FeatureMatrix:
        self.load()
        if not text:
            return ()
        features = self._features(text)
        marginals = self.model.predict_marginals_single(features)
        hard_tags = list(self.model.predict_single(features))
        lengths = self._word_lengths(hard_tags)
        rows: list[tuple[float, ...]] = []
        for position, probabilities in enumerate(marginals):
            left = {tag: float(probabilities.get(tag, 0.0)) for tag in _TAGS}
            right = (
                {tag: float(marginals[position + 1].get(tag, 0.0)) for tag in _TAGS}
                if position + 1 < len(text)
                else {tag: 0.0 for tag in _TAGS}
            )
            hard_boundary = hard_tags[position] in {"E", "S"} and (
                position + 1 == len(text) or hard_tags[position + 1] in {"B", "S"}
            )
            uncertainty = _entropy(left)
            if position + 1 < len(text):
                uncertainty = (uncertainty + _entropy(right)) / 2.0
            rows.append(
                (
                    left["E"],
                    left["S"],
                    right["B"],
                    right["S"],
                    float(hard_boundary),
                    uncertainty,
                    min(lengths[position], self.config.max_word_length)
                    / self.config.max_word_length,
                    (
                        min(lengths[position + 1], self.config.max_word_length)
                        / self.config.max_word_length
                        if position + 1 < len(text)
                        else 0.0
                    ),
                )
            )
        return tuple(rows)


class SegmentationKnowledgeProvider(KnowledgeFeatureProvider):
    """E4：冻结CRF分词器的可压缩软词界知识及精确重叠保护。"""

    def __init__(self, config: SegmentationKnowledgeConfig) -> None:
        self.config = config
        self.adapter = _FrozenCRFAdapter(config)
        self.overlap_trie = _StringTrie()
        self.annotation_sequences = 0
        self._loaded = False
        self._statistics: Counter[str] = Counter()

    @property
    def dimension(self) -> int:
        return self.config.dimension

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self.config.representation == "compact":
            return (
                "seg_boundary_probability",
                "seg_hard_boundary",
                "seg_boundary_confidence",
            )
        return (
            "seg_left_E_probability",
            "seg_left_S_probability",
            "seg_right_B_probability",
            "seg_right_S_probability",
            "seg_hard_boundary",
            "seg_bies_uncertainty",
            "seg_left_word_length",
            "seg_right_word_length",
        )

    def _represent(self, matrix: FeatureMatrix) -> FeatureMatrix:
        """压缩掉互相依赖的BIES分量与容易过拟合的预测词长。"""

        if self.config.representation == "full":
            return matrix
        rows: list[tuple[float, ...]] = []
        for row in matrix:
            left_end = row[0] + row[1]
            right_start = row[2] + row[3]
            # 文本末位没有右字符，此时只使用当前字符的E/S概率。
            boundary_probability = (
                0.5 * (left_end + right_start)
                if right_start > 0.0
                else left_end
            )
            rows.append(
                (
                    min(max(boundary_probability, 0.0), 1.0),
                    row[4],
                    1.0 - row[5],
                )
            )
        return tuple(rows)

    def _load(self) -> None:
        if self._loaded:
            return
        self.adapter.load()
        unique: set[str] = set()
        for path in self.config.annotation_paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                text = "".join(character for character in line if is_tangut(character))
                if len(text) >= self.config.min_overlap_length:
                    unique.add(text)
        for text in unique:
            self.overlap_trie.insert(text, text)
        self.annotation_sequences = len(unique)
        self._loaded = True
        LOGGER.info(
            "E4冻结分词器：CRF=%s，辞书=%s，gap=%s（%s）；人工分词重叠审计序列=%d，策略=%s，最短匹配=%d",
            self.config.model_path,
            self.config.lexicon_path,
            self.config.gap_path,
            self.adapter.gap_metric,
            self.annotation_sequences,
            self.config.overlap_policy,
            self.config.min_overlap_length,
        )

    def _overlap_mask(self, text: str) -> tuple[list[bool], int]:
        covered = [False] * len(text)
        matches = 0
        if self.config.overlap_policy == "allow":
            return covered, matches
        for start, end, _ in self.overlap_trie.matches(text):
            matches += 1
            covered[start:end] = [True] * (end - start)
        if matches and self.config.overlap_policy == "error":
            raise RuntimeError(
                f"E4检测到{matches}处与人工分词训练语料精确重叠；"
                "请使用mask_exact屏蔽，或仅在污染上界分析中显式使用allow"
            )
        return covered, matches

    def _run(self, text: str) -> FeatureMatrix:
        rows: list[tuple[float, ...]] = [(0.0,) * self.dimension for _ in text]
        position = 0
        while position < len(text):
            if not is_tangut(text[position]):
                position += 1
                continue
            end = position + 1
            while end < len(text) and is_tangut(text[end]):
                end += 1
            run = text[position:end]
            covered, matches = self._overlap_mask(run)
            self._statistics["精确重叠匹配数"] += matches
            self._statistics["重叠字符数"] += sum(covered)
            cursor = 0
            while cursor < len(run):
                if covered[cursor]:
                    cursor += 1
                    continue
                piece_end = cursor + 1
                while piece_end < len(run) and not covered[piece_end]:
                    piece_end += 1
                predicted = self._represent(
                    self.adapter.predict(run[cursor:piece_end])
                )
                rows[position + cursor : position + piece_end] = predicted
                cursor = piece_end
            # 一个间隔的左右任一字符属于重叠片段时，整行分词特征均屏蔽。
            for local in range(len(run)):
                if covered[local] or (local + 1 < len(run) and covered[local + 1]):
                    rows[position + local] = (0.0,) * self.dimension
                    self._statistics["屏蔽间隔数"] += 1
            position = end
        return tuple(rows)

    def _extract(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        self._load()
        by_block: dict[tuple[str, int, int], FeatureMatrix] = {}
        output: list[FeatureMatrix] = []
        for chunk in sequences:
            key = (chunk.document_id, chunk.block_start, chunk.block_end)
            matrix = by_block.get(key)
            if matrix is None:
                text = "".join(chunk.document_tokens[chunk.block_start : chunk.block_end])
                matrix = self._run(text)
                by_block[key] = matrix
            start = chunk.offset - chunk.block_start
            sliced = matrix[start : start + len(chunk.tokens)]
            output.append(sliced)
            self._statistics["间隔总数"] += len(sliced)
            self._statistics["有效分词特征间隔数"] += sum(any(row) for row in sliced)
        return output

    def fit_transform(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        if not sequences:
            raise ValueError("E4训练序列不能为空")
        self._statistics.clear()
        matrices = self._extract(sequences)
        total = self._statistics["间隔总数"]
        LOGGER.info(
            "E4软分词知识：维度=%d，训练有效覆盖率=%.2f%%，精确重叠=%d处，屏蔽间隔=%d（%.2f%%）",
            self.dimension,
            100.0 * self._statistics["有效分词特征间隔数"] / max(total, 1),
            self._statistics["精确重叠匹配数"],
            self._statistics["屏蔽间隔数"],
            100.0 * self._statistics["屏蔽间隔数"] / max(total, 1),
        )
        return matrices

    def transform(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        return self._extract(sequences)

    def metadata(self) -> dict[str, object]:
        total = self._statistics["间隔总数"]
        return {
            "名称": "冻结CRF软分词知识",
            "维度": self.dimension,
            "表示方式": self.config.representation,
            "融合方式": self.config.fusion,
            "特征名称": list(self.feature_names),
            "gap度量": self.adapter.gap_metric,
            "重叠策略": self.config.overlap_policy,
            "重叠审计最短字符数": self.config.min_overlap_length,
            "人工分词审计序列数": self.annotation_sequences,
            **dict(self._statistics),
            "有效覆盖率": self._statistics["有效分词特征间隔数"] / max(total, 1),
            "重叠屏蔽率": self._statistics["屏蔽间隔数"] / max(total, 1),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "format": "frozen_crf_segmentation_knowledge_v2",
            "resource_paths": {
                "model": str(self.config.model_path),
                "lexicon": str(self.config.lexicon_path),
                "gap": str(self.config.gap_path),
            },
            "resource_sha256": dict(self.adapter.resource_hashes),
            "overlap_policy": self.config.overlap_policy,
            "min_overlap_length": self.config.min_overlap_length,
            "max_word_length": self.config.max_word_length,
            "representation": self.config.representation,
            "fusion": self.config.fusion,
            "feature_names": list(self.feature_names),
        }
