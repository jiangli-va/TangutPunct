from __future__ import annotations

import hashlib
import json
import logging
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from config import POSKnowledgeConfig, POSRelationKnowledgeConfig
from data.corpus import SequenceChunk, is_tangut

from .base import FeatureMatrix, KnowledgeFeatureProvider


LOGGER = logging.getLogger(__name__)

_DICT_NAMES = (
    "B2", "B3", "B4", "B5P",
    "I3", "I4", "I5P",
    "E2", "E3", "E4", "E5P",
    "rel_seen_B", "rel_seen_I", "rel_seen_E",
    "rel_unseen_B", "rel_unseen_I", "rel_unseen_E",
    "has_yi", "has_yin", "has_book_title",
)
_B_INDEX = {"2": 0, "3": 1, "4": 2, "5P": 3}
_I_INDEX = {"3": 4, "4": 5, "5P": 6}
_E_INDEX = {"2": 7, "3": 8, "4": 9, "5P": 10}

# 顺序就是模型输入中七组概率的顺序。每个原始标签恰好属于一组。
POS_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    ("nominal", frozenset({"n", "nb", "nc", "nh", "nl", "no", "ns", "t"})),
    ("predicate", frozenset({"v", "a", "b", "l"})),
    ("adverbial", frozenset({"d"})),
    ("pronominal", frozenset({"r", "rd", "ri", "rp"})),
    ("functional", frozenset({"u", "c", "p"})),
    ("quantity", frozenset({"m", "mc", "mo", "q"})),
    (
        "grammatical",
        frozenset(
            {
                "Dir1.", "Dir2.", "Erg.", "Loc.", "Nom.", "Obj.",
                "Fut.", "Pfv.", "Quot.", "1sg.", "2sg.", "pl.",
            }
        ),
    ),
)
_POS_TO_GROUP = {
    tag: index
    for index, (_, tags) in enumerate(POS_GROUPS)
    for tag in tags
}
FINE_POS_LABELS: tuple[str, ...] = tuple(sorted(_POS_TO_GROUP))
_FINE_POS_INDEX = {label: index for index, label in enumerate(FINE_POS_LABELS)}
_BIES_LABELS = ("B", "I", "E", "S")
_BIES_INDEX = {label: index for index, label in enumerate(_BIES_LABELS)}
POS_RELATION_SCALAR_DIMENSION = 4
POS_RELATION_DIMENSION = len(FINE_POS_LABELS) * 2 + POS_RELATION_SCALAR_DIMENSION


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _length_bin(length: int) -> str:
    if length <= 2:
        return "2"
    if length == 3:
        return "3"
    if length == 4:
        return "4"
    return "5P"


class _StringTrie:
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


class _FrozenJointPOSAdapter:
    """只读恢复CRF-Joint-full，并在本工程复现其字符特征模板。"""

    def __init__(self, config: POSKnowledgeConfig) -> None:
        self.config = config
        self.model: Any | None = None
        self.word_set: set[str] = set()
        self.known_characters: set[str] = set()
        self.lexicon = _StringTrie()
        self.reliability: dict[str, dict[str, Any]] = {}
        self.class_priors: dict[str, float] = {}
        self.pos_labels: tuple[str, ...] = ()
        self.annotation_path_from_manifest: Path | None = None
        self.resource_hashes: dict[str, str] = {}
        self.loaded = False

    def _check_manifest_hash(
        self, path: Path, expected: str | None, resource_name: str
    ) -> str:
        actual = _file_sha256(path)
        if expected and actual.lower() != expected.lower():
            raise RuntimeError(
                f"E5{resource_name}的SHA256与manifest不一致：{path}"
            )
        return actual

    def load(self) -> None:
        if self.loaded:
            return
        if not self.config.model_path or not self.config.lexicon_state_path:
            raise ValueError("E5冻结词性资源路径不完整")
        if not self.config.manifest_path:
            raise ValueError("E5缺少POS资源manifest")
        try:
            import joblib
        except ImportError as error:
            raise RuntimeError(
                "E5加载CRF-Joint-full需要joblib/sklearn-crfsuite环境"
            ) from error

        manifest = json.loads(
            self.config.manifest_path.read_text(encoding="utf-8")
        )
        if manifest.get("格式") != "tangut_pos_crf_joint_full_manifest_v1":
            raise ValueError("E5 POS manifest格式不受支持")
        products = manifest.get("产物", {})
        resources = manifest.get("资源", {})
        model_hash = self._check_manifest_hash(
            self.config.model_path,
            products.get("模型SHA256"),
            "词性模型",
        )
        lexicon_hash = self._check_manifest_hash(
            self.config.lexicon_state_path,
            products.get("词典状态SHA256"),
            "词典状态",
        )

        payload = joblib.load(self.config.model_path)
        if (
            not isinstance(payload, dict)
            or payload.get("format") != "tangut_pos_crf_joint_full_v1"
            or "model" not in payload
        ):
            raise ValueError("E5 POS模型不是CRF-Joint-full导出包")
        if int(payload.get("dict_feature_level", -1)) != 5:
            raise ValueError("E5当前只支持词典特征级别5的CRF-Joint-full")
        if int(payload.get("gap_feature_level", -1)) != 0:
            raise ValueError("E5当前冻结POS模型不应包含无标注gap统计")
        if list(payload.get("dict_feature_indices", ())) != list(range(20)):
            raise ValueError("E5 POS模型的词典特征列与预期20维模板不一致")
        self.model = payload["model"]
        if not hasattr(self.model, "predict_marginals_single"):
            raise ValueError("E5词性CRF不支持联合标签边缘概率输出")
        self.word_set = set(payload.get("word_set", ()))
        self.known_characters = {
            character for word in self.word_set for character in word
        }
        self.pos_labels = tuple(payload.get("pos_labels", ()))
        unmapped = sorted(set(self.pos_labels) - set(_POS_TO_GROUP))
        missing = sorted(set(_POS_TO_GROUP) - set(self.pos_labels))
        if unmapped or missing:
            raise ValueError(
                "E5七组词性映射与冻结模型不一致："
                f"未映射={unmapped}，模型缺失={missing}"
            )

        with self.config.lexicon_state_path.open("rb") as stream:
            state = pickle.load(stream)
        for word, info in state.get("trie_entries", ()):
            self.lexicon.insert(word, (word, info))
        self.reliability = dict(state.get("reliability", {}))
        self.class_priors = dict(state.get("class_priors", {}))

        annotation_value = resources.get("监督语料")
        if annotation_value:
            self.annotation_path_from_manifest = Path(annotation_value).resolve()
            expected = resources.get("监督语料SHA256")
            if self.annotation_path_from_manifest.exists():
                self._check_manifest_hash(
                    self.annotation_path_from_manifest, expected, "人工词性语料"
                )
        self.resource_hashes = {
            "model": model_hash,
            "lexicon_state": lexicon_hash,
            "manifest": _file_sha256(self.config.manifest_path),
        }
        self.loaded = True

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
            bucket = _length_bin(end - start)
            rows[start][_B_INDEX[bucket]] = 1.0
            rows[end - 1][_E_INDEX[bucket]] = 1.0
            if bucket != "2":
                for position in range(start + 1, end - 1):
                    rows[position][_I_INDEX[bucket]] = 1.0
            for position in range(start, end):
                rows[position][17] = max(
                    rows[position][17], float(bool(info.get("has_yi")))
                )
                rows[position][18] = max(
                    rows[position][18], float(bool(info.get("has_yin")))
                )
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

    def _features(self, text: str) -> list[dict[str, str | float]]:
        lexicon = self._lexicon_vectors(text)
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
            output.append(features)
        return output

    def predict_group_probabilities(self, text: str) -> tuple[tuple[float, ...], ...]:
        self.load()
        if not text:
            return ()
        marginals = self.model.predict_marginals_single(self._features(text))
        rows: list[tuple[float, ...]] = []
        for probabilities in marginals:
            groups = [0.0] * len(POS_GROUPS)
            for label, probability in probabilities.items():
                if "-" not in label:
                    raise ValueError(f"E5遇到非法联合词性标签：{label!r}")
                _, pos = label.split("-", 1)
                group = _POS_TO_GROUP.get(pos)
                if group is None:
                    raise ValueError(f"E5遇到未映射词性标签：{pos!r}")
                groups[group] += float(probability)
            total = sum(groups)
            if total <= 0:
                raise RuntimeError("E5词性CRF返回了全零边缘概率")
            rows.append(tuple(value / total for value in groups))
        return tuple(rows)

    def predict_joint_probabilities(self, text: str) -> tuple[tuple[float, ...], ...]:
        """返回固定顺序的BIES×36类联合边缘概率，不做七组压缩。"""

        self.load()
        if not text:
            return ()
        marginals = self.model.predict_marginals_single(self._features(text))
        row_dimension = len(_BIES_LABELS) * len(FINE_POS_LABELS)
        rows: list[tuple[float, ...]] = []
        for probabilities in marginals:
            row = [0.0] * row_dimension
            for label, probability in probabilities.items():
                if "-" not in label:
                    raise ValueError(f"E7/E8遇到非法联合词性标签：{label!r}")
                boundary, pos = label.split("-", 1)
                boundary_index = _BIES_INDEX.get(boundary)
                pos_index = _FINE_POS_INDEX.get(pos)
                if boundary_index is None or pos_index is None:
                    raise ValueError(f"E7/E8遇到未知联合词性标签：{label!r}")
                row[boundary_index * len(FINE_POS_LABELS) + pos_index] += float(
                    probability
                )
            total = sum(row)
            if total <= 0:
                raise RuntimeError("E7/E8词性CRF返回了全零联合边缘概率")
            rows.append(tuple(value / total for value in row))
        return tuple(rows)


class POSKnowledgeProvider(KnowledgeFeatureProvider):
    """E5：把36类联合词性边缘概率聚合成字符级七组软知识。"""

    # POS模型与外层五折无关；同一进程中两个阶段、五个折共享冻结推理结果。
    # key含模型/词典/人工标注哈希和完整TAB块文本哈希，不会跨资源误复用。
    _GLOBAL_BLOCK_CACHE: dict[
        tuple[str, str], tuple[FeatureMatrix, dict[str, int]]
    ] = {}

    def __init__(self, config: POSKnowledgeConfig) -> None:
        self.config = config
        self.adapter = _FrozenJointPOSAdapter(config)
        self.overlap_trie = _StringTrie()
        self.annotation_sequences = 0
        self._loaded = False
        self._statistics: Counter[str] = Counter()
        self._cache_signature = ""

    @property
    def dimension(self) -> int:
        return self.config.raw_dimension

    @property
    def feature_names(self) -> tuple[str, ...]:
        groups = tuple(name for name, _ in POS_GROUPS)
        if self.config.representation == "coarse_soft_character":
            return (
                *(f"pos_self_{name}_probability" for name in groups),
                "pos_self_confidence",
            )
        return (
            *(f"pos_left_{name}_probability" for name in groups),
            *(f"pos_right_{name}_probability" for name in groups),
            "pos_left_confidence",
            "pos_right_confidence",
            "pos_transition_strength",
        )

    @staticmethod
    def _confidence(probabilities: tuple[float, ...]) -> float:
        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0
        )
        return min(max(1.0 - entropy / math.log(len(POS_GROUPS)), 0.0), 1.0)

    @staticmethod
    def _gap_row(
        left: tuple[float, ...], right: tuple[float, ...] | None
    ) -> tuple[float, ...]:
        left_confidence = POSKnowledgeProvider._confidence(left)
        if right is None:
            return (*left, *(0.0 for _ in POS_GROUPS), left_confidence, 0.0, 0.0)
        right_confidence = POSKnowledgeProvider._confidence(right)
        transition = 1.0 - sum(a * b for a, b in zip(left, right))
        return (
            *left,
            *right,
            left_confidence,
            right_confidence,
            min(max(transition, 0.0), 1.0),
        )

    @staticmethod
    def _character_row(probabilities: tuple[float, ...]) -> tuple[float, ...]:
        """当前字符的七组软词性概率及其自身置信度。"""

        return (*probabilities, POSKnowledgeProvider._confidence(probabilities))

    def _annotation_paths(self) -> tuple[Path, ...]:
        if self.config.annotation_paths:
            return self.config.annotation_paths
        path = self.adapter.annotation_path_from_manifest
        return (path,) if path is not None and path.exists() else ()

    def _load(self) -> None:
        if self._loaded:
            return
        self.adapter.load()
        paths = self._annotation_paths()
        if self.config.overlap_policy != "allow" and not paths:
            raise ValueError("E5重叠保护未找到人工词性标注语料")
        unique: set[str] = set()
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                text = "".join(character for character in line if is_tangut(character))
                if len(text) >= self.config.min_overlap_length:
                    unique.add(text)
        for text in unique:
            self.overlap_trie.insert(text, text)
        self.annotation_sequences = len(unique)
        annotation_hashes = {
            str(path.resolve()): _file_sha256(path) for path in paths
        }
        signature_payload = {
            "resources": self.adapter.resource_hashes,
            "annotations": annotation_hashes,
            "overlap_policy": self.config.overlap_policy,
            "min_overlap_length": self.config.min_overlap_length,
            "representation": self.config.representation,
        }
        self._cache_signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self._loaded = True
        if type(self).__name__ == "POSRelationKnowledgeProvider":
            LOGGER.info(
                "E7/E8冻结词性资源：CRF-Joint-full=%s，词典状态=%s；"
                "保留BIES×36类联合边缘概率；人工词性重叠审计序列=%d，"
                "策略=%s，最短精确匹配=%d",
                self.config.model_path,
                self.config.lexicon_state_path,
                self.annotation_sequences,
                self.config.overlap_policy,
                self.config.min_overlap_length,
            )
        else:
            LOGGER.info(
                "E5冻结词性模型：CRF-Joint-full=%s，词典状态=%s；36类→7组；"
                "人工词性重叠审计序列=%d，策略=%s，最短精确匹配=%d",
                self.config.model_path,
                self.config.lexicon_state_path,
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
                f"E5检测到{matches}处与人工词性训练语料精确重叠；"
                "正式实验应消除重叠，或使用mask_exact屏蔽"
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
            probabilities: list[tuple[float, ...] | None] = [None] * len(run)
            cursor = 0
            while cursor < len(run):
                if covered[cursor]:
                    cursor += 1
                    continue
                piece_end = cursor + 1
                while piece_end < len(run) and not covered[piece_end]:
                    piece_end += 1
                predicted = self.adapter.predict_group_probabilities(
                    run[cursor:piece_end]
                )
                probabilities[cursor:piece_end] = predicted
                cursor = piece_end
            for local, left in enumerate(probabilities):
                global_position = position + local
                if left is None:
                    continue
                if self.config.representation == "coarse_soft_character":
                    # 特征行与当前字符一一对应；相邻POS关系交由双向BiLSTM学习。
                    rows[global_position] = self._character_row(left)
                    continue
                if local + 1 < len(run):
                    right = probabilities[local + 1]
                    if right is None:
                        self._statistics["屏蔽间隔数"] += 1
                        continue
                    rows[global_position] = self._gap_row(left, right)
                elif global_position + 1 == len(text):
                    # TAB块末保留左字符词性；不存在的右侧分布固定为零。
                    rows[global_position] = self._gap_row(left, None)
                # 若run因□/@/…结束，间隔左右涉及缺字，整行保持全零。
            position = end
        return tuple(rows)

    def _extract(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        self._load()
        output: list[FeatureMatrix] = []
        by_block: dict[tuple[str, int, int, str], FeatureMatrix] = {}
        inferred_blocks = 0
        shared_cache_hits = 0
        for chunk in sequences:
            text = "".join(
                chunk.document_tokens[chunk.block_start : chunk.block_end]
            )
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            local_key = (
                chunk.document_id,
                chunk.block_start,
                chunk.block_end,
                text_hash,
            )
            cache_key = (self._cache_signature, text_hash)
            matrix = by_block.get(local_key)
            if matrix is None:
                cached = self._GLOBAL_BLOCK_CACHE.get(cache_key)
                if cached is None:
                    inferred_blocks += 1
                    LOGGER.debug(
                        "E5冻结POS推理：%s TAB块[%d:%d]，%d字（本次新计算第%d块）",
                        chunk.document_id,
                        chunk.block_start,
                        chunk.block_end,
                        len(text),
                        inferred_blocks,
                    )
                    before = {
                        name: self._statistics[name]
                        for name in (
                            "精确重叠匹配数",
                            "重叠字符数",
                            "屏蔽间隔数",
                        )
                    }
                    matrix = self._run(text)
                    audit_delta = {
                        name: self._statistics[name] - before[name]
                        for name in before
                    }
                    self._GLOBAL_BLOCK_CACHE[cache_key] = (matrix, audit_delta)
                else:
                    shared_cache_hits += 1
                    matrix, audit_delta = cached
                    self._statistics.update(audit_delta)
                by_block[local_key] = matrix
            start = chunk.offset - chunk.block_start
            sliced = matrix[start : start + len(chunk.tokens)]
            output.append(sliced)
            self._statistics["间隔总数"] += len(sliced)
            self._statistics["有效词性特征间隔数"] += sum(any(row) for row in sliced)
            self._statistics["词性词表内字符数"] += sum(
                character in self.adapter.known_characters for character in chunk.tokens
            )
            self._statistics["西夏字符数"] += sum(
                is_tangut(character) for character in chunk.tokens
            )
        LOGGER.debug(
            "E5冻结POS批次完成：输入神经切块=%d，TAB块=%d，新推理=%d，"
            "跨折/阶段共享缓存命中=%d",
            len(sequences),
            len(by_block),
            inferred_blocks,
            shared_cache_hits,
        )
        return output

    def fit_transform(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        if not sequences:
            raise ValueError("E5训练序列不能为空")
        self._statistics.clear()
        matrices = self._extract(sequences)
        total = self._statistics["间隔总数"]
        tangut = self._statistics["西夏字符数"]
        LOGGER.info(
            "E5软词性知识：原始%d维，训练有效覆盖率=%.2f%%，"
            "词性模型字符覆盖率=%.2f%%，精确重叠=%d处",
            self.dimension,
            100.0 * self._statistics["有效词性特征间隔数"] / max(total, 1),
            100.0 * self._statistics["词性词表内字符数"] / max(tangut, 1),
            self._statistics["精确重叠匹配数"],
        )
        return matrices

    def transform(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        return self._extract(sequences)

    def metadata(self) -> dict[str, object]:
        total = self._statistics["间隔总数"]
        tangut = self._statistics["西夏字符数"]
        return {
            "名称": "冻结CRF-Joint-full七组软词性知识",
            "原始维度": self.dimension,
            "投影维度": self.config.projection_dimension,
            "表示方式": self.config.representation,
            "融合方式": self.config.fusion,
            "词性分组": {
                name: sorted(tags) for name, tags in POS_GROUPS
            },
            "特征名称": list(self.feature_names),
            "整通道dropout": self.config.channel_dropout,
            "重叠策略": self.config.overlap_policy,
            "重叠审计最短字符数": self.config.min_overlap_length,
            "人工词性审计序列数": self.annotation_sequences,
            **dict(self._statistics),
            "有效覆盖率": self._statistics["有效词性特征间隔数"] / max(total, 1),
            "字符覆盖率": self._statistics["词性词表内字符数"] / max(tangut, 1),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "format": "frozen_crf_joint_pos_knowledge_v2",
            "resource_paths": {
                "model": str(self.config.model_path),
                "lexicon_state": str(self.config.lexicon_state_path),
                "manifest": str(self.config.manifest_path),
            },
            "resource_sha256": dict(self.adapter.resource_hashes),
            "representation": self.config.representation,
            "fusion": self.config.fusion,
            "group_scheme": self.config.group_scheme,
            "groups": {
                name: sorted(tags) for name, tags in POS_GROUPS
            },
            "raw_dimension": self.dimension,
            "projection_dimension": self.config.projection_dimension,
            "channel_dropout": self.config.channel_dropout,
            "overlap_policy": self.config.overlap_policy,
            "min_overlap_length": self.config.min_overlap_length,
            "feature_names": list(self.feature_names),
        }


class POSRelationKnowledgeProvider(POSKnowledgeProvider):
    """E7/E8：字符后位置的左词结束POS、右词开始POS及边界置信度。"""

    # 与E5粗粒度表示隔离，防止相同文本哈希误复用不同维度的缓存。
    _GLOBAL_BLOCK_CACHE: dict[
        tuple[str, str], tuple[FeatureMatrix, dict[str, int]]
    ] = {}

    def __init__(
        self,
        resource_config: POSKnowledgeConfig,
        relation_config: POSRelationKnowledgeConfig,
    ) -> None:
        super().__init__(resource_config)
        self.relation_config = relation_config

    @property
    def dimension(self) -> int:
        return POS_RELATION_DIMENSION

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            *(f"pos_left_end_{tag}_probability" for tag in FINE_POS_LABELS),
            *(f"pos_right_start_{tag}_probability" for tag in FINE_POS_LABELS),
            "pos_left_word_end_probability",
            "pos_right_word_start_probability",
            "pos_left_entropy",
            "pos_right_entropy",
        )

    @staticmethod
    def _normalized_distribution(
        values: list[float],
    ) -> tuple[tuple[float, ...], float, float]:
        total = sum(values)
        if total <= 0:
            return (0.0,) * len(values), 0.0, 0.0
        distribution = tuple(value / total for value in values)
        entropy = -sum(
            probability * math.log(probability)
            for probability in distribution
            if probability > 0
        ) / math.log(len(distribution))
        return distribution, min(max(total, 0.0), 1.0), min(max(entropy, 0.0), 1.0)

    @staticmethod
    def _left_end(joint: tuple[float, ...]) -> list[float]:
        count = len(FINE_POS_LABELS)
        end_offset = _BIES_INDEX["E"] * count
        single_offset = _BIES_INDEX["S"] * count
        return [
            joint[end_offset + index] + joint[single_offset + index]
            for index in range(count)
        ]

    @staticmethod
    def _right_start(joint: tuple[float, ...]) -> list[float]:
        count = len(FINE_POS_LABELS)
        begin_offset = _BIES_INDEX["B"] * count
        single_offset = _BIES_INDEX["S"] * count
        return [
            joint[begin_offset + index] + joint[single_offset + index]
            for index in range(count)
        ]

    @classmethod
    def _relation_row(
        cls,
        left_joint: tuple[float, ...],
        right_joint: tuple[float, ...] | None,
    ) -> tuple[float, ...]:
        left, end_probability, left_entropy = cls._normalized_distribution(
            cls._left_end(left_joint)
        )
        if right_joint is None:
            right = (0.0,) * len(FINE_POS_LABELS)
            start_probability = 0.0
            right_entropy = 0.0
        else:
            right, start_probability, right_entropy = cls._normalized_distribution(
                cls._right_start(right_joint)
            )
        return (
            *left,
            *right,
            end_probability,
            start_probability,
            left_entropy,
            right_entropy,
        )

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
            probabilities: list[tuple[float, ...] | None] = [None] * len(run)
            cursor = 0
            while cursor < len(run):
                if covered[cursor]:
                    cursor += 1
                    continue
                piece_end = cursor + 1
                while piece_end < len(run) and not covered[piece_end]:
                    piece_end += 1
                predicted = self.adapter.predict_joint_probabilities(
                    run[cursor:piece_end]
                )
                probabilities[cursor:piece_end] = predicted
                cursor = piece_end
            for local, left_joint in enumerate(probabilities):
                if left_joint is None:
                    continue
                global_position = position + local
                if local + 1 < len(run):
                    right_joint = probabilities[local + 1]
                    if right_joint is None:
                        self._statistics["屏蔽间隔数"] += 1
                        continue
                    rows[global_position] = self._relation_row(
                        left_joint, right_joint
                    )
                elif global_position + 1 == len(text):
                    # TAB块末仍保留左侧细词性；右词及其边界置信度为零。
                    rows[global_position] = self._relation_row(left_joint, None)
                # 若连续西夏字因缺字结束，不跨缺字构造左右关系。
            position = end
        return tuple(rows)

    def fit_transform(self, sequences: list[SequenceChunk]) -> list[FeatureMatrix]:
        if not sequences:
            raise ValueError("E7/E8训练序列不能为空")
        self._statistics.clear()
        matrices = self._extract(sequences)
        total = self._statistics["间隔总数"]
        tangut = self._statistics["西夏字符数"]
        LOGGER.info(
            "E7/E8细粒度词性关系：BIES×%d类→左右词性分布＋4维边界统计，"
            "原始%d维，训练有效覆盖率=%.2f%%，词性模型字符覆盖率=%.2f%%，"
            "精确重叠=%d处",
            len(FINE_POS_LABELS),
            self.dimension,
            100.0 * self._statistics["有效词性特征间隔数"] / max(total, 1),
            100.0 * self._statistics["词性词表内字符数"] / max(tangut, 1),
            self._statistics["精确重叠匹配数"],
        )
        return matrices

    def metadata(self) -> dict[str, object]:
        total = self._statistics["间隔总数"]
        tangut = self._statistics["西夏字符数"]
        return {
            "名称": "冻结CRF-Joint-full细粒度左右词性关系",
            "原始维度": self.dimension,
            "细粒度词性数": len(FINE_POS_LABELS),
            "细粒度词性顺序": list(FINE_POS_LABELS),
            "表示方式": "left_end_and_right_start_from_bies_pos_marginals",
            "特征名称": list(self.feature_names),
            "重叠策略": self.config.overlap_policy,
            "重叠审计最短字符数": self.config.min_overlap_length,
            "人工词性审计序列数": self.annotation_sequences,
            **dict(self._statistics),
            "有效覆盖率": self._statistics["有效词性特征间隔数"] / max(total, 1),
            "字符覆盖率": self._statistics["词性词表内字符数"] / max(tangut, 1),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "format": "frozen_crf_joint_pos_relation_v1",
            "resource_paths": {
                "model": str(self.config.model_path),
                "lexicon_state": str(self.config.lexicon_state_path),
                "manifest": str(self.config.manifest_path),
            },
            "resource_sha256": dict(self.adapter.resource_hashes),
            "fine_pos_labels": list(FINE_POS_LABELS),
            "raw_dimension": self.dimension,
            "overlap_policy": self.config.overlap_policy,
            "min_overlap_length": self.config.min_overlap_length,
            "feature_names": list(self.feature_names),
        }
