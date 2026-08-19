from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tasks import BOUNDARY, OUTSIDE, Task


TANGUT_RANGES = ((0x17000, 0x18AFF), (0x18D00, 0x18D8F))


def is_tangut(character: str) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in TANGUT_RANGES)


def is_punctuation(character: str, missing_characters: set[str]) -> bool:
    """将非正文字符视作待恢复标点；缺字符方框始终是正文 token。"""
    if character in missing_characters or is_tangut(character):
        return False
    return unicodedata.category(character)[0] in {"P", "S"}


@dataclass(frozen=True)
class PreparedDocument:
    document_id: str
    volume_number: int
    tokens: tuple[str, ...]
    labels: tuple[str, ...]
    cut_offsets: tuple[int, ...] = ()
    domain: str = "unknown"
    source_path: str = ""
    source_line: int = 0
    feature_channels: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def annotated(
        self,
        labels: Iterable[str],
        feature_channels: Iterable[tuple[str, Iterable[str]]] = (),
    ) -> "PreparedDocument":
        """生成某一实验阶段的文献视图，不复制正文和切断边界。"""
        stage_labels = tuple(labels)
        if len(stage_labels) != len(self.tokens):
            raise ValueError(f"{self.document_id} 的阶段标签长度与正文不一致")
        channels = tuple(
            (name, tuple(values)) for name, values in feature_channels
        )
        for name, values in channels:
            if len(values) != len(self.tokens):
                raise ValueError(f"{self.document_id} 的特征通道 {name!r} 长度不一致")
        return PreparedDocument(
            document_id=self.document_id,
            volume_number=self.volume_number,
            tokens=self.tokens,
            labels=stage_labels,
            cut_offsets=self.cut_offsets,
            domain=self.domain,
            source_path=self.source_path,
            source_line=self.source_line,
            feature_channels=channels,
        )

    def chunks(self, max_length: int) -> list["SequenceChunk"]:
        if max_length <= 0:
            raise ValueError("max_length 必须大于 0")
        boundaries = (0, *self.cut_offsets, len(self.tokens))
        chunks: list[SequenceChunk] = []
        for block_start, block_end in zip(boundaries, boundaries[1:]):
            if block_start >= block_end:
                continue
            for start in range(block_start, block_end, max_length):
                end = min(start + max_length, block_end)
                chunks.append(
                    SequenceChunk(
                        document_id=self.document_id,
                        volume_number=self.volume_number,
                        tokens=self.tokens[start:end],
                        labels=self.labels[start:end],
                        offset=start,
                        document_length=len(self.tokens),
                        document_tokens=self.tokens,
                        block_start=block_start,
                        block_end=block_end,
                        domain=self.domain,
                        document_feature_channels=self.feature_channels,
                    )
                )
        return chunks


@dataclass(frozen=True)
class SequenceChunk:
    document_id: str
    volume_number: int
    tokens: tuple[str, ...]
    labels: tuple[str, ...]
    offset: int
    document_length: int
    document_tokens: tuple[str, ...]
    block_start: int
    block_end: int
    domain: str
    document_feature_channels: tuple[tuple[str, tuple[str, ...]], ...]


class CorpusReader:
    """读取“每卷一行”的文本，并构造字符后的间隙标签。"""

    def __init__(
        self,
        paths: Path | Iterable[Path],
        boundary_punctuation: Iterable[str],
        missing_characters: Iterable[str] = ("□", "@", "…"),
        missing_volume_numbers: Iterable[int] = (38,),
        ignored_editorial_symbols: Iterable[str] = ("√", "×", "△"),
        domains: Iterable[str] | None = None,
    ) -> None:
        if isinstance(paths, (str, Path)):
            self.paths = (Path(paths),)
        else:
            self.paths = tuple(Path(path) for path in paths)
        if not self.paths:
            raise ValueError("至少需要一个语料文件")
        if domains is None:
            self.domains = tuple("unknown" for _ in self.paths)
        else:
            self.domains = tuple(domains)
            if len(self.domains) != len(self.paths):
                raise ValueError("domains 数量必须与语料文件数量一致")
        self.boundary_punctuation = set(boundary_punctuation)
        self.missing_characters = set(missing_characters)
        self.missing_volume_numbers = set(missing_volume_numbers)
        self.ignored_editorial_symbols = set(ignored_editorial_symbols)

    def read(self, task: Task) -> list[PreparedDocument]:
        documents: list[PreparedDocument] = []
        used_source_ids: set[str] = set()
        for source_index, (path, domain) in enumerate(zip(self.paths, self.domains), 1):
            source_id = path.stem
            if source_id in used_source_ids:
                source_id = f"{source_id}_{source_index}"
            used_source_ids.add(source_id)
            lines = path.read_text(encoding="utf-8").splitlines()
            volume_number = 0
            for line_index, line in enumerate(lines, 1):
                volume_number += 1
                while volume_number in self.missing_volume_numbers:
                    volume_number += 1
                if not line:
                    continue
                tokens, labels, cut_offsets = self._encode(line, task)
                if not tokens:
                    raise ValueError(
                        f"{path} 第 {line_index} 行（卷 {volume_number}）没有正文字符"
                    )
                documents.append(
                    PreparedDocument(
                        document_id=f"{source_id}_volume_{volume_number:03d}",
                        volume_number=volume_number,
                        tokens=tuple(tokens),
                        labels=tuple(labels),
                        cut_offsets=tuple(cut_offsets),
                        domain=domain,
                        source_path=str(path),
                        source_line=line_index,
                    )
                )
        return documents

    def _encode(self, text: str, task: Task) -> tuple[list[str], list[str], list[int]]:
        tokens: list[str] = []
        punctuation_after: list[str] = []
        cut_offsets: list[int] = []
        leading_punctuation = ""
        block_token_start = 0
        for character in text:
            if character == "\t":
                offset = len(tokens)
                if offset and (not cut_offsets or cut_offsets[-1] != offset):
                    cut_offsets.append(offset)
                leading_punctuation = ""
                block_token_start = offset
                continue
            if character.isspace():
                continue
            if character in self.ignored_editorial_symbols:
                continue
            if is_punctuation(character, self.missing_characters):
                if len(tokens) > block_token_start:
                    if punctuation_after[-1].startswith("^") and "|" not in punctuation_after[-1]:
                        punctuation_after[-1] += "|" + character
                    else:
                        punctuation_after[-1] += character
                else:
                    leading_punctuation += character
                continue
            tokens.append(character)
            # 极少见的文献首/TAB 块首标点附着到首字标签，
            # 以 ^ 标记其在字前，避免跨越删除片段附到前一块。
            punctuation_after.append(
                "^" + leading_punctuation if leading_punctuation else ""
            )
            leading_punctuation = ""

        if task is Task.BOUNDARY:
            labels = [
                BOUNDARY if any(mark in self.boundary_punctuation for mark in punctuation) else OUTSIDE
                for punctuation in punctuation_after
            ]
        else:
            labels = [punctuation or OUTSIDE for punctuation in punctuation_after]
        if cut_offsets and cut_offsets[-1] == len(tokens):
            cut_offsets.pop()
        return tokens, labels, cut_offsets
