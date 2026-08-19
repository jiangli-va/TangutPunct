from __future__ import annotations

from dataclasses import dataclass

from data.corpus import SequenceChunk
from .base import FeatureExtractor


@dataclass(frozen=True)
class BasicCharacterFeatures(FeatureExtractor):
    missing_characters: frozenset[str] = frozenset({"□"})
    window: int = 3
    use_ngrams: bool = True
    use_missing: bool = True
    use_document_edges: bool = True
    use_stage_features: bool = True
    stage_feature_window: int = 1

    def transform(self, sequence: SequenceChunk) -> list[dict[str, object]]:
        return [self._at(index, sequence) for index in range(len(sequence.tokens))]

    def _at(
        self, index: int, sequence: SequenceChunk
    ) -> dict[str, object]:
        tokens = sequence.document_tokens
        absolute_index = sequence.offset + index
        features: dict[str, object] = {"bias": 1.0, "char": tokens[absolute_index]}
        for distance in range(1, self.window + 1):
            left = absolute_index - distance
            right = absolute_index + distance
            features[f"char-{distance}"] = self._context_character(
                tokens, left, sequence.block_start, sequence.block_end, distance, True
            )
            features[f"char+{distance}"] = self._context_character(
                tokens, right, sequence.block_start, sequence.block_end, distance, False
            )

        if self.use_ngrams:
            # 相邻二元组合，以及覆盖当前字的连续三元组合。
            for start_delta, name in ((-1, "bi-1:0"), (0, "bi0:+1")):
                features[name] = self._ngram(
                    tokens,
                    absolute_index + start_delta,
                    2,
                    sequence.block_start,
                    sequence.block_end,
                )
            for start_delta, name in ((-2, "tri-2:0"), (-1, "tri-1:+1"), (0, "tri0:+2")):
                features[name] = self._ngram(
                    tokens,
                    absolute_index + start_delta,
                    3,
                    sequence.block_start,
                    sequence.block_end,
                )

        if self.use_missing:
            features["is_missing"] = tokens[absolute_index] in self.missing_characters
            for distance in range(1, self.window + 1):
                for sign in (-1, 1):
                    neighbour = absolute_index + sign * distance
                    key = f"missing{sign * distance:+d}"
                    features[key] = (
                        sequence.block_start <= neighbour < sequence.block_end
                        and tokens[neighbour] in self.missing_characters
                    )

        if self.use_document_edges:
            features["document_bos"] = absolute_index == 0
            features["document_eos"] = absolute_index == sequence.document_length - 1
            features["cut_bos"] = (
                absolute_index == sequence.block_start and sequence.block_start > 0
            )
            features["cut_eos"] = (
                absolute_index == sequence.block_end - 1
                and sequence.block_end < sequence.document_length
            )

        if self.use_stage_features:
            for channel_name, values in sequence.document_feature_channels:
                features[f"stage:{channel_name}:0"] = values[absolute_index]
                for distance in range(1, self.stage_feature_window + 1):
                    for sign in (-1, 1):
                        neighbour = absolute_index + sign * distance
                        value = (
                            values[neighbour]
                            if sequence.block_start <= neighbour < sequence.block_end
                            else "<CUT>"
                        )
                        features[
                            f"stage:{channel_name}:{sign * distance:+d}"
                        ] = value
        return features

    @staticmethod
    def _context_character(
        tokens: tuple[str, ...],
        index: int,
        block_start: int,
        block_end: int,
        distance: int,
        is_left: bool,
    ) -> str:
        if index < block_start:
            prefix = "BOS" if block_start == 0 else "CUT_LEFT"
            return f"<{prefix}{distance}>"
        if index >= block_end:
            prefix = "EOS" if block_end == len(tokens) else "CUT_RIGHT"
            return f"<{prefix}{distance}>"
        return tokens[index]

    @staticmethod
    def _ngram(
        tokens: tuple[str, ...],
        start: int,
        length: int,
        block_start: int,
        block_end: int,
    ) -> str:
        values = []
        for index in range(start, start + length):
            if index < block_start:
                values.append("<BOS>" if block_start == 0 else "<CUT_LEFT>")
            elif index >= block_end:
                values.append("<EOS>" if block_end == len(tokens) else "<CUT_RIGHT>")
            else:
                values.append(tokens[index])
        return "|".join(values)
