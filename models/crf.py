from __future__ import annotations

from pathlib import Path
import logging

from data.corpus import SequenceChunk
from features.base import FeatureExtractor
from .base import SequenceTagger


LOGGER = logging.getLogger(__name__)


class CRFTagger(SequenceTagger):
    def __init__(self, feature_extractor: FeatureExtractor, **parameters: object) -> None:
        try:
            import sklearn_crfsuite
        except ImportError as error:
            raise RuntimeError(
                "缺少 sklearn-crfsuite；请运行 pip install -r requirements.txt"
            ) from error
        self.feature_extractor = feature_extractor
        # sklearn-crfsuite 0.3.x 的 verbose 训练器在某些 L-BFGS 终止状态会
        # 触发 trainer.py 异常；调试信息由工程日志层稳定输出。
        self.model = sklearn_crfsuite.CRF(algorithm="lbfgs", **parameters)

    def fit(self, train: list[SequenceChunk], dev: list[SequenceChunk] | None = None) -> None:
        # sklearn-crfsuite 的 LBFGS 无早停接口；dev 保留给调参器和后续模型使用。
        LOGGER.debug("正在为 %d 个训练序列块抽取特征", len(train))
        x_train = [self.feature_extractor.transform(sequence) for sequence in train]
        y_train = [list(sequence.labels) for sequence in train]
        LOGGER.debug("特征抽取完成，调用 CRFsuite/L-BFGS")
        self.model.fit(x_train, y_train)

    def predict(self, sequences: list[SequenceChunk]) -> list[list[str]]:
        LOGGER.debug("正在为 %d 个测试序列块抽取特征", len(sequences))
        features = [self.feature_extractor.transform(sequence) for sequence in sequences]
        return self.model.predict(features)

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
