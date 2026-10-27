"""Outbound port: what the domain needs from any mobility-prediction engine."""

from abc import ABC, abstractmethod

from src.domain.entities.character import Character
from src.domain.entities.prediction_result import PredictionResult


class PredictorPort(ABC):
    """Contract every prediction adapter (scikit-learn, remote model, stub) must fulfil."""

    @abstractmethod
    def predict(self, character: Character) -> PredictionResult:
        """Predict whether the character achieves social mobility.

        Implementations must not mutate the character and must return a
        PredictionResult whose has_mobility agrees with its probability and
        threshold.

        Args:
            character: The validated profile x to evaluate.

        Returns:
            The calibrated probability p(x) together with the decision implied
            by the deployed policy.

        Raises:
            Exception: An implementation-specific error when the engine cannot
                score the character (the scikit-learn adapter raises
                PredictionError).
        """
