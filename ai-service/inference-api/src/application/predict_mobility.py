"""Use case: predict whether a player character achieves social mobility.

This module is the application layer of a hexagonal (ports-and-adapters)
architecture. It holds no statistical logic of its own. It coordinates the
domain entities (Character, PredictionResult) with an abstract prediction
engine (PredictorPort), and it enforces at runtime the type contract that
connects them. The estimation of p(x) = P(Y = 1 | X = x) is delegated
entirely to the injected adapter.
"""

from src.domain.entities.character import Character
from src.domain.entities.prediction_result import PredictionResult
from src.domain.ports.predictor_port import PredictorPort


class PredictMobilityUseCase:
    """Orchestrates a mobility prediction through an injected PredictorPort.

    The use case depends only on the port, never on a concrete adapter, so the
    scikit-learn engine can be swapped (e.g. for a stub in tests) without
    touching application or domain code.
    """

    def __init__(self, predictor: PredictorPort) -> None:
        """Binds the use case to a prediction engine (dependency injection).

        Args:
            predictor: Any implementation of PredictorPort.

        Raises:
            TypeError: If predictor does not implement PredictorPort.
        """
        if not isinstance(predictor, PredictorPort):
            raise TypeError(f"predictor must implement PredictorPort, got {type(predictor).__name__}")
        self._predictor = predictor

    def execute(self, character: Character) -> PredictionResult:
        """Predicts the social mobility of one character.

        Both sides of the port are checked at runtime: the input must be a
        validated Character entity, and the adapter must return a
        PredictionResult. A faulty adapter therefore cannot leak an
        unvalidated value to the transport layer.

        Args:
            character: The validated profile to evaluate.

        Returns:
            The adapter's prediction, consistent with the deployed decision
            policy.

        Raises:
            TypeError: If character is not a Character, or if the adapter
                returns something other than a PredictionResult.
        """
        if not isinstance(character, Character):
            raise TypeError(f"character must be a Character entity, got {type(character).__name__}")
        result = self._predictor.predict(character)
        if not isinstance(result, PredictionResult):
            raise TypeError(
                f"{type(self._predictor).__name__}.predict must return a PredictionResult, "
                f"got {type(result).__name__}"
            )
        return result

    def __call__(self, character: Character) -> PredictionResult:
        """Allows the use case to be invoked as a function; equivalent to execute().

        Args:
            character: The validated profile to evaluate.

        Returns:
            The prediction returned by execute().
        """
        return self.execute(character)
