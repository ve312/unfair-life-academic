"""The outcome of a social-mobility prediction for one character."""

from dataclasses import dataclass

from src.domain.entities.decision_policy import DecisionPolicy, MobilityZone
from src.domain.exceptions import InvalidPredictionError


@dataclass(frozen=True, slots=True, kw_only=True)
class PredictionResult:
    """Immutable, self-validating, deterministic prediction outcome.

    probability is the model's estimated chance that the character earns above
    50K (1994 dollars). zone places it relative to the uncertainty band
    [lower_bound, upper_bound]; inside the band the model abstains and the game
    decides by a weighted coin flip with P(mobility) = probability. has_mobility
    is the single-threshold decision (probability >= threshold), kept for
    clients of the earlier API. Every field is checked against the others.

    Internal consistency is an invariant of the type. A result whose zone
    contradicts its probability and band, or whose has_mobility contradicts
    its threshold, cannot be constructed. The API can therefore never return
    a decision that disagrees with the policy recorded in
    traceability_report.json.

    Attributes:
        probability: Calibrated estimate p(x) in [0, 1].
        threshold: Single-cut decision threshold of the policy.
        lower_bound: Lower bound l of the uncertainty band.
        upper_bound: Upper bound u of the uncertainty band.
        zone: Decision zone of the probability relative to (l, u).
        has_mobility: The single-threshold decision, probability >= threshold.

    Raises:
        InvalidPredictionError: On construction, if any field is invalid or
            inconsistent with the others.
    """

    probability: float
    threshold: float
    lower_bound: float
    upper_bound: float
    zone: MobilityZone
    has_mobility: bool

    def __post_init__(self) -> None:
        """Validates the probability, the policy bounds and the mutual consistency of all fields.

        The bounds are validated by building a DecisionPolicy, which reuses
        its invariants (0 <= l < u <= 1). The zone and has_mobility are then
        recomputed from the probability and compared with the stated values.

        Raises:
            InvalidPredictionError: If any check fails.
        """
        if isinstance(self.probability, bool) or not isinstance(self.probability, (int, float)):
            raise InvalidPredictionError(f"'probability' must be a number, got {self.probability!r}")
        if not 0.0 <= self.probability <= 1.0:
            raise InvalidPredictionError(f"'probability' must be between 0 and 1, got {self.probability}")
        object.__setattr__(self, "probability", float(self.probability))

        policy = DecisionPolicy(lower_bound=self.lower_bound, upper_bound=self.upper_bound, threshold=self.threshold)
        for name in ("lower_bound", "upper_bound", "threshold"):
            object.__setattr__(self, name, getattr(policy, name))

        if not isinstance(self.zone, MobilityZone):
            raise InvalidPredictionError(f"'zone' must be a MobilityZone, got {self.zone!r}")
        if self.zone != policy.zone_of(self.probability):
            raise InvalidPredictionError(
                f"zone={self.zone} contradicts probability {self.probability:.4f} and band "
                f"[{self.lower_bound:.2f}, {self.upper_bound:.2f}]"
            )
        if not isinstance(self.has_mobility, bool):
            raise InvalidPredictionError(f"'has_mobility' must be a bool, got {self.has_mobility!r}")
        if self.has_mobility != (self.probability >= self.threshold):
            raise InvalidPredictionError(
                f"has_mobility={self.has_mobility} contradicts probability {self.probability:.4f} "
                f"and threshold {self.threshold:.4f}"
            )

    @classmethod
    def from_probability(cls, probability: float, policy: DecisionPolicy) -> "PredictionResult":
        """Builds a consistent result by applying the decision policy to a probability.

        This is the preferred constructor: the zone and the threshold
        decision are derived from the policy, so they agree with the
        probability by construction.

        Args:
            probability: The model's calibrated probability of mobility.
            policy: The decision policy estimated on the validation set.

        Returns:
            The corresponding PredictionResult.
        """
        return cls(
            probability=probability,
            threshold=policy.threshold,
            lower_bound=policy.lower_bound,
            upper_bound=policy.upper_bound,
            zone=policy.zone_of(probability),
            has_mobility=probability >= policy.threshold,
        )
