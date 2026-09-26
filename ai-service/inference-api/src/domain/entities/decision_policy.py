"""How a mobility probability becomes a decision (selective classification).

This module implements the decision rule of a selective classifier: a
classifier that may abstain when its confidence is insufficient. Given the
calibrated probability p = P(Y = 1 | x) returned by the model, and an
uncertainty band (l, u) with 0 <= l < u <= 1, the rule partitions [0, 1]
into three zones:

    p >= u        MOBILITY      the model decides "yes"
    p <= l        NO_MOBILITY   the model decides "no"
    l <  p <  u   UNCERTAIN     the model abstains

The bounds are not chosen here. They are estimated on the validation
partition by step 04 of the training pipeline (04_evaluator.choose_band), as
the solution of

    maximise   coverage = P(p <= l or p >= u)
    subject to P(Y = 1 | p >= u) >= 0.95  and  P(Y = 0 | p <= l) >= 0.95,

and they reach the inference API through traceability_report.json. In the
game, the abstention zone is resolved by the BFF with a Bernoulli(p) draw
("systemic luck"). Because p is calibrated, the in-game frequency of
mobility inside the band then matches the real-world frequency.

The domain layer is deliberately independent of the model. Any estimator
that returns a calibrated probability, including the stacking ensemble, is
governed by the same policy.
"""

from dataclasses import dataclass
from enum import StrEnum

from src.domain.exceptions import InvalidPredictionError


class MobilityZone(StrEnum):
    """The three decision zones of the selective classifier.

    The string values are part of the public API contract with the BFF, which
    maps each zone to a game outcome. Changing them is a breaking change.

    Attributes:
        MOBILITY: p >= upper_bound; a confident positive decision.
        NO_MOBILITY: p <= lower_bound; a confident negative decision.
        UNCERTAIN: lower_bound < p < upper_bound; the model abstains, and the
            game resolves the outcome by weighted chance.
    """

    MOBILITY = "mobility"  # p >= upper_bound: confident yes
    NO_MOBILITY = "no_mobility"  # p <= lower_bound: confident no
    UNCERTAIN = "uncertain"  # in between: the model abstains (the game rolls weighted luck)


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionPolicy:
    """Immutable decision policy chosen on the validation set (training-pipeline step 04).

    The class is a value object. It is frozen, so a policy loaded at start-up
    cannot be altered while requests are served, and it validates its
    invariants on construction, so an invalid policy can never exist in
    memory.

    Attributes:
        lower_bound: The lower bound l of the uncertainty band. Probabilities
            at or below it are decided as NO_MOBILITY, whose validation
            negative predictive value is at least 95%.
        upper_bound: The upper bound u of the uncertainty band. Probabilities
            at or above it are decided as MOBILITY, whose validation positive
            predictive value is at least 95%.
        threshold: The single cut-off that reaches 95% precision without
            abstaining. It is reported for compatibility and comparison; the
            game uses the zones.

    Raises:
        InvalidPredictionError: On construction, if a value is not a real
            number in [0, 1] or if lower_bound >= upper_bound, which would
            leave the band empty or inverted.
    """

    lower_bound: float
    upper_bound: float
    threshold: float

    def __post_init__(self) -> None:
        """Validates the invariants 0 <= l < u <= 1 and 0 <= threshold <= 1, normalising values to float.

        Booleans are rejected explicitly. bool is a subclass of int in
        Python, so True would otherwise pass as the probability 1.0.

        Raises:
            InvalidPredictionError: If an invariant does not hold.
        """
        for name in ("lower_bound", "upper_bound", "threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise InvalidPredictionError(f"'{name}' must be a number between 0 and 1, got {value!r}")
            object.__setattr__(self, name, float(value))
        if not self.lower_bound < self.upper_bound:
            raise InvalidPredictionError(
                f"lower_bound {self.lower_bound} must be below upper_bound {self.upper_bound}"
            )

    def zone_of(self, probability: float) -> MobilityZone:
        """Maps a probability to its decision zone.

        Both bounds are inclusive on the deterministic side (p >= u and
        p <= l), which matches exactly how step 04 counted the deterministic
        decisions when it estimated the band. The precision guarantees
        measured on validation therefore apply unchanged in production.

        Args:
            probability: The model's calibrated probability of mobility, in
                [0, 1].

        Returns:
            The MobilityZone that contains the probability.
        """
        if probability >= self.upper_bound:
            return MobilityZone.MOBILITY
        if probability <= self.lower_bound:
            return MobilityZone.NO_MOBILITY
        return MobilityZone.UNCERTAIN
