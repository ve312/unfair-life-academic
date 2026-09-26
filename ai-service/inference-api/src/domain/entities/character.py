"""The player's character profile, expressed in the model's feature schema.

Field names and value formats match the training data exactly (see
training-pipeline/artifacts/feature_schema.json): the categorical values use
the census spellings, e.g. workclass="Private", education="HS-grad",
native_country="Mexico". Unrecognised category values are allowed here; the
ML adapter decides how to encode them.
"""

from dataclasses import asdict, dataclass, fields

from src.domain.exceptions import InvalidCharacterError

# Numeric ranges observed in the training data. Values outside them would ask
# the model to extrapolate, so the domain rejects them.
AGE_RANGE = (17, 90)
EDUCATION_NUM_RANGE = (1, 16)
HOURS_PER_WEEK_RANGE = (1, 99)

CATEGORICAL_FIELDS = (
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
)
NUMERIC_FIELDS = (
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Character:
    """Immutable, self-validating profile of a player character.

    The entity is the domain representation of the feature vector x on which
    the model estimates p(x). Its invariants restrict x to the support of the
    training distribution: numeric traits must lie inside the ranges observed
    in the 1994 census. Outside that support a statistical model extrapolates,
    and its probabilities carry no empirical guarantee. Being frozen, a
    validated profile cannot be altered afterwards.

    fnlwgt is a census sampling weight with no in-game meaning. It is still
    accepted for compatibility, but the Phase 1 model does not use it.

    Attributes:
        age: Age in years, within AGE_RANGE.
        workclass: Employment type, e.g. 'Private'.
        education: Highest education level, e.g. 'HS-grad'.
        education_num: Ordinal years of schooling, within EDUCATION_NUM_RANGE.
        marital_status: Census marital status, e.g. 'Married-civ-spouse'.
        occupation: Census occupation group, e.g. 'Prof-specialty'.
        relationship: Role within the household, e.g. 'Husband'.
        race: Census race category.
        sex: 'Male' or 'Female'.
        hours_per_week: Weekly working hours, within HOURS_PER_WEEK_RANGE.
        native_country: Country of birth, e.g. 'Mexico'.
        capital_gain: Capital gains in 1994 dollars (non-negative).
        capital_loss: Capital losses in 1994 dollars (non-negative).
        fnlwgt: Optional census sampling weight (positive), unused by the model.

    Raises:
        InvalidCharacterError: On construction, if a field has the wrong type
            or lies outside its admissible range.
    """

    age: int
    workclass: str
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    hours_per_week: int
    native_country: str
    capital_gain: int = 0
    capital_loss: int = 0
    fnlwgt: int | None = None

    def __post_init__(self) -> None:
        """Validates every invariant and normalises categorical values by trimming whitespace.

        Booleans are rejected explicitly among the integers. bool is a
        subclass of int in Python, so True would otherwise pass as the age 1.

        Raises:
            InvalidCharacterError: If any field violates its type or range.
        """
        for name in CATEGORICAL_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise InvalidCharacterError(f"'{name}' must be a non-empty string, got {value!r}")
            object.__setattr__(self, name, value.strip())

        for name in NUMERIC_FIELDS:
            value = getattr(self, name)
            if value is None and name == "fnlwgt":
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise InvalidCharacterError(f"'{name}' must be an integer, got {value!r}")

        self._require_range("age", self.age, AGE_RANGE)
        self._require_range("education_num", self.education_num, EDUCATION_NUM_RANGE)
        self._require_range("hours_per_week", self.hours_per_week, HOURS_PER_WEEK_RANGE)
        for name in ("capital_gain", "capital_loss"):
            if getattr(self, name) < 0:
                raise InvalidCharacterError(f"'{name}' cannot be negative, got {getattr(self, name)}")
        if self.fnlwgt is not None and self.fnlwgt <= 0:
            raise InvalidCharacterError(f"'fnlwgt' must be positive when provided, got {self.fnlwgt}")

    @staticmethod
    def _require_range(name: str, value: int, bounds: tuple[int, int]) -> None:
        """Checks that a value lies in the closed interval [low, high].

        Args:
            name: Field name used in the error message.
            value: The value to check.
            bounds: The pair (low, high), both inclusive.

        Raises:
            InvalidCharacterError: If the value lies outside the interval.
        """
        low, high = bounds
        if not low <= value <= high:
            raise InvalidCharacterError(f"'{name}' must be between {low} and {high}, got {value}")

    def to_features(self) -> dict[str, int | str | None]:
        """Returns the profile as a flat feature mapping.

        Returns:
            A dictionary {feature_name: value} with one entry per field,
            using the training schema's names.
        """
        return asdict(self)

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Lists the traits a Character carries, in declaration order.

        The ML adapter uses this list at start-up to verify that the model
        never requires a trait that the domain cannot supply.

        Returns:
            The names of the dataclass fields.
        """
        return tuple(field.name for field in fields(cls))
