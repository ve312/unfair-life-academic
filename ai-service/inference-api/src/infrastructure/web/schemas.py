"""HTTP request/response schemas for the inference API.

Validation here is strict and happens before anything reaches the domain:
  * integers must be JSON integers ("28", 28.5 and true are rejected),
  * numeric bounds are the domain's own ranges (single source of truth),
  * text is trimmed and must be non-empty,
  * unknown fields are rejected, so typos in the BFF payload fail loudly.
Category values are free text by design: values the model never saw in
training (e.g. an unlisted country) are accepted and simply add no signal.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.domain.entities.character import AGE_RANGE, EDUCATION_NUM_RANGE, HOURS_PER_WEEK_RANGE, Character
from src.domain.entities.decision_policy import MobilityZone
from src.domain.entities.prediction_result import PredictionResult

CategoryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class CharacterRequest(BaseModel):
    """A player character profile, using the census value spellings the model was trained on."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "age": 34,
                    "workclass": "Private",
                    "education": "HS-grad",
                    "education_num": 9,
                    "marital_status": "Married-civ-spouse",
                    "occupation": "Adm-clerical",
                    "relationship": "Wife",
                    "race": "White",
                    "sex": "Female",
                    "hours_per_week": 40,
                    "native_country": "Mexico",
                    "capital_gain": 0,
                    "capital_loss": 0,
                }
            ]
        },
    )

    age: int = Field(ge=AGE_RANGE[0], le=AGE_RANGE[1], description="Age in years")
    workclass: CategoryText = Field(description="Employment type, e.g. 'Private', 'Self-emp-inc', 'State-gov'")
    education: CategoryText = Field(description="Highest education, e.g. 'HS-grad', 'Bachelors', 'Masters'")
    education_num: int = Field(
        ge=EDUCATION_NUM_RANGE[0], le=EDUCATION_NUM_RANGE[1],
        description="Education level as a number (1 = Preschool ... 16 = Doctorate)",
    )
    marital_status: CategoryText = Field(description="e.g. 'Never-married', 'Married-civ-spouse', 'Divorced'")
    occupation: CategoryText = Field(description="e.g. 'Adm-clerical', 'Prof-specialty', 'Farming-fishing'")
    relationship: CategoryText = Field(description="e.g. 'Husband', 'Wife', 'Own-child', 'Not-in-family'")
    race: CategoryText = Field(description="e.g. 'White', 'Black', 'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Other'")
    sex: CategoryText = Field(description="'Male' or 'Female'")
    hours_per_week: int = Field(
        ge=HOURS_PER_WEEK_RANGE[0], le=HOURS_PER_WEEK_RANGE[1], description="Weekly working hours",
    )
    native_country: CategoryText = Field(
        description="Country of birth, e.g. 'Mexico', 'Guatemala', 'United-States'. Unrecognised countries add no advantage.",
    )
    capital_gain: int = Field(default=0, ge=0, description="Yearly capital gains in USD")
    capital_loss: int = Field(default=0, ge=0, description="Yearly capital losses in USD")
    fnlwgt: int | None = Field(
        default=None, gt=0,
        description="Census sampling weight. Accepted for compatibility but not used: the Phase 1 model excludes it",
    )

    def to_domain(self) -> Character:
        """Maps the validated request onto the pure domain entity.

        Returns:
            The Character built from the request fields.

        Raises:
            InvalidCharacterError: If a domain rule rejects the profile.
        """
        return Character(**self.model_dump())


class PredictionResponse(BaseModel):
    """Social-mobility prediction for one character."""

    probability: float = Field(ge=0.0, le=1.0, description="Model probability that income exceeds 50K (1994 dollars)")
    zone: MobilityZone = Field(
        description="'mobility' (p >= upper_bound), 'no_mobility' (p <= lower_bound) or 'uncertain' "
        "(inside the band: the model abstains and the game decides by weighted luck)",
    )
    lower_bound: float = Field(ge=0.0, le=1.0, description="Lower edge of the uncertainty band")
    upper_bound: float = Field(ge=0.0, le=1.0, description="Upper edge of the uncertainty band")
    threshold: float = Field(ge=0.0, le=1.0, description="Single-cut threshold that reached 95% precision")
    has_mobility: bool = Field(description="probability >= threshold (single-threshold decision, kept for compatibility)")

    @classmethod
    def from_domain(cls, result: PredictionResult) -> "PredictionResponse":
        """Serialises a domain prediction into the public response schema.

        Args:
            result: The domain prediction.

        Returns:
            The equivalent response model.
        """
        return cls(
            probability=result.probability,
            zone=result.zone,
            lower_bound=result.lower_bound,
            upper_bound=result.upper_bound,
            threshold=result.threshold,
            has_mobility=result.has_mobility,
        )


class HealthResponse(BaseModel):
    """Liveness report that also identifies the model and the policy being served.

    Exposing the model name and the band lets a monitor verify which version
    of the model is live, not merely that the process is running.

    Attributes:
        status: Always 'ok' when the model is loaded.
        model: The deployed model's name from traceability_report.json.
        lower_bound: Lower edge of the uncertainty band.
        upper_bound: Upper edge of the uncertainty band.
        threshold: Single-cut threshold of the policy.
    """

    status: str
    model: str
    lower_bound: float
    upper_bound: float
    threshold: float


class ErrorResponse(BaseModel):
    """Error body returned by FastAPI's HTTPException handler.

    Attributes:
        detail: Human-readable description of the error.
    """

    detail: str
