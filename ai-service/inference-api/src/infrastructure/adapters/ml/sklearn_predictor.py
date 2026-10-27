"""scikit-learn implementation of PredictorPort, backed by the Phase 1 artifacts.

Artifacts are loaded once, at construction time, from the training pipeline:
    best_model.joblib         trained classifier exposing predict_proba
    imputer.joblib            SimpleImputer(median) for the numeric columns
    encoder.joblib            OneHotEncoder for the categorical columns
    feature_schema.json       column groups and the exact feature order
    traceability_report.json  the decision policy (uncertainty band + threshold) chosen on validation data

predict() reproduces the training transform exactly (see transform_steps in
feature_schema.json): impute numeric columns -> one-hot encode categoricals ->
order columns as feature_columns -> predict_proba -> apply the decision policy.
The result is deterministic; the weighted coin flip inside the band is a game
rule applied by the BFF.

Training-serving consistency. A model's guarantees hold only if a profile is
represented in production exactly as the training rows were. Any divergence
in imputation statistics, category vocabulary or column order is
training-serving skew, and it silently invalidates every reported metric.
This adapter therefore reuses the serialised training transformers instead
of re-implementing them. Before serving, it verifies that the model, the
transformers and the schema agree with one another, and if they do not, it
fails at start-up.
"""

import json
import logging
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.domain.entities.character import Character
from src.domain.entities.decision_policy import DecisionPolicy
from src.domain.entities.prediction_result import PredictionResult
from src.domain.exceptions import InvalidPredictionError
from src.domain.ports.predictor_port import PredictorPort

logger = logging.getLogger(__name__)

# <ai-service>/inference-api/src/infrastructure/adapters/ml/sklearn_predictor.py -> parents[5] == <ai-service>
DEFAULT_ARTIFACTS_DIR = Path(__file__).resolve().parents[5] / "training-pipeline" / "artifacts"

MODEL_FILE = "best_model.joblib"
IMPUTER_FILE = "imputer.joblib"
ENCODER_FILE = "encoder.joblib"
SCHEMA_FILE = "feature_schema.json"
REPORT_FILE = "traceability_report.json"
POSITIVE_CLASS = 1


class ArtifactLoadError(RuntimeError):
    """The Phase 1 artifacts are missing, unreadable, or inconsistent with each other."""


class PredictionError(RuntimeError):
    """The model could not produce a prediction for a valid character."""


class SklearnPredictor(PredictorPort):
    """Predicts social mobility with the trained scikit-learn pipeline artifacts.

    The deployed estimator is treated as a black box that exposes
    predict_proba. The adapter therefore serves the stacking ensemble, or any
    other scikit-learn classifier, without modification.

    Attributes:
        policy: The decision policy applied to every probability.
        threshold: The policy's single-cut threshold.
        model_name: The deployed model's name, as recorded by the pipeline.
    """

    def __init__(self, artifacts_dir: Path | str | None = None, policy: DecisionPolicy | None = None) -> None:
        """Loads and validates all artifacts.

        artifacts_dir defaults to <ai-service>/training-pipeline/artifacts.
        policy defaults to the decision_policy in traceability_report.json
        (uncertainty band and threshold chosen on the validation set, reported
        on the untouched test set); pass one only to deliberately override it.

        Args:
            artifacts_dir: Folder that holds the five artifacts.
            policy: Optional explicit decision policy, used in place of the
                reported one.

        Raises:
            ArtifactLoadError: If an artifact is missing or unreadable, or the
                artifacts are inconsistent with each other.
        """
        self._artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR
        if not self._artifacts_dir.is_dir():
            raise ArtifactLoadError(f"Artifacts directory not found: {self._artifacts_dir}")

        schema = self._load_json(SCHEMA_FILE)
        try:
            self._numeric_columns: list[str] = list(schema["numeric_columns"])
            self._categorical_columns: list[str] = list(schema["categorical_columns"])
            self._feature_columns: list[str] = list(schema["feature_columns"])
        except KeyError as exc:
            raise ArtifactLoadError(f"{SCHEMA_FILE} is missing key {exc}") from exc

        self._model = self._load_joblib(MODEL_FILE)
        self._imputer = self._load_joblib(IMPUTER_FILE)
        self._encoder = self._load_joblib(ENCODER_FILE)

        report = self._load_json(REPORT_FILE)
        self._model_name: str = report.get("model", {}).get("name", type(self._model).__name__)
        self._policy = self._resolve_policy(policy, report)

        self._validate_consistency()
        logger.info(
            "SklearnPredictor ready: model=%s, features=%d, band=[%.2f, %.2f], threshold=%.6f, artifacts=%s",
            self._model_name, len(self._feature_columns), self._policy.lower_bound, self._policy.upper_bound,
            self._policy.threshold, self._artifacts_dir,
        )

    @property
    def policy(self) -> DecisionPolicy:
        """DecisionPolicy: The uncertainty band and threshold applied to every probability."""
        return self._policy

    @property
    def threshold(self) -> float:
        """float: The policy's single-cut threshold."""
        return self._policy.threshold

    @property
    def model_name(self) -> str:
        """str: The deployed model's name from traceability_report.json, or its class name as a fallback."""
        return self._model_name

    def predict(self, character: Character) -> PredictionResult:
        """Estimates p(x) for one character and applies the decision policy.

        Args:
            character: The validated profile x.

        Returns:
            A PredictionResult that holds the positive-class probability
            P(Y = 1 | x), its zone and the threshold decision.

        Raises:
            PredictionError: If the model fails, or returns a non-finite
                probability.
        """
        features = self._to_feature_frame(character)
        try:
            probabilities = self._model.predict_proba(features)
        except Exception as exc:
            raise PredictionError(f"Model failed to score character: {exc}") from exc

        probability = float(probabilities[0, self._positive_index])
        if not math.isfinite(probability):
            raise PredictionError(f"Model returned a non-finite probability: {probability}")
        return PredictionResult.from_probability(probability, self._policy)

    def _to_feature_frame(self, character: Character) -> pd.DataFrame:
        """Maps a Character to the single-row numeric frame the model was trained on.

        The transform is applied in the training order. Numeric traits are
        imputed with the training medians (a missing fnlwgt becomes NaN
        before imputation). Categorical traits are one-hot encoded with the
        training vocabulary, and a category never seen in training becomes
        the all-zeros vector and is logged. The columns are finally reordered
        as feature_columns.

        Args:
            character: The validated profile.

        Returns:
            A 1 x d float64 frame with the columns of feature_schema.json.
        """
        values = character.to_features()

        numeric_raw = pd.DataFrame(
            [[np.nan if values[column] is None else float(values[column]) for column in self._numeric_columns]],
            columns=self._numeric_columns,
        )
        numeric = pd.DataFrame(
            self._imputer.transform(numeric_raw),
            columns=self._numeric_columns,
        )

        categorical_raw = pd.DataFrame(
            [[str(values[column]) for column in self._categorical_columns]],
            columns=self._categorical_columns,
            dtype=object,
        )
        unknown = [
            f"{column}={categorical_raw.at[0, column]!r}"
            for column, categories in zip(self._categorical_columns, self._encoder.categories_)
            if categorical_raw.at[0, column] not in set(categories)
        ]
        if unknown:
            logger.warning("Categories unseen in training are encoded as all zeros: %s", ", ".join(unknown))
        with warnings.catch_warnings():
            # Already logged above in a clearer form.
            warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
            encoded_values = self._encoder.transform(categorical_raw)
        encoded = pd.DataFrame(
            encoded_values,
            columns=self._encoder.get_feature_names_out(self._categorical_columns),
        )

        features = pd.concat([numeric, encoded], axis=1)
        return features[self._feature_columns].astype("float64")

    def _resolve_policy(self, override: DecisionPolicy | None, report: dict) -> DecisionPolicy:
        """Selects the decision policy: an explicit override, or the one recorded by the pipeline.

        Args:
            override: An explicit policy, or None to use the report.
            report: The parsed traceability_report.json.

        Returns:
            The validated DecisionPolicy.

        Raises:
            ArtifactLoadError: If the report's policy is missing or invalid.
        """
        if override is not None:
            logger.warning("Decision policy overridden to %s (report value ignored)", override)
            return override
        try:
            policy = report["decision_policy"]
            return DecisionPolicy(
                lower_bound=policy["lower_bound"],
                upper_bound=policy["upper_bound"],
                threshold=policy["deterministic_threshold"],
            )
        except KeyError as exc:
            raise ArtifactLoadError(f"{REPORT_FILE} decision_policy is missing {exc}; rerun 04_evaluator.py") from exc
        except InvalidPredictionError as exc:
            raise ArtifactLoadError(f"{REPORT_FILE} has an invalid decision_policy: {exc}") from exc

    def _validate_consistency(self) -> None:
        """Verifies that the model, the transformers, the schema and the domain agree (fail-fast).

        The checks guard against training-serving skew:

        1. The model exposes predict_proba and knows the positive class, whose
           column index is cached.
        2. The model was fitted on exactly the schema's feature columns.
        3. The imputer and the encoder were fitted on the schema's numeric and
           categorical columns.
        4. Their combined outputs reproduce feature_columns in order.
        5. Every trait the model needs is carried by the Character entity.

        Raises:
            ArtifactLoadError: On the first inconsistency found.
        """
        if not hasattr(self._model, "predict_proba"):
            raise ArtifactLoadError(f"{type(self._model).__name__} does not support predict_proba")
        classes = list(getattr(self._model, "classes_", []))
        if POSITIVE_CLASS not in classes:
            raise ArtifactLoadError(f"Model classes {classes} do not include the positive class {POSITIVE_CLASS}")
        self._positive_index = classes.index(POSITIVE_CLASS)

        expected_features = getattr(self._model, "feature_names_in_", None)
        if expected_features is not None and list(expected_features) != self._feature_columns:
            raise ArtifactLoadError("best_model.joblib was trained on different feature columns than feature_schema.json")

        imputer_columns = list(getattr(self._imputer, "feature_names_in_", self._numeric_columns))
        if imputer_columns != self._numeric_columns:
            raise ArtifactLoadError(f"imputer.joblib columns {imputer_columns} != schema {self._numeric_columns}")

        encoder_columns = list(getattr(self._encoder, "feature_names_in_", self._categorical_columns))
        if encoder_columns != self._categorical_columns:
            raise ArtifactLoadError(f"encoder.joblib columns {encoder_columns} != schema {self._categorical_columns}")

        encoded_columns = list(self._encoder.get_feature_names_out(self._categorical_columns))
        if self._numeric_columns + encoded_columns != self._feature_columns:
            raise ArtifactLoadError("Imputer + encoder outputs do not reproduce feature_schema.json feature_columns")

        # The model may use a subset of the Character traits (Phase 1 dropped fnlwgt, a census
        # sampling weight); it must never need a trait the Character does not carry.
        character_fields = set(Character.field_names())
        schema_fields = set(self._numeric_columns + self._categorical_columns)
        if not schema_fields <= character_fields:
            raise ArtifactLoadError(f"Model needs traits the Character lacks: {sorted(schema_fields - character_fields)}")
        unused = sorted(character_fields - schema_fields)
        if unused:
            logger.info("Character traits accepted but not used by this model: %s", ", ".join(unused))

    def _path(self, filename: str) -> Path:
        """Resolves an artifact path inside the artifacts directory.

        Args:
            filename: The artifact's file name.

        Returns:
            The absolute path to the artifact.

        Raises:
            ArtifactLoadError: If the file does not exist.
        """
        path = self._artifacts_dir / filename
        if not path.is_file():
            raise ArtifactLoadError(f"Required artifact not found: {path}")
        return path

    def _load_joblib(self, filename: str) -> object:
        """Deserialises a joblib artifact.

        Deserialisation can execute code, so only artifacts produced by this
        project's own training pipeline may be loaded.

        Args:
            filename: The artifact's file name.

        Returns:
            The deserialised object.

        Raises:
            ArtifactLoadError: If the file is missing or cannot be loaded.
        """
        path = self._path(filename)
        try:
            return joblib.load(path)
        except Exception as exc:
            raise ArtifactLoadError(f"Could not load {path.name}: {exc}") from exc

    def _load_json(self, filename: str) -> dict:
        """Reads and parses a JSON artifact.

        Args:
            filename: The artifact's file name.

        Returns:
            The parsed JSON object.

        Raises:
            ArtifactLoadError: If the file is missing, unreadable or not valid
                JSON.
        """
        path = self._path(filename)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactLoadError(f"Could not read {path.name}: {exc}") from exc
