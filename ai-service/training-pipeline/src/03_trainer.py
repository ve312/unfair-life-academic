"""Step 03 of the Unfair Life training pipeline: train, compare and stack the models.

This module estimates the conditional probability of social mobility,
P(Y = 1 | X = x), where Y = 1 denotes an annual income above USD 50,000 (1994)
and x is the vector of census traits produced by step 02. It proceeds in three
stages, all of which are evaluated exclusively on the VALIDATION partition; the
test partition remains sealed until step 04, so that the final report is an
unbiased estimate of generalisation performance.

1. Candidate comparison. Five model families are each fitted twice, on the
   natural training distribution (76/24) and on the SMOTENC-balanced copy
   (50/50): Logistic Regression, k-Nearest Neighbours, an RBF Support Vector
   Machine (Nystroem approximation with sigmoid calibration), Random Forest
   and HistGradientBoosting. A sixth candidate, HistGradientBoosting with
   inverse-frequency class weights, is fitted on the natural distribution
   only, which gives 11 candidates. The paired fits constitute the SMOTE
   ablation study.

2. Stacked generalisation (Wolpert, 1992). The deployed model is a two-layer
   stacking ensemble trained on the natural distribution:

       Layer 1 (base learners)   F_hgb(x)  HistGradientBoosting log-odds
                                 F_lr(x)   Logistic Regression log-odds
       Layer 2 (meta-learner)    F(x) = b + w_hgb * F_hgb(x) + w_lr * F_lr(x)
                                 p(x) = sigma(F(x)) = 1 / (1 + exp(-F(x)))

   The meta-learner is fitted on out-of-fold predictions obtained by 5-fold
   stratified cross-validation, so that it learns from base-learner outputs on
   data those learners did not see. This prevents the meta-learner from
   rewarding the base learners' in-sample overfitting.

3. Stacking ablation. Two richer base layers are evaluated as alternatives
   and documented: one adding a Random Forest, the other adding a Random
   Forest and k-Nearest Neighbours.

Design rationale for the deployed architecture (validated empirically and
recorded in model_comparison.json):

* Complementary inductive biases. Gradient boosting is a low-bias,
  non-linear learner that captures interactions and thresholds; logistic
  regression is a high-bias, low-variance linear learner. Combining a flexible
  and a rigid estimator is the classical justification for stacking, because
  their errors are only partially correlated.
* Combination in log-odds space. Both base learners expose their log-odds
  through `decision_function`, and the meta-learner combines them linearly
  before a single sigmoid. A logistic meta-learner applied to raw
  probabilities instead composes two sigmoids. Because the base probabilities
  are already calibrated, that composition distorts them: during design, the
  expected calibration error (ECE) of that variant was roughly three times
  that of the base gradient-boosting model. The log-odds combination is a
  logarithmic opinion pool and preserves calibration.
* Exact explainability. Because F(x) is an affine function of two additive
  scores, the linearity axiom of Shapley values gives the ensemble's SHAP
  attribution exactly as
  phi_j(F) = w_hgb * phi_j(F_hgb) + w_lr * phi_j(F_lr).
  Adding a Random Forest would break this property, since the logit of an
  averaged probability is not additive in the features.
* Parsimony. In the ablation, the Random Forest receives a small meta-weight,
  k-Nearest Neighbours receives a weight of zero, and neither improves
  validation log-loss or calibration in a measurable way.

Run from the training-pipeline folder:
    .venv/Scripts/python.exe src/03_trainer.py
"""

import io
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

PIPELINE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PIPELINE_DIR / "data" / "processed"
ARTIFACTS_DIR = PIPELINE_DIR / "artifacts"
FEATURE_SCHEMA_FILE = ARTIFACTS_DIR / "feature_schema.json"
BEST_MODEL_FILE = ARTIFACTS_DIR / "best_model.joblib"
REPORT_FILE = ARTIFACTS_DIR / "model_comparison.json"
VALIDATION_PREDICTIONS_FILE = ARTIFACTS_DIR / "validation_predictions.parquet"

TARGET_COLUMN = "income_above_50k"
# A single seed governs every stochastic component, which makes the whole stage reproducible bit for bit.
RANDOM_STATE = 42
CALIBRATION_BINS = 10
TRAINING_SETS = ("natural", "smote")

STACKING_NAME = "Stacking (HGB + LR -> LR) [natural]"
STACKING_FAMILY = "Stacking ensemble"
# Five stratified folds: each base learner is trained on 80% of the rows to produce out-of-fold scores,
# which balances the bias of the meta-features against the cost of five refits per base learner.
STACKING_FOLDS = 5
# Numerical floor applied before the logit in the ablation meta-learners, whose base learners return
# probabilities that may be exactly 0 or 1 (e.g. pure Random Forest leaves): logit(1e-6) is about -13.8.
LOGIT_EPSILON = 1e-6
# The meta-learner's logit must equal the affine combination of base log-odds up to round-off.
ADDITIVITY_TOLERANCE = 1e-9


class TrainingError(Exception):
    """Raised when inputs are missing or inconsistent, or a model fails to train or verify."""


def log(message: str) -> None:
    """Writes a progress message to standard output, prefixed with the stage name.

    Args:
        message: Human-readable progress or diagnostic text.
    """
    print(f"[03_trainer] {message}", flush=True)


def load() -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Loads the encoded partitions produced by step 02 and verifies their schema.

    Every partition must reproduce exactly the column order declared in
    feature_schema.json, because a trained estimator associates each learned
    weight with a column position. Even a permutation of the columns would
    silently invalidate the model.

    Returns:
        A mapping from partition name ('natural', 'smote', 'validation') to a
        tuple (X, y): the float64 feature matrix and the binary target vector.

    Raises:
        TrainingError: If a partition file is missing or its columns differ
            from the declared feature schema.
    """
    files = {"natural": "train", "smote": "train_smote", "validation": "validation"}
    data = {}
    for key, name in files.items():
        x_path, y_path = PROCESSED_DIR / f"X_{name}.parquet", PROCESSED_DIR / f"y_{name}.parquet"
        if not x_path.is_file() or not y_path.is_file():
            raise TrainingError(f"Missing {x_path.name} / {y_path.name}; run 02_balancer.py first")
        data[key] = (pd.read_parquet(x_path).astype("float64"), pd.read_parquet(y_path)[TARGET_COLUMN].to_numpy())
    schema = json.loads(FEATURE_SCHEMA_FILE.read_text(encoding="utf-8"))
    for key, (X, y) in data.items():
        if list(X.columns) != schema["feature_columns"]:
            raise TrainingError(f"{key} columns do not match feature_schema.json; rerun 02_balancer.py")
        log(f"Loaded {key:<10} X {X.shape}, positive rate {y.mean():.2%}")
    return data


def candidates() -> dict[str, object]:
    """Builds fresh, unfitted instances of every candidate model family.

    Scale-sensitive estimators (logistic regression, k-NN, SVM) are wrapped in
    a pipeline with their own StandardScaler, which standardises each feature
    as z = (x - mu) / sd using training statistics only. Without this, features
    with wide ranges such as capital_gain would dominate the distance metrics
    and the regularisation penalties. Tree ensembles are invariant to monotone
    transformations of the features and therefore need no scaling.

    The tree-ensemble hyperparameters come from a prior randomised search
    (16 configurations, 3-fold cross-validation). The gradient-boosting model
    minimises the regularised binary log-loss with shrinkage nu = 0.05 and an
    L2 penalty lambda = 1.0, and stops early once the loss on an internal 10%
    hold-out has not improved for 50 consecutive iterations.

    Returns:
        A mapping from family name to an unfitted scikit-learn estimator.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
        ]),
        # Exact kernel SVC is quadratic in rows; the Nystroem RBF approximation is the
        # scalable equivalent. Calibration gives it probabilities like the other models.
        "SVM (RBF, calibrated)": CalibratedClassifierCV(
            Pipeline([
                ("scaler", StandardScaler()),
                ("kernel", Nystroem(kernel="rbf", n_components=800, random_state=RANDOM_STATE)),
                ("model", LinearSVC(C=0.5, max_iter=5000, random_state=RANDOM_STATE)),
            ]),
            method="sigmoid",
            cv=3,
        ),
        # Hyperparameters from the Phase 0 randomized search (16 candidates x 3 folds).
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=50, min_samples_split=5, min_samples_leaf=1,
            max_features="sqrt", bootstrap=True, n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=1500, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting (class-weighted)": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=1500, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def stacking_folds() -> StratifiedKFold:
    """Returns the cross-validation splitter that generates the out-of-fold meta-features.

    Stratification preserves the 24% positive rate in every fold, so each base
    learner is trained and scored under the same class prior as the full
    training set. Shuffling with a fixed seed makes the folds reproducible.

    Returns:
        A seeded, shuffled StratifiedKFold with STACKING_FOLDS splits.
    """
    return StratifiedKFold(n_splits=STACKING_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def stacking_ensemble() -> StackingClassifier:
    """Builds the deployed two-layer stacking ensemble (unfitted).

    Layer 1 contains the gradient-boosting model and the logistic regression,
    with the hyperparameters used in the candidate comparison. With
    stack_method='decision_function', each base learner contributes its
    log-odds F_k(x) = log(p_k(x) / (1 - p_k(x))) rather than a probability.

    Layer 2 is an L2-regularised logistic regression that learns
    F(x) = b + w_hgb * F_hgb(x) + w_lr * F_lr(x) by maximum likelihood on the
    out-of-fold log-odds. Because it minimises the log-loss, a strictly proper
    scoring rule, its output sigma(F(x)) is a calibrated probability. After
    fitting, scikit-learn refits each base learner on the full training set,
    and those refitted learners serve predictions.

    Pure scikit-learn components are used deliberately. The serialised model
    then depends only on importable library classes, so the inference API can
    deserialise it without access to this module.

    Returns:
        An unfitted StackingClassifier.
    """
    base = candidates()
    return StackingClassifier(
        estimators=[("hgb", base["HistGradientBoosting"]), ("lr", base["Logistic Regression"])],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=stacking_folds(),
        stack_method="decision_function",
    )


def logit_meta_learner() -> Pipeline:
    """Builds a logistic meta-learner that operates on the logits of probability inputs.

    Some base learners, such as Random Forest and k-NN, expose probabilities
    but no log-odds. To combine them in the same log-odds space as the deployed
    ensemble, each probability is clipped to [epsilon, 1 - epsilon] and mapped
    through logit(p) = log(p / (1 - p)) before the logistic regression. The
    clipping keeps the transform finite for pure leaves (p in {0, 1}).

    Returns:
        An unfitted pipeline: clip -> logit -> LogisticRegression.
    """
    return Pipeline([
        ("clip", FunctionTransformer(np.clip, kw_args={"a_min": LOGIT_EPSILON, "a_max": 1 - LOGIT_EPSILON})),
        ("logit", FunctionTransformer(logit)),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def stacking_alternatives() -> dict[str, StackingClassifier]:
    """Builds the alternative base layers evaluated in the stacking ablation (unfitted).

    The Random Forest used here grows leaves of at least 10 samples. Larger
    leaves estimate class frequencies with lower variance, which improves the
    calibration of its probabilities, and they bound the size of the
    serialised forest. The phase 0 forest, whose leaves may hold a single
    sample, serialises to roughly 35 MB.

    Returns:
        A mapping from descriptive name to an unfitted StackingClassifier.
    """
    def forest() -> RandomForestClassifier:
        """Builds the calibration-oriented Random Forest used in the ablation.

        Returns:
            An unfitted RandomForestClassifier with leaves of at least 10 samples.
        """
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt", n_jobs=-1,
                                      random_state=RANDOM_STATE)

    def stack(layer: list[tuple[str, object]]) -> StackingClassifier:
        """Builds a stacking ensemble whose meta-learner combines the layer's logits.

        Args:
            layer: The named base learners of layer 1.

        Returns:
            An unfitted StackingClassifier with out-of-fold probability meta-features.
        """
        return StackingClassifier(estimators=layer, final_estimator=logit_meta_learner(), cv=stacking_folds(),
                                  stack_method="predict_proba")

    base = candidates
    return {
        "Stacking (HGB + RF + LR -> LR)": stack([
            ("hgb", base()["HistGradientBoosting"]), ("rf", forest()), ("lr", base()["Logistic Regression"]),
        ]),
        "Stacking (HGB + RF + LR + KNN -> LR)": stack([
            ("hgb", base()["HistGradientBoosting"]), ("rf", forest()), ("lr", base()["Logistic Regression"]),
            ("knn", base()["KNN"]),
        ]),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = CALIBRATION_BINS) -> float:
    """Computes the expected calibration error (ECE) over equal-width probability bins.

    The unit interval is partitioned into B bins S_1, ..., S_B. The ECE is the
    sample-weighted mean absolute gap between the average predicted
    probability and the observed positive frequency within each bin:

        ECE = sum_b (|S_b| / N) * | mean(p_i, i in S_b) - mean(y_i, i in S_b) |

    A perfectly calibrated model, for which P(Y = 1 | p(X) = q) = q, has an ECE
    of zero. Calibration matters in this project because the game samples
    outcomes with probability p(x).

    Args:
        y: Binary ground-truth labels, shape (N,).
        p: Predicted probabilities of the positive class, shape (N,).
        bins: Number of equal-width bins B.

    Returns:
        The expected calibration error, in [0, 1].
    """
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def validation_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Computes ranking, probabilistic and threshold metrics for one model.

    The metrics form three groups:

    * Ranking: average precision (the area under the precision-recall curve,
      which is informative under class imbalance) and ROC-AUC (the probability
      that a random positive is ranked above a random negative).
    * Probabilistic: the Brier score, (1/N) * sum (p_i - y_i)^2; the log-loss,
      -(1/N) * sum [y_i log p_i + (1 - y_i) log(1 - p_i)]; and the ECE. These
      evaluate the probability itself, not only the order it induces.
    * Thresholded at 0.5: accuracy, precision, recall and F1, reported only
      as a conventional reference.

    Args:
        y: Binary ground-truth labels, shape (N,).
        p: Predicted probabilities of the positive class, shape (N,).

    Returns:
        A mapping from metric name to value.
    """
    predicted = (p >= 0.5).astype(int)
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))),
        "ece": expected_calibration_error(y, p),
        "mean_predicted_probability": float(p.mean()),
        "observed_positive_rate": float(y.mean()),
        "accuracy_at_0_5": float(accuracy_score(y, predicted)),
        "precision_at_0_5": float(precision_score(y, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, predicted, zero_division=0)),
        "f1_at_0_5": float(f1_score(y, predicted, zero_division=0)),
    }


def serialized_megabytes(model: object) -> float:
    """Measures the compressed size of a fitted model, as it would be written to disk.

    Args:
        model: Any fitted, picklable estimator.

    Returns:
        The size in mebibytes of the model serialised with joblib (compress=3).
    """
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    return round(buffer.tell() / 1_048_576, 2)


def fit_timed(name: str, model: object, X: pd.DataFrame, y: np.ndarray) -> float:
    """Fits a model and returns the elapsed wall-clock time.

    Args:
        name: Model name used in error messages.
        model: Unfitted estimator, fitted in place.
        X: Training feature matrix.
        y: Training labels.

    Returns:
        The training time in seconds.

    Raises:
        TrainingError: If the estimator fails to train.
    """
    start = time.perf_counter()
    try:
        model.fit(X, y)
    except (ValueError, MemoryError) as exc:
        raise TrainingError(f"{name} failed to train: {exc}") from exc
    return time.perf_counter() - start


def log_metrics(name: str, metrics: dict[str, float], seconds: float) -> None:
    """Logs the headline validation metrics of one model on a single line.

    Args:
        name: Model name.
        metrics: Output of validation_metrics.
        seconds: Training time in seconds.
    """
    log(f"  {name:<46} AP {metrics['average_precision']:.4f} | AUC {metrics['roc_auc']:.4f} | "
        f"Brier {metrics['brier']:.4f} | ECE {metrics['ece']:.4f} | {seconds:6.1f}s")


def train_all(data) -> tuple[list[dict], dict[str, object], pd.DataFrame]:
    """Fits every candidate family on both training distributions and scores it on validation.

    Class weighting and SMOTE are alternative remedies for class imbalance and
    are not combined, so the class-weighted family is trained on the natural
    distribution only. This gives 11 candidates.

    Args:
        data: Output of load().

    Returns:
        A tuple (results, fitted, predictions): the per-candidate metric
        records, the fitted estimators keyed by name, and a frame of
        validation probabilities with one column per candidate.

    Raises:
        TrainingError: If any candidate fails to train.
    """
    X_val, y_val = data["validation"]
    results, fitted = [], {}
    predictions = pd.DataFrame({TARGET_COLUMN: y_val})
    for training_set in TRAINING_SETS:
        X_train, y_train = data[training_set]
        for family, model in candidates().items():
            if training_set == "smote" and "class-weighted" in family:
                continue  # class weights and SMOTE are alternative balancing methods, not combined
            name = f"{family} [{training_set}]"
            seconds = fit_timed(name, model, X_train, y_train)
            p = model.predict_proba(X_val)[:, 1]
            metrics = validation_metrics(y_val, p)
            results.append({"name": name, "family": family, "training_set": training_set,
                            "train_seconds": round(seconds, 1), "validation": metrics})
            fitted[name] = model
            predictions[name] = p
            log_metrics(name, metrics, seconds)
    return results, fitted, predictions


def meta_weights(model: StackingClassifier) -> dict[str, float]:
    """Extracts the meta-learner's intercept and per-base-learner weights.

    For the deployed ensemble, the weights w_k multiply each base learner's
    log-odds, so they are directly interpretable. A weight near zero means that
    the base learner adds no information beyond the others.

    Args:
        model: A fitted StackingClassifier whose final estimator is a
            LogisticRegression, or a pipeline ending in one.

    Returns:
        A mapping with the key 'intercept' and one key per base learner.
    """
    final = model.final_estimator_
    regression = final.named_steps["model"] if isinstance(final, Pipeline) else final
    weights = {name: float(w) for name, w in zip(model.named_estimators_, regression.coef_.ravel())}
    return {"intercept": float(regression.intercept_[0]), **weights}


def verify_additivity(model: StackingClassifier, X: pd.DataFrame) -> float:
    """Verifies that the ensemble's log-odds equal the affine combination of its base log-odds.

    The identity F(x) = b + sum_k w_k * F_k(x) underpins the exact SHAP
    decomposition of the ensemble, phi_j(F) = sum_k w_k * phi_j(F_k), which
    follows from the linearity axiom of Shapley values. It is checked
    numerically on the validation set.

    Args:
        model: The fitted deployed StackingClassifier.
        X: Feature matrix on which to check the identity.

    Returns:
        The maximum absolute deviation observed.

    Raises:
        TrainingError: If the deviation exceeds ADDITIVITY_TOLERANCE.
    """
    weights = meta_weights(model)
    combined = weights["intercept"] + sum(
        weights[name] * estimator.decision_function(X) for name, estimator in model.named_estimators_.items()
    )
    deviation = float(np.abs(combined - model.decision_function(X)).max())
    if deviation > ADDITIVITY_TOLERANCE:
        raise TrainingError(f"Stacking log-odds are not additive in the base log-odds (max deviation {deviation:.2e})")
    return deviation


def train_stacking(data, best_single: dict) -> tuple[dict, StackingClassifier, np.ndarray, dict]:
    """Fits the deployed stacking ensemble on the natural distribution and documents it.

    Args:
        data: Output of load().
        best_single: Metric record of the best single candidate, used as the
            reference against which the ensemble's gain is measured.

    Returns:
        A tuple (result, model, p, details): the ensemble's metric record in
        the same format as the candidates, the fitted ensemble, its validation
        probabilities, and a description of its architecture, meta-weights,
        base-learner performance and gain over the best single model.

    Raises:
        TrainingError: If the ensemble fails to train or the additivity check
            fails.
    """
    X_train, y_train = data["natural"]
    X_val, y_val = data["validation"]
    model = stacking_ensemble()
    seconds = fit_timed(STACKING_NAME, model, X_train, y_train)
    p = model.predict_proba(X_val)[:, 1]
    metrics = validation_metrics(y_val, p)
    log_metrics(STACKING_NAME, metrics, seconds)

    deviation = verify_additivity(model, X_val)
    weights = meta_weights(model)
    log(f"  Meta-learner: F(x) = {weights['intercept']:+.4f} + {weights['hgb']:.4f} * F_hgb(x) "
        f"+ {weights['lr']:.4f} * F_lr(x)  (additivity verified, max deviation {deviation:.1e})")

    base_learners = {
        name: validation_metrics(y_val, estimator.predict_proba(X_val)[:, 1])
        for name, estimator in model.named_estimators_.items()
    }
    reference = best_single["validation"]
    details = {
        "architecture": {
            "layer_1_base_learners": {
                "hgb": "HistGradientBoostingClassifier (same hyperparameters as the candidate)",
                "lr": "StandardScaler + LogisticRegression (same hyperparameters as the candidate)",
            },
            "layer_2_meta_learner": "LogisticRegression on the base learners' log-odds",
            "stack_method": "decision_function",
            "meta_features": f"out-of-fold log-odds, {STACKING_FOLDS}-fold stratified CV (random_state={RANDOM_STATE})",
            "training_set": "natural",
            "formula": "p(x) = sigmoid(intercept + w_hgb * F_hgb(x) + w_lr * F_lr(x))",
        },
        "meta_learner": weights,
        "additivity_max_deviation": deviation,
        "base_learners_validation": base_learners,
        "gain_over_best_single_model": {
            "reference_model": best_single["name"],
            **{f"delta_{key}": metrics[key] - reference[key]
               for key in ("average_precision", "roc_auc", "brier", "log_loss", "ece")},
        },
        "serialized_mb": serialized_megabytes(model),
    }
    result = {"name": STACKING_NAME, "family": STACKING_FAMILY, "training_set": "natural",
              "train_seconds": round(seconds, 1), "validation": metrics}
    return result, model, p, details


def stacking_ablation(data, deployed: dict, deployed_details: dict) -> list[dict]:
    """Evaluates richer base layers to justify the composition of the deployed ensemble.

    Args:
        data: Output of load().
        deployed: Metric record of the deployed ensemble.
        deployed_details: Details returned by train_stacking for the deployed
            ensemble.

    Returns:
        One record per architecture, the deployed one first, each with its
        validation metrics, meta-weights, serialised size, and whether its
        SHAP decomposition is exact.

    Raises:
        TrainingError: If an alternative fails to train.
    """
    X_train, y_train = data["natural"]
    X_val, y_val = data["validation"]
    rows = [{
        "name": deployed["name"].removesuffix(" [natural]"),
        "deployed": True,
        "exact_shap_decomposition": True,
        "meta_learner": deployed_details["meta_learner"],
        "serialized_mb": deployed_details["serialized_mb"],
        "train_seconds": deployed["train_seconds"],
        "validation": deployed["validation"],
    }]
    for name, model in stacking_alternatives().items():
        seconds = fit_timed(name, model, X_train, y_train)
        metrics = validation_metrics(y_val, model.predict_proba(X_val)[:, 1])
        log_metrics(name, metrics, seconds)
        rows.append({
            "name": name,
            "deployed": False,
            "exact_shap_decomposition": False,
            "meta_learner": meta_weights(model),
            "serialized_mb": serialized_megabytes(model),
            "train_seconds": round(seconds, 1),
            "validation": metrics,
        })
    return rows


def smote_comparison(results: list[dict]) -> list[dict]:
    """Summarises, per family, the effect of SMOTE on validation performance.

    A positive change in average precision would favour SMOTE. The mean
    predicted probability is compared with the observed positive rate: under
    calibration the two coincide, and a mean well above the observed rate
    signals that the balanced training prior inflated the probabilities.

    Args:
        results: Candidate metric records from train_all.

    Returns:
        One record per family that was trained on both distributions, sorted
        by family name.
    """
    by_name = {r["name"]: r["validation"] for r in results}
    rows = []
    for family in {r["family"] for r in results if r["training_set"] == "smote"}:
        natural, smote = by_name[f"{family} [natural]"], by_name[f"{family} [smote]"]
        rows.append({
            "family": family,
            "average_precision_natural": natural["average_precision"],
            "average_precision_smote": smote["average_precision"],
            "delta_average_precision": smote["average_precision"] - natural["average_precision"],
            "ece_natural": natural["ece"],
            "ece_smote": smote["ece"],
            "mean_probability_natural": natural["mean_predicted_probability"],
            "mean_probability_smote": smote["mean_predicted_probability"],
            "observed_positive_rate": natural["observed_positive_rate"],
        })
    return sorted(rows, key=lambda r: r["family"])


def print_table(results: list[dict], deployed_name: str, best_single_name: str) -> None:
    """Prints every model's validation metrics, ranked by average precision.

    Args:
        results: Metric records of the candidates and the ensemble.
        deployed_name: Name of the deployed model, marked with '*'.
        best_single_name: Name of the best single candidate, marked with '+'.
    """
    header = f"{'Model (validation set)':<48}{'AP':>8}{'ROC-AUC':>9}{'Brier':>8}{'ECE':>8}{'Acc@.5':>8}{'Prec@.5':>9}{'Rec@.5':>8}"
    print("\n" + "-" * len(header) + "\n" + header + "\n" + "-" * len(header))
    for r in sorted(results, key=lambda r: -r["validation"]["average_precision"]):
        m = r["validation"]
        mark = " *" if r["name"] == deployed_name else " +" if r["name"] == best_single_name else ""
        print(f"{r['name'] + mark:<48}{m['average_precision']:>8.4f}{m['roc_auc']:>9.4f}{m['brier']:>8.4f}{m['ece']:>8.4f}"
              f"{m['accuracy_at_0_5']:>8.4f}{m['precision_at_0_5']:>9.4f}{m['recall_at_0_5']:>8.4f}")
    print("-" * len(header))
    print("* deployed: stacking ensemble | + best single candidate (highest AP, tie-break lower Brier)\n")


def main() -> int:
    """Runs the stage end to end: compare, stack, ablate, persist and verify.

    Returns:
        The process exit code: 0 on success, 1 on a TrainingError.
    """
    try:
        data = load()
        log("Training every candidate on the natural and the SMOTE-balanced training set:")
        results, fitted, predictions = train_all(data)
        best_single = max(results, key=lambda r: (round(r["validation"]["average_precision"], 4),
                                                  -r["validation"]["brier"]))

        log(f"Training the stacking ensemble ({STACKING_FOLDS}-fold out-of-fold log-odds, natural distribution):")
        stacked, model, p_stacked, details = train_stacking(data, best_single)
        results.append(stacked)
        predictions[STACKING_NAME] = p_stacked
        print_table(results, STACKING_NAME, best_single["name"])

        gain = details["gain_over_best_single_model"]
        log(f"Ensemble vs {best_single['name']}: AP {gain['delta_average_precision']:+.4f} | "
            f"Brier {gain['delta_brier']:+.4f} | log-loss {gain['delta_log_loss']:+.4f} | ECE {gain['delta_ece']:+.4f}")

        log("Stacking ablation (alternative base layers, validation set):")
        ablation = stacking_ablation(data, stacked, details)

        comparison = smote_comparison(results)
        log("SMOTE ablation (validation set):")
        for row in comparison:
            log(f"  {row['family']:<26} AP natural {row['average_precision_natural']:.4f} -> SMOTE "
                f"{row['average_precision_smote']:.4f} ({row['delta_average_precision']:+.4f}) | mean probability "
                f"{row['mean_probability_natural']:.3f} -> {row['mean_probability_smote']:.3f} "
                f"(true rate {row['observed_positive_rate']:.3f})")

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, BEST_MODEL_FILE, compress=3)
        predictions.to_parquet(VALIDATION_PREDICTIONS_FILE, index=False)
        REPORT_FILE.write_text(json.dumps({
            "selection_rule": "deployed model: stacking ensemble (architecture fixed by design, see 'stacking' and "
                              "'stacking_ablation'); single candidates ranked by highest validation average "
                              "precision, tie-break lower Brier score",
            "selected_model": STACKING_NAME,
            "selected_family": STACKING_FAMILY,
            "selected_training_set": "natural",
            "best_single_model": best_single["name"],
            "stacking": details,
            "models": results,
            "stacking_ablation": ablation,
            "smote_ablation": comparison,
        }, indent=2), encoding="utf-8")
        log(f"Deployed {STACKING_NAME} -> {BEST_MODEL_FILE.relative_to(PIPELINE_DIR)} "
            f"({BEST_MODEL_FILE.stat().st_size / 1_048_576:.1f} MB)")
        log(f"Saved {REPORT_FILE.relative_to(PIPELINE_DIR)} and {VALIDATION_PREDICTIONS_FILE.relative_to(PIPELINE_DIR)}")

        reloaded = joblib.load(BEST_MODEL_FILE).predict_proba(data["validation"][0])[:, 1]
        if not np.allclose(reloaded, predictions[STACKING_NAME]):
            raise TrainingError("Reloaded best model does not reproduce its validation predictions")
        log("Verified: reloaded best_model.joblib reproduces its validation predictions")
    except TrainingError as exc:
        log(f"ERROR: {exc}")
        return 1
    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
