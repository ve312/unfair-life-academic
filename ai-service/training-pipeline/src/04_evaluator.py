"""Step 04 of the Unfair Life training pipeline: decision policy and final report.

This module turns the probabilistic classifier from step 03 into the decision
policy that the game applies, and it produces the final, unbiased performance
report. The protocol separates selection from assessment. Every free
parameter of the policy is chosen on the VALIDATION partition. The TEST
partition is then used exactly once, here, to measure how those frozen
decisions perform on people the model has never seen. Reusing the test set
for any choice would make its estimates optimistically biased.

1. Selective classification (the deployed policy). The classifier may
   abstain. Given a calibrated probability p = p(x), the decision rule is

       p >= u            -> mobility      (deterministic)
       p <= l            -> no mobility   (deterministic)
       l <  p <  u       -> abstain; in the game, a Bernoulli(p) draw
                            ("systemic luck") applied by the BFF

   The band (l, u) solves the constrained optimisation problem

       maximise   coverage(l, u) = |{i : p_i <= l or p_i >= u}| / |V|
       subject to P(Y = 1 | p >= u) >= 0.95  and  P(Y = 0 | p <= l) >= 0.95,

   where V is the validation set and each deterministic outcome must be
   supported by at least MIN_DECISIONS cases. The problem is solved by an
   exhaustive search over a 0.01 grid. Maximising coverage under a precision
   floor is the risk-coverage trade-off of selective prediction: the model
   decides as often as it can while keeping the error rate of its decisions
   controlled.

2. Deterministic threshold (reported alongside). This is the lowest single
   cut-off t in [0.50, 0.99] whose validation precision reaches 95% with a
   recall above 15%. It serves as a reference for a model that never abstains.

3. Uncertainty quantification. Proportions carry Wilson score intervals.
   Coverage and accuracy carry percentile bootstrap intervals over
   BOOTSTRAP_RESAMPLES resamples of the test set. The game's expected accuracy
   is computed in closed form and corroborated by COIN_FLIP_SIMULATIONS Monte
   Carlo replays.

The estimator is treated as a black box that exposes predict_proba. The
policy therefore applies unchanged to the stacking ensemble deployed by
step 03, whose architecture is copied into the report for traceability.

Run from the training-pipeline folder:
    .venv/Scripts/python.exe src/04_evaluator.py
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, precision_score, recall_score, roc_auc_score

PIPELINE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PIPELINE_DIR / "data" / "processed"
ARTIFACTS_DIR = PIPELINE_DIR / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "best_model.joblib"
COMPARISON_FILE = ARTIFACTS_DIR / "model_comparison.json"
FEATURE_SCHEMA_FILE = ARTIFACTS_DIR / "feature_schema.json"
REPORT_FILE = ARTIFACTS_DIR / "traceability_report.json"
TEST_PREDICTIONS_FILE = ARTIFACTS_DIR / "test_predictions.parquet"

TARGET_COLUMN = "income_above_50k"
TARGET_PRECISION = 0.95
MIN_RECALL = 0.15
# Lower bound on the support of each deterministic outcome, so that a precision estimate never rests on a
# handful of cases (at n = 30 the Wilson half-width around 95% is still about 9 percentage points).
MIN_DECISIONS = 30
THRESHOLD_RANGE = (0.50, 0.99)
LOWER_GRID = np.round(np.arange(0.01, 0.50, 0.01), 2)
UPPER_GRID = np.round(np.arange(0.50, 1.00, 0.01), 2)
BOOTSTRAP_RESAMPLES = 2000
COIN_FLIP_SIMULATIONS = 2000
RANDOM_STATE = 42
# Two-sided 97.5% quantile of the standard normal distribution, for 95% intervals.
WILSON_Z = 1.959964


class EvaluationError(Exception):
    """Raised when inputs are missing or no admissible policy exists."""


def log(message: str) -> None:
    """Writes a progress message to standard output, prefixed with the stage name.

    Args:
        message: Human-readable progress or diagnostic text.
    """
    print(f"[04_evaluator] {message}", flush=True)


def wilson(successes: int, trials: int) -> list[float]:
    """Computes the Wilson score 95% confidence interval for a binomial proportion.

    For an observed proportion pi_hat = k / n, the interval is

        (pi_hat + z^2 / 2n  +/-  z * sqrt(pi_hat (1 - pi_hat) / n + z^2 / 4n^2)) / (1 + z^2 / n).

    It inverts the score test rather than the Wald test. Its coverage is
    therefore close to nominal even when pi_hat is near 0 or 1, as with
    precisions around 95%, and its limits always lie inside [0, 1].

    Args:
        successes: Number of successes k.
        trials: Number of trials n.

    Returns:
        The interval [lower, upper], or [0.0, 0.0] when there are no trials.
    """
    if trials == 0:
        return [0.0, 0.0]
    p = successes / trials
    denominator = 1 + WILSON_Z**2 / trials
    centre = (p + WILSON_Z**2 / (2 * trials)) / denominator
    margin = WILSON_Z * math.sqrt(p * (1 - p) / trials + WILSON_Z**2 / (4 * trials**2)) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def load() -> tuple[object, str, dict | None, dict[str, tuple[pd.DataFrame, np.ndarray]]]:
    """Loads the deployed model, its identity, and the validation and test partitions.

    Returns:
        A tuple (model, name, stacking, data): the fitted estimator, its name
        as recorded by step 03, the stacking description (None for a single
        model), and a mapping from partition name to (X, y).

    Raises:
        EvaluationError: If a required artifact is missing, the model does not
            expose probabilities, or a partition's columns differ from the
            feature schema.
    """
    for path in (MODEL_FILE, COMPARISON_FILE, FEATURE_SCHEMA_FILE):
        if not path.is_file():
            raise EvaluationError(f"Required file not found: {path}. Run steps 02 and 03 first.")
    model = joblib.load(MODEL_FILE)
    if not hasattr(model, "predict_proba"):
        raise EvaluationError(f"{type(model).__name__} has no predict_proba")
    schema = json.loads(FEATURE_SCHEMA_FILE.read_text(encoding="utf-8"))
    comparison = json.loads(COMPARISON_FILE.read_text(encoding="utf-8"))
    name = comparison["selected_model"]
    data = {}
    for split in ("validation", "test"):
        X = pd.read_parquet(PROCESSED_DIR / f"X_{split}.parquet").astype("float64")
        y = pd.read_parquet(PROCESSED_DIR / f"y_{split}.parquet")[TARGET_COLUMN].to_numpy()
        if list(X.columns) != schema["feature_columns"]:
            raise EvaluationError(f"X_{split} columns do not match feature_schema.json")
        data[split] = (X, y)
    log(f"Loaded model '{name}', validation {data['validation'][0].shape}, test {data['test'][0].shape}")
    return model, name, comparison.get("stacking"), data


# ------------------------------------------------------------------ policy search (validation)

def choose_band(p: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Finds the widest-coverage uncertainty band whose two deterministic outcomes reach the precision floor.

    The search is exhaustive over LOWER_GRID x UPPER_GRID. A lower bound is
    discarded as soon as its negative predictive value, P(Y = 0 | p <= l),
    falls short of TARGET_PRECISION. For each admissible lower bound, every
    upper bound is tested on its positive predictive value, P(Y = 1 | p >= u).
    Coverage is non-increasing as the band widens, so the optimum is the
    narrowest band that satisfies both constraints.

    Args:
        p: Validation probabilities, shape (N,).
        y: Validation labels, shape (N,).

    Returns:
        A tuple (lower, upper, evaluated): the optimal bounds and the number
        of candidate bands examined.

    Raises:
        EvaluationError: If no band satisfies the constraints.
    """
    best = None
    evaluated = 0
    for lower in LOWER_GRID:
        negative = p <= lower
        n_negative = int(negative.sum())
        if n_negative < MIN_DECISIONS or (1 - y[negative]).mean() < TARGET_PRECISION:
            continue
        for upper in UPPER_GRID:
            evaluated += 1
            positive = p >= upper
            n_positive = int(positive.sum())
            if n_positive < MIN_DECISIONS or y[positive].mean() < TARGET_PRECISION:
                continue
            coverage = (n_positive + n_negative) / len(p)
            if best is None or coverage > best[0]:
                best = (coverage, float(lower), float(upper))
    if best is None:
        raise EvaluationError("No uncertainty band reaches the precision target on validation")
    return best[1], best[2], evaluated


def choose_threshold(p: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Finds the lowest single threshold whose precision reaches the target, subject to a recall floor.

    Raising the threshold trades recall for precision. The lowest admissible
    threshold therefore keeps as much recall as the precision target allows.
    The candidate thresholds are the observed probabilities, which is exact,
    because precision and recall change only at those values. If no threshold
    reaches the target, the one with the highest precision is returned and
    the status records the shortfall.

    Args:
        p: Validation probabilities, shape (N,).
        y: Validation labels, shape (N,).

    Returns:
        A tuple (threshold, status).

    Raises:
        EvaluationError: If no threshold keeps recall above MIN_RECALL.
    """
    positives = int(y.sum())
    candidates = np.unique(np.concatenate([[THRESHOLD_RANGE[0]], p[(p >= THRESHOLD_RANGE[0]) & (p <= THRESHOLD_RANGE[1])]]))
    admissible = []
    for t in candidates:
        flagged = p >= t
        tp, n = int((flagged & (y == 1)).sum()), int(flagged.sum())
        if n == 0 or tp / positives <= MIN_RECALL:
            continue
        admissible.append((float(t), tp / n, tp / positives))
    if not admissible:
        raise EvaluationError("No threshold keeps recall above the minimum on validation")
    meeting = [row for row in admissible if row[1] >= TARGET_PRECISION]
    if meeting:
        return min(row[0] for row in meeting), "TARGET_MET_ON_VALIDATION"
    return max(admissible, key=lambda row: (row[1], row[2]))[0], "TARGET_NOT_REACHABLE_CLOSEST_PRECISION_SELECTED"


# ------------------------------------------------------------------ evaluation (test)

def selective_metrics(p: np.ndarray, y: np.ndarray, lower: float, upper: float) -> dict:
    """Evaluates the selective-classification policy on one partition.

    Coverage is the share of cases decided deterministically. Accuracy on
    covered cases and the two precisions measure the reliability of those
    decisions. Inside the band, the observed positive rate is compared with
    the mean predicted probability. For a calibrated model the two coincide,
    which is the condition under which a Bernoulli(p) draw reproduces
    real-world frequencies.

    Args:
        p: Probabilities, shape (N,).
        y: Labels, shape (N,).
        lower: Lower bound l of the uncertainty band.
        upper: Upper bound u of the uncertainty band.

    Returns:
        Coverage, accuracy and precisions with Wilson intervals, in-band
        calibration statistics, and the 2 x 3 decision matrix.
    """
    positive, negative = p >= upper, p <= lower
    covered = positive | negative
    uncertain = ~covered
    correct = (positive & (y == 1)) | (negative & (y == 0))
    tp, n_pos = int((positive & (y == 1)).sum()), int(positive.sum())
    tn, n_neg = int((negative & (y == 0)).sum()), int(negative.sum())
    return {
        "coverage": float(covered.mean()),
        "covered_cases": int(covered.sum()),
        "uncertain_cases": int(uncertain.sum()),
        "accuracy_on_covered": float(correct.sum() / max(covered.sum(), 1)),
        "accuracy_on_covered_95ci": wilson(int(correct.sum()), int(covered.sum())),
        "precision_mobility": float(tp / max(n_pos, 1)),
        "precision_mobility_95ci": wilson(tp, n_pos),
        "precision_no_mobility": float(tn / max(n_neg, 1)),
        "precision_no_mobility_95ci": wilson(tn, n_neg),
        "share_of_high_earners_decided_deterministically": float(tp / max(int(y.sum()), 1)),
        "positive_rate_inside_band": float(y[uncertain].mean()) if uncertain.any() else None,
        "mean_probability_inside_band": float(p[uncertain].mean()) if uncertain.any() else None,
        "decision_matrix": {
            "actual_no_mobility": {"predicted_no": int((negative & (y == 0)).sum()), "luck": int((uncertain & (y == 0)).sum()),
                                   "predicted_yes": int((positive & (y == 0)).sum())},
            "actual_mobility": {"predicted_no": int((negative & (y == 1)).sum()), "luck": int((uncertain & (y == 1)).sum()),
                                "predicted_yes": int((positive & (y == 1)).sum())},
        },
    }


def bootstrap_selective(p: np.ndarray, y: np.ndarray, lower: float, upper: float) -> dict:
    """Estimates percentile bootstrap 95% intervals for coverage and accuracy on covered cases.

    Each of the BOOTSTRAP_RESAMPLES replicates draws N indices with
    replacement, which approximates sampling a fresh test set from the same
    population, and recomputes both statistics. The 2.5th and 97.5th
    percentiles of the replicates delimit the interval. Accuracy on covered
    cases is a ratio of two random quantities, so the bootstrap is preferred
    here over a closed-form binomial interval. A dedicated generator seeded
    with RANDOM_STATE makes the result reproducible.

    Args:
        p: Test probabilities, shape (N,).
        y: Test labels, shape (N,).
        lower: Lower bound of the band.
        upper: Upper bound of the band.

    Returns:
        The number of resamples and both percentile intervals.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    coverage, accuracy = [], []
    for _ in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, len(p), len(p))
        ps, ys = p[idx], y[idx]
        pos, neg = ps >= upper, ps <= lower
        covered = pos | neg
        coverage.append(covered.mean())
        accuracy.append((((pos & (ys == 1)) | (neg & (ys == 0))).sum()) / max(covered.sum(), 1))
    return {
        "resamples": BOOTSTRAP_RESAMPLES,
        "coverage_95ci": [float(np.percentile(coverage, 2.5)), float(np.percentile(coverage, 97.5))],
        "accuracy_on_covered_95ci": [float(np.percentile(accuracy, 2.5)), float(np.percentile(accuracy, 97.5))],
    }


def game_accuracy(p: np.ndarray, y: np.ndarray, lower: float, upper: float) -> dict:
    """Computes the expected accuracy of the in-game outcome over all players.

    Outside the band the outcome is the deterministic decision. Inside it, a
    Bernoulli(p_i) draw is correct with probability p_i if y_i = 1 and
    1 - p_i if y_i = 0. By linearity of expectation,

        E[accuracy] = (1/N) * [ sum_{i outside band} 1(decision_i = y_i)
                                + sum_{i inside band} (y_i p_i + (1 - y_i)(1 - p_i)) ].

    The closed form is corroborated by COIN_FLIP_SIMULATIONS Monte Carlo
    replays of the whole test set. The 2.5th and 97.5th percentiles of the
    simulated accuracies show the spread a real population of players would
    experience.

    Args:
        p: Test probabilities, shape (N,).
        y: Test labels, shape (N,).
        lower: Lower bound of the band.
        upper: Upper bound of the band.

    Returns:
        The analytical expectation, the simulated 95% range, the number of
        simulations, and the expected mobility rate inside the band.
    """
    deterministic = np.where(p >= upper, 1, np.where(p <= lower, 0, -1))
    uncertain = deterministic == -1
    correct_deterministic = (deterministic[~uncertain] == y[~uncertain]).sum()
    # Closed form: a flip with P(yes) = p is right with probability p if y = 1, else 1 - p.
    expected_correct_uncertain = np.where(y[uncertain] == 1, p[uncertain], 1 - p[uncertain]).sum()
    expected = float((correct_deterministic + expected_correct_uncertain) / len(p))

    rng = np.random.default_rng(RANDOM_STATE)
    simulated = []
    for _ in range(COIN_FLIP_SIMULATIONS):
        flips = (rng.random(int(uncertain.sum())) < p[uncertain]).astype(int)
        simulated.append((correct_deterministic + (flips == y[uncertain]).sum()) / len(p))
    flips_say_yes = float(p[uncertain].mean()) if uncertain.any() else 0.0
    return {
        "expected_accuracy_all_players": expected,
        "simulated_accuracy_95_range": [float(np.percentile(simulated, 2.5)), float(np.percentile(simulated, 97.5))],
        "simulations": COIN_FLIP_SIMULATIONS,
        "expected_mobility_rate_inside_band": flips_say_yes,
    }


def threshold_metrics(p: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Evaluates a single-threshold classifier y_hat = 1(p >= t).

    Args:
        p: Probabilities, shape (N,).
        y: Labels, shape (N,).
        threshold: Cut-off t.

    Returns:
        Accuracy, precision (with its Wilson interval), recall, the number of
        positive predictions, and the confusion matrix.
    """
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "precision_95ci": wilson(int(tp), int(tp + fp)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "predicted_positive": int(tp + fp),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def describe_model(model: object, name: str, stacking: dict | None) -> dict:
    """Builds the identity block of the report, including the ensemble architecture when present.

    Copying the architecture and meta-weights from model_comparison.json into
    the traceability report makes the production contract self-describing.
    The inference API reads this file, and it then states which estimator,
    with which combination weights, produced every reported figure.

    Args:
        model: The deployed estimator.
        name: Its name as recorded by step 03.
        stacking: The stacking description from step 03, or None.

    Returns:
        The model identity block.
    """
    identity = {"name": name, "class": type(model).__name__, "file": "artifacts/best_model.joblib"}
    if stacking is not None:
        identity["architecture"] = stacking["architecture"]
        identity["meta_learner"] = stacking["meta_learner"]
    return identity


def print_report(report: dict) -> None:
    """Prints a human-readable summary of the final report.

    Args:
        report: The report dictionary written to traceability_report.json.
    """
    sel, thr, game = report["selective_classification"]["test"], report["deterministic_threshold"]["test"], report["game_mechanic"]["test"]
    band, boot = report["decision_policy"], report["selective_classification"]["bootstrap"]
    base = report["reference_at_0_5"]["test"]
    rule = "=" * 96
    print(f"\n{rule}\n  UNFAIR LIFE - PHASE 1 TRACEABILITY REPORT (test set: {report['data']['test_rows']:,} people, unseen)\n{rule}")
    print(f"  Model: {report['model']['name']} | ROC-AUC {report['calibration']['test']['roc_auc']:.4f} | "
          f"Brier {report['calibration']['test']['brier']:.4f}")
    print(f"  Reference, threshold 0.50: accuracy {base['accuracy']:.2%}, precision {base['precision']:.2%}, recall {base['recall']:.2%}")
    print("-" * 96)
    print(f"  1. SELECTIVE CLASSIFICATION (deployed) - band chosen on validation: [{band['lower_bound']:.2f}, {band['upper_bound']:.2f}]")
    print(f"     Coverage (decided deterministically): {sel['coverage']:.2%}  (bootstrap 95% CI {boot['coverage_95ci'][0]:.2%}-{boot['coverage_95ci'][1]:.2%})")
    print(f"     Accuracy on decided cases:          {sel['accuracy_on_covered']:.2%}  (Wilson 95% CI {sel['accuracy_on_covered_95ci'][0]:.2%}-{sel['accuracy_on_covered_95ci'][1]:.2%})")
    print(f"     Precision 'mobility'    (p >= {band['upper_bound']:.2f}): {sel['precision_mobility']:.2%}  "
          f"(95% CI {sel['precision_mobility_95ci'][0]:.2%}-{sel['precision_mobility_95ci'][1]:.2%})")
    print(f"     Precision 'no mobility' (p <= {band['lower_bound']:.2f}): {sel['precision_no_mobility']:.2%}  "
          f"(95% CI {sel['precision_no_mobility_95ci'][0]:.2%}-{sel['precision_no_mobility_95ci'][1]:.2%})")
    m = sel["decision_matrix"]
    print(f"     {'':<22}{'no mobility':>14}{'luck':>10}{'mobility':>12}")
    print(f"     {'actual <=50K':<22}{m['actual_no_mobility']['predicted_no']:>14,}{m['actual_no_mobility']['luck']:>10,}{m['actual_no_mobility']['predicted_yes']:>12,}")
    print(f"     {'actual >50K':<22}{m['actual_mobility']['predicted_no']:>14,}{m['actual_mobility']['luck']:>10,}{m['actual_mobility']['predicted_yes']:>12,}")
    print(f"     Inside the band: {sel['uncertain_cases']:,} people, true high-earner rate {sel['positive_rate_inside_band']:.2%}, "
          f"mean probability {sel['mean_probability_inside_band']:.2%} (calibration check)")
    print(f"  2. GAME MECHANIC (coin flip inside the band): expected accuracy over everyone {game['expected_accuracy_all_players']:.2%} "
          f"(simulated 95% range {game['simulated_accuracy_95_range'][0]:.2%}-{game['simulated_accuracy_95_range'][1]:.2%})")
    print(f"  3. DETERMINISTIC THRESHOLD - {report['deterministic_threshold']['status']}: t = {thr['threshold']:.4f}")
    print(f"     Precision {thr['precision']:.2%} (95% CI {thr['precision_95ci'][0]:.2%}-{thr['precision_95ci'][1]:.2%}), "
          f"recall {thr['recall']:.2%}, accuracy {thr['accuracy']:.2%}, flagged {thr['predicted_positive']:,}")
    print(rule + "\n")


def main() -> int:
    """Runs the stage end to end: choose the policy on validation, assess it on test, persist the report.

    Returns:
        The process exit code: 0 on success, 1 on an EvaluationError.
    """
    try:
        model, name, stacking, data = load()
        p_val = model.predict_proba(data["validation"][0])[:, 1]
        p_test = model.predict_proba(data["test"][0])[:, 1]
        y_val, y_test = data["validation"][1], data["test"][1]

        lower, upper, evaluated = choose_band(p_val, y_val)
        log(f"Band chosen on validation: [{lower:.2f}, {upper:.2f}] ({evaluated:,} candidate bands evaluated)")
        threshold, status = choose_threshold(p_val, y_val)
        log(f"Deterministic threshold chosen on validation: {threshold:.4f} ({status})")

        report = {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "model": describe_model(model, name, stacking),
            "data": {"validation_rows": int(len(y_val)), "test_rows": int(len(y_test)),
                     "test_positive_rate": float(y_test.mean())},
            "decision_policy": {
                "type": "selective_classification",
                "lower_bound": lower,
                "upper_bound": upper,
                "deterministic_threshold": threshold,
                "inside_band": "weighted coin flip with P(mobility) = probability (applied by the BFF)",
                "selection_rule": f"maximise validation coverage s.t. precision >= {TARGET_PRECISION:.0%} for both "
                                  f"deterministic outcomes, each with >= {MIN_DECISIONS} cases",
            },
            "selective_classification": {
                "validation": selective_metrics(p_val, y_val, lower, upper),
                "test": selective_metrics(p_test, y_test, lower, upper),
                "bootstrap": bootstrap_selective(p_test, y_test, lower, upper),
            },
            "game_mechanic": {"test": game_accuracy(p_test, y_test, lower, upper)},
            "deterministic_threshold": {
                "status": status,
                "rule": f"lowest threshold in {list(THRESHOLD_RANGE)} with validation precision >= {TARGET_PRECISION:.0%} "
                        f"and recall > {MIN_RECALL:.0%}",
                "validation": threshold_metrics(p_val, y_val, threshold),
                "test": threshold_metrics(p_test, y_test, threshold),
            },
            "reference_at_0_5": {"test": threshold_metrics(p_test, y_test, 0.5)},
            "calibration": {"test": {"roc_auc": float(roc_auc_score(y_test, p_test)),
                                     "brier": float(brier_score_loss(y_test, p_test))}},
            # Kept for consumers of the Phase 0 report format.
            "threshold_search": {"selected_threshold": threshold, "status": status},
        }
        print_report(report)
        REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
        pd.DataFrame({"probability": p_test, TARGET_COLUMN: y_test}).to_parquet(TEST_PREDICTIONS_FILE, index=False)
        log(f"Saved {REPORT_FILE.relative_to(PIPELINE_DIR)} and {TEST_PREDICTIONS_FILE.relative_to(PIPELINE_DIR)}")
    except EvaluationError as exc:
        log(f"ERROR: {exc}")
        return 1
    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
