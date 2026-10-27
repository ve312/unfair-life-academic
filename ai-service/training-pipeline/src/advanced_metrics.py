"""Advanced evaluation of the mobility model: discrimination, resampling, capacity, residuals and the accuracy ceiling.

Steps 03 and 04 select and evaluate the model with one validation split and one test split. This
script adds the statistical checks a reviewer would ask for next, and writes every number it
plots to artifacts/advanced_metrics_report.json. Model configurations are imported from
03_trainer.py (candidates, stacking_ensemble, expected_calibration_error) and the decision policy
from 04_evaluator.py, so nothing here is a re-implementation that could drift from production.

    1. Discrimination   ROC and precision-recall curves of every candidate (validation), Gini and
                        Kolmogorov-Smirnov statistics, 2000-resample bootstrap intervals on the
                        test set, and paired bootstrap differences against the deployed ensemble.
    2. Resampling       5 x 2 repeated stratified cross-validation on the training set only, with
                        the Nadeau-Bengio corrected resampled t-test (stacking vs gradient boosting).
    3. Capacity         Learning curves (training size) and a hyperparameter sensitivity grid of the
                        gradient boosting base learner, plus its early-stopping trace.
    4. Residuals        Binned residual plot, Murphy decomposition of the Brier score, calibration
                        slope and intercept, and calibration and selective precision by subgroup.
    5. Ceiling          Label conflicts among identical census profiles, the Bayes-error estimate of a
                        calibrated model, and the risk-coverage curve of selective classification.
    6. 95% guarantee    Wilson, Clopper-Pearson and Jeffreys intervals, the posterior probability of
                        reaching 95%, the sample size a 95% lower bound would need, and the band a
                        conservative (lower-bound) rule would have chosen.

The test set is only scored with already-fitted artifacts; no choice is made on it.

Run from the training-pipeline folder (about 10 minutes, dominated by cross-validation):
    .venv/Scripts/python.exe src/advanced_metrics.py
    .venv/Scripts/python.exe src/advanced_metrics.py --skip-cv   # sections 1, 4, 5 and 6 only (keeps the last CV numbers)
"""

import argparse
import importlib.util
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold

PIPELINE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PIPELINE_DIR / "data" / "processed"
ARTIFACTS_DIR = PIPELINE_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
REPORT_FILE = ARTIFACTS_DIR / "advanced_metrics_report.json"

BLUE, GOLD, GREEN, RED, GREY, PURPLE, TEAL = "#5a7bb5", "#d9a441", "#2f8f4e", "#d64545", "#8c8c8c", "#8e6bb8", "#3a9e9e"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})

BOOTSTRAP = 2000
CV_SPLITS, CV_REPEATS = 5, 2
LEARNING_FRACTIONS = (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
LEARNING_SEEDS = (0, 1, 2)
GRID_LEARNING_RATE = (0.02, 0.05, 0.10, 0.20)
GRID_MAX_LEAF_NODES = (8, 15, 31, 63)
TRAITS = ["age", "workclass", "education", "education_num", "marital_status", "occupation", "relationship",
          "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country"]


def load_module(name: str, path: Path):
    """Imports a pipeline script whose file name is not a valid module name (e.g. 03_trainer.py)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_module("trainer", PIPELINE_DIR / "src" / "03_trainer.py")
evaluator = load_module("evaluator", PIPELINE_DIR / "src" / "04_evaluator.py")
SEED = trainer.RANDOM_STATE
TARGET = trainer.TARGET_COLUMN


def save(fig, name: str) -> str:
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(PIPELINE_DIR)}")
    return str(path.relative_to(PIPELINE_DIR)).replace("\\", "/")


def read_split(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.read_parquet(PROCESSED_DIR / f"X_{name}.parquet").astype("float64")
    y = pd.read_parquet(PROCESSED_DIR / f"y_{name}.parquet")[TARGET].to_numpy()
    return X, y


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between the score distributions of the two classes."""
    return float(stats.ks_2samp(p[y == 1], p[y == 0]).statistic)


def scalar_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    auc = roc_auc_score(y, p)
    return {"roc_auc": float(auc), "gini": float(2 * auc - 1), "average_precision": float(average_precision_score(y, p)),
            "ks": ks_statistic(y, p), "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12))),
            "ece": float(trainer.expected_calibration_error(y, p))}


def interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


# ---------------------------------------------------------------- 1. discrimination

def section_discrimination(val: pd.DataFrame, test: pd.DataFrame, deployed: str) -> dict:
    y_val = val[TARGET].to_numpy()
    natural = [c for c in val.columns if c.endswith("[natural]")]
    per_model = {name: scalar_metrics(y_val, val[name].to_numpy()) for name in natural}

    # Paired bootstrap on validation: the same resample scores every model, so the differences are paired.
    rng = np.random.default_rng(SEED)
    n = len(y_val)
    deltas = {name: {"roc_auc": [], "average_precision": [], "brier": []} for name in natural if name != deployed}
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, n)
        yb = y_val[idx]
        ref = val[deployed].to_numpy()[idx]
        ref_auc, ref_ap, ref_brier = roc_auc_score(yb, ref), average_precision_score(yb, ref), brier_score_loss(yb, ref)
        for name in deltas:
            pb = val[name].to_numpy()[idx]
            deltas[name]["roc_auc"].append(ref_auc - roc_auc_score(yb, pb))
            deltas[name]["average_precision"].append(ref_ap - average_precision_score(yb, pb))
            deltas[name]["brier"].append(ref_brier - brier_score_loss(yb, pb))
    paired = {name: {metric: {"delta": float(np.mean(v)), "ci95": interval(np.array(v)),
                              "share_of_resamples_favouring_ensemble": float(np.mean(np.array(v) < 0) if metric == "brier" else np.mean(np.array(v) > 0))}
                     for metric, v in d.items()} for name, d in deltas.items()}

    # Test set: the deployed ensemble only, with percentile bootstrap intervals.
    y_test, p_test = test[TARGET].to_numpy(), test["probability"].to_numpy()
    point = scalar_metrics(y_test, p_test)
    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in ("roc_auc", "gini", "average_precision", "ks", "brier")}
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(y_test), len(y_test))
        yb, pb = y_test[idx], p_test[idx]
        auc = roc_auc_score(yb, pb)
        boot["roc_auc"].append(auc)
        boot["gini"].append(2 * auc - 1)
        boot["average_precision"].append(average_precision_score(yb, pb))
        boot["ks"].append(ks_statistic(yb, pb))
        boot["brier"].append(brier_score_loss(yb, pb))
    test_ci = {k: interval(np.array(v)) for k, v in boot.items()}

    colors = [BLUE, GOLD, GREEN, RED, PURPLE, TEAL, "black"]
    order = sorted(natural, key=lambda c: -per_model[c]["average_precision"])
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    for name, color in zip(order, colors):
        lw = 2.6 if name == deployed else 1.3
        fpr, tpr, _ = roc_curve(y_val, val[name])
        axes[0].plot(fpr, tpr, color=color, lw=lw, label=f"{name.replace(' [natural]', '')}: AUC {per_model[name]['roc_auc']:.3f}, Gini {per_model[name]['gini']:.3f}")
        prec, rec, _ = precision_recall_curve(y_val, val[name])
        axes[1].plot(rec, prec, color=color, lw=lw, label=f"{name.replace(' [natural]', '')}: AP {per_model[name]['average_precision']:.3f}")
    axes[0].plot([0, 1], [0, 1], color=GREY, ls=":", label="chance (AUC 0.5)")
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC curves (validation, natural training set)")
    axes[0].legend(fontsize=7, loc="lower right")
    axes[1].axhline(y_val.mean(), color=GREY, ls=":", label=f"chance (AP = prevalence {y_val.mean():.3f})")
    axes[1].axhline(0.95, color=RED, ls="--", lw=0.8, label="95% precision")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall curves (validation)")
    axes[1].legend(fontsize=7, loc="lower left")

    ax = axes[2]
    grid = np.linspace(0, 1, 501)
    cdf_pos = np.searchsorted(np.sort(p_test[y_test == 1]), grid, side="right") / (y_test == 1).sum()
    cdf_neg = np.searchsorted(np.sort(p_test[y_test == 0]), grid, side="right") / (y_test == 0).sum()
    k = int(np.argmax(cdf_neg - cdf_pos))
    ax.plot(grid, cdf_neg, color=BLUE, label="<= $50K (real)")
    ax.plot(grid, cdf_pos, color=GOLD, label="> $50K (real)")
    ax.vlines(grid[k], cdf_pos[k], cdf_neg[k], color=RED, lw=2, label=f"KS = {point['ks']:.3f} at p = {grid[k]:.2f}")
    ax.axvspan(evaluator_band()[0], evaluator_band()[1], color=GREY, alpha=0.12, label="luck band")
    ax.set_xlabel("predicted probability (test)")
    ax.set_ylabel("cumulative share of the class")
    ax.set_title(f"Class separation on test: AUC {point['roc_auc']:.3f} "
                 f"[{test_ci['roc_auc'][0]:.3f}, {test_ci['roc_auc'][1]:.3f}]")
    ax.legend(fontsize=8, loc="lower right")
    path = save(fig, "discrimination_curves")
    return {"figure": path, "validation": per_model, "paired_bootstrap_vs_deployed": paired,
            "test": {"point": point, "bootstrap_ci95": test_ci, "resamples": BOOTSTRAP}}


_BAND = None


def evaluator_band() -> tuple[float, float]:
    global _BAND
    if _BAND is None:
        policy = json.loads((ARTIFACTS_DIR / "traceability_report.json").read_text(encoding="utf-8"))["decision_policy"]
        _BAND = (policy["lower_bound"], policy["upper_bound"])
    return _BAND


# ---------------------------------------------------------------- 2. cross-validation

def corrected_t_test(differences: np.ndarray, n_train: int, n_test: int) -> dict:
    """Nadeau and Bengio (2003): resampled t-test with the variance inflated for overlapping training sets."""
    k = len(differences)
    mean, var = differences.mean(), differences.var(ddof=1)
    t = mean / np.sqrt((1 / k + n_test / n_train) * var) if var > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return {"mean_difference": float(mean), "t": float(t), "df": k - 1, "p_value": float(p)}


def section_cross_validation(X: pd.DataFrame, y: np.ndarray) -> dict:
    models = {**{name: model for name, model in trainer.candidates().items() if "class-weighted" not in name},
              "Stacking (HGB + LR -> LR)": trainer.stacking_ensemble()}
    splitter = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED)
    folds = list(splitter.split(X, y))
    scores = {name: {k: [] for k in ("roc_auc", "average_precision", "brier", "ece", "log_loss")} for name in models}
    seconds = {}
    for name, model in models.items():
        start = time.perf_counter()
        for train_idx, test_idx in folds:
            fitted = clone(model).fit(X.iloc[train_idx], y[train_idx])
            p = fitted.predict_proba(X.iloc[test_idx])[:, 1]
            m = scalar_metrics(y[test_idx], p)
            for key in scores[name]:
                scores[name][key].append(m[key])
        seconds[name] = time.perf_counter() - start
        print(f"  CV {name}: AP {np.mean(scores[name]['average_precision']):.4f} in {seconds[name]:.0f}s")

    n_test = len(folds[0][1])
    n_train = len(X) - n_test
    stack, hgb = "Stacking (HGB + LR -> LR)", "HistGradientBoosting"
    tests = {metric: corrected_t_test(np.array(scores[stack][metric]) - np.array(scores[hgb][metric]), n_train, n_test)
             for metric in ("roc_auc", "average_precision", "brier", "ece")}
    # Friedman test over the per-fold average precision of all families: are the models distinguishable at all?
    friedman = stats.friedmanchisquare(*[scores[name]["average_precision"] for name in models])
    mean_ranks = pd.DataFrame({name: scores[name]["average_precision"] for name in models}).rank(axis=1, ascending=False).mean()

    names = sorted(models, key=lambda n: -np.mean(scores[n]["average_precision"]))
    short = {"Logistic Regression": "LogReg", "KNN": "KNN", "SVM (RBF, calibrated)": "SVM", "Random Forest": "RandForest",
             "HistGradientBoosting": "HGB", "Stacking (HGB + LR -> LR)": "Stacking"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, label in zip(axes, ("average_precision", "brier", "ece"),
                                 ("average precision (higher is better)", "Brier score (lower is better)",
                                  "expected calibration error (lower is better)")):
        data = [scores[n][metric] for n in names]
        box = ax.boxplot(data, patch_artist=True, widths=0.6)
        for patch, n in zip(box["boxes"], names):
            patch.set_facecolor(GREEN if n.startswith("Stacking") else BLUE)
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(names) + 1), [short[n] for n in names], rotation=15)
        ax.set_ylabel(label)
    axes[0].set_title(f"{CV_REPEATS}x{CV_SPLITS} stratified CV on the training set\n"
                      f"Friedman chi2 = {friedman.statistic:.1f}, p = {friedman.pvalue:.1e}")
    axes[1].set_title("Probabilistic accuracy per fold")
    axes[2].set_title(f"Calibration per fold\nstacking - HGB: corrected t-test p = {tests['ece']['p_value']:.3f}")
    path = save(fig, "cross_validation")
    summary = {name: {metric: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "folds": [float(x) for x in v]}
                      for metric, v in s.items()} for name, s in scores.items()}
    return {"figure": path, "scheme": f"RepeatedStratifiedKFold(n_splits={CV_SPLITS}, n_repeats={CV_REPEATS}, random_state={SEED}) on X_train",
            "models": summary, "seconds": seconds, "stacking_vs_hgb_corrected_t_test": tests,
            "friedman_average_precision": {"chi2": float(friedman.statistic), "p_value": float(friedman.pvalue),
                                           "mean_rank": {k: float(v) for k, v in mean_ranks.items()}}}


# ---------------------------------------------------------------- 3. capacity

def section_capacity(X: pd.DataFrame, y: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray, deployed_model) -> dict:
    base = trainer.candidates()
    curves = {"HistGradientBoosting": base["HistGradientBoosting"], "Logistic Regression": base["Logistic Regression"]}
    learning = {name: {"fraction": [], "rows": [], "average_precision": [], "brier": [], "roc_auc": [], "train_average_precision": []} for name in curves}
    for name, model in curves.items():
        for fraction in LEARNING_FRACTIONS:
            ap, brier, auc, train_ap = [], [], [], []
            for seed in LEARNING_SEEDS:
                if fraction < 1:
                    rows = pd.Series(y).groupby(y, group_keys=False).apply(lambda g: g.sample(frac=fraction, random_state=seed)).index.to_numpy()
                else:
                    rows = np.arange(len(y))
                fitted = clone(model).fit(X.iloc[rows], y[rows])
                p = fitted.predict_proba(X_val)[:, 1]
                ap.append(average_precision_score(y_val, p))
                brier.append(brier_score_loss(y_val, p))
                auc.append(roc_auc_score(y_val, p))
                train_ap.append(average_precision_score(y[rows], fitted.predict_proba(X.iloc[rows])[:, 1]))
                if fraction == 1:
                    break  # the full set has no sampling variance
            learning[name]["fraction"].append(fraction)
            learning[name]["rows"].append(int(round(fraction * len(y))))
            for key, values in (("average_precision", ap), ("brier", brier), ("roc_auc", auc), ("train_average_precision", train_ap)):
                learning[name][key].append({"mean": float(np.mean(values)), "std": float(np.std(values))})
        print(f"  learning curve {name}: done")

    grid_ap = np.zeros((len(GRID_LEARNING_RATE), len(GRID_MAX_LEAF_NODES)))
    grid_brier = np.zeros_like(grid_ap)
    grid_iters = np.zeros_like(grid_ap)
    reference = base["HistGradientBoosting"]
    for i, lr in enumerate(GRID_LEARNING_RATE):
        for j, leaves in enumerate(GRID_MAX_LEAF_NODES):
            fitted = clone(reference).set_params(learning_rate=lr, max_leaf_nodes=leaves).fit(X, y)
            p = fitted.predict_proba(X_val)[:, 1]
            grid_ap[i, j] = average_precision_score(y_val, p)
            grid_brier[i, j] = brier_score_loss(y_val, p)
            grid_iters[i, j] = fitted.n_iter_
    print("  hyperparameter grid: done")

    hgb = deployed_model.named_estimators_["hgb"]
    train_loss, val_loss = -np.asarray(hgb.train_score_), -np.asarray(hgb.validation_score_)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.2))
    ax = axes[0]
    for name, color in (("HistGradientBoosting", BLUE), ("Logistic Regression", GOLD)):
        rows = learning[name]["rows"]
        mean = np.array([v["mean"] for v in learning[name]["average_precision"]])
        sd = np.array([v["std"] for v in learning[name]["average_precision"]])
        train = [v["mean"] for v in learning[name]["train_average_precision"]]
        ax.plot(rows, mean, marker="o", color=color, label=f"{name}: validation")
        ax.fill_between(rows, mean - sd, mean + sd, color=color, alpha=0.2)
        ax.plot(rows, train, ls="--", color=color, label=f"{name}: training")
    ax.set_xlabel("training rows")
    ax.set_ylabel("average precision")
    hgb_ap = [v["mean"] for v in learning["HistGradientBoosting"]["average_precision"]]
    half = LEARNING_FRACTIONS.index(0.50)
    ax.set_title(f"Learning curves: diminishing returns\n(HGB: doubling 50% -> 100% of the rows adds {hgb_ap[-1] - hgb_ap[half]:+.4f} AP)")
    ax.legend(fontsize=7)

    ax = axes[1]
    for name, color in (("HistGradientBoosting", BLUE), ("Logistic Regression", GOLD)):
        ax.plot(learning[name]["rows"], [v["mean"] for v in learning[name]["brier"]], marker="o", color=color, label=name)
    ax.axhline(y_val.mean() * (1 - y_val.mean()), color=GREY, ls=":", label="Brier of the base rate (no model)")
    ax.set_xlabel("training rows")
    ax.set_ylabel("Brier score (validation)")
    ax.set_title("Probabilistic error vs training size")
    ax.legend(fontsize=8)

    ax = axes[2]
    image = ax.imshow(grid_ap, cmap="viridis", aspect="auto")
    for i in range(len(GRID_LEARNING_RATE)):
        for j in range(len(GRID_MAX_LEAF_NODES)):
            deployed = GRID_LEARNING_RATE[i] == 0.05 and GRID_MAX_LEAF_NODES[j] == 31
            ax.text(j, i, f"{grid_ap[i, j]:.4f}\n{int(grid_iters[i, j])} it", ha="center", va="center", fontsize=8,
                    color="white" if grid_ap[i, j] < grid_ap.mean() else "black", fontweight="bold" if deployed else "normal")
    ax.set_xticks(range(len(GRID_MAX_LEAF_NODES)), GRID_MAX_LEAF_NODES)
    ax.set_yticks(range(len(GRID_LEARNING_RATE)), GRID_LEARNING_RATE)
    ax.set_xlabel("max_leaf_nodes")
    ax.set_ylabel("learning_rate")
    fig.colorbar(image, ax=ax, label="validation average precision")
    ax.set_title(f"HGB sensitivity (bold = deployed)\nrange of AP across 16 configs: {grid_ap.max() - grid_ap.min():.4f}")

    ax = axes[3]
    ax.plot(np.arange(len(train_loss)), train_loss, color=BLUE, label="training loss")
    ax.plot(np.arange(len(val_loss)), val_loss, color=GOLD, label="internal 10% hold-out loss")
    best = int(np.argmin(val_loss))
    ax.axvline(best, color=RED, ls=":", label=f"best hold-out loss at iteration {best}")
    ax.set_xlabel("boosting iteration")
    ax.set_ylabel("binary log-loss")
    ax.set_title(f"Early stopping of the deployed HGB: {hgb.n_iter_} iterations")
    ax.legend(fontsize=8)
    path = save(fig, "capacity_analysis")
    return {"figure": path, "learning_curves": learning, "learning_seeds": list(LEARNING_SEEDS),
            "hyperparameter_grid": {"learning_rate": list(GRID_LEARNING_RATE), "max_leaf_nodes": list(GRID_MAX_LEAF_NODES),
                                    "validation_average_precision": grid_ap.tolist(), "validation_brier": grid_brier.tolist(),
                                    "iterations": grid_iters.astype(int).tolist(),
                                    "deployed": {"learning_rate": 0.05, "max_leaf_nodes": 31},
                                    "ap_range": float(grid_ap.max() - grid_ap.min())},
            "early_stopping": {"iterations": int(hgb.n_iter_), "best_holdout_iteration": best,
                               "final_train_loss": float(train_loss[-1]), "final_holdout_loss": float(val_loss[-1])}}


# ---------------------------------------------------------------- 4. residuals and subgroups

def decode(X: pd.DataFrame, prefix: str) -> pd.Series:
    columns = [c for c in X.columns if c.startswith(prefix + "_")]
    return X[columns].idxmax(axis=1).str[len(prefix) + 1:].where(X[columns].max(axis=1) > 0, "Unknown")


def murphy(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    """Brier = reliability - resolution + uncertainty (Murphy, 1973), on equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    base = y.mean()
    rel = res = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            rel += m.sum() * (p[m].mean() - y[m].mean()) ** 2
            res += m.sum() * (y[m].mean() - base) ** 2
    n = len(y)
    return {"reliability": rel / n, "resolution": res / n, "uncertainty": float(base * (1 - base)),
            "brier": float(brier_score_loss(y, p)), "skill_score": float(1 - brier_score_loss(y, p) / (base * (1 - base)))}


def section_residuals(X_test: pd.DataFrame, test: pd.DataFrame) -> dict:
    y, p = test[TARGET].to_numpy(), test["probability"].to_numpy()
    lower, upper = evaluator_band()
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    recal = LogisticRegression(C=1e6, max_iter=1000).fit(logit.reshape(-1, 1), y)
    slope, intercept = float(recal.coef_[0, 0]), float(recal.intercept_[0])
    # Calibration-in-the-large: intercept of a logistic model with logit(p) as a fixed offset.
    citl = float(np.log(y.mean() / (1 - y.mean())) - np.log(p.mean() / (1 - p.mean())))
    deviance = np.sign(y - p) * np.sqrt(-2 * (y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))))

    # Binned residuals (Gelman and Hill, 2007): mean(y - p) in quantile bins, with +-2 standard errors.
    order = np.argsort(p)
    n_bins = int(np.sqrt(len(p)))
    binned = []
    for chunk in np.array_split(order, n_bins):
        r = y[chunk] - p[chunk]
        binned.append((float(p[chunk].mean()), float(r.mean()), float(2 * np.sqrt(p[chunk].mean() * (1 - p[chunk].mean()) / len(chunk)))))
    outside = float(np.mean([abs(b[1]) > b[2] for b in binned]))

    groups = {
        "sex": decode(X_test, "sex"),
        "race": decode(X_test, "race"),
        "age band": pd.cut(X_test["age"], [16, 25, 35, 45, 55, 65, 91], labels=["17-25", "26-35", "36-45", "46-55", "56-65", "66+"]).astype(str),
        "education": pd.cut(X_test["education_num"], [0, 8, 9, 12, 13, 16], labels=["< HS", "HS-grad", "some college", "bachelors", "postgrad"]).astype(str),
    }
    subgroups = {}
    for family, labels in groups.items():
        subgroups[family] = {}
        for label in sorted(labels.unique()):
            m = (labels == label).to_numpy()
            if m.sum() < 100:
                continue
            pos, neg = m & (p >= upper), m & (p <= lower)
            subgroups[family][label] = {
                "n": int(m.sum()), "observed_rate": float(y[m].mean()), "mean_probability": float(p[m].mean()),
                "calibration_gap": float(y[m].mean() - p[m].mean()),
                "roc_auc": float(roc_auc_score(y[m], p[m])) if 0 < y[m].mean() < 1 else None,
                "precision_mobility": float(y[pos].mean()) if pos.sum() >= 30 else None, "n_mobility": int(pos.sum()),
                "precision_no_mobility": float(1 - y[neg].mean()) if neg.sum() >= 30 else None, "n_no_mobility": int(neg.sum()),
            }
    decomposition = murphy(y, p)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1.1, 1, 1.3]})
    ax = axes[0]
    xs, ms, se = zip(*binned)
    ax.fill_between(xs, -np.array(se), se, color=GREY, alpha=0.25, label="+-2 standard errors")
    ax.scatter(xs, ms, s=14, color=BLUE, label=f"{n_bins} quantile bins")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvspan(lower, upper, color=GOLD, alpha=0.08, label="luck band")
    ax.set_xlabel("mean predicted probability in the bin")
    ax.set_ylabel("mean residual  y - p")
    ax.set_title(f"Binned residuals (test): {outside:.0%} of bins outside +-2 SE\n"
                 f"calibration slope {slope:.3f}, intercept {intercept:+.3f}")
    ax.legend(fontsize=8)

    ax = axes[1]
    parts = ["uncertainty", "resolution", "reliability", "brier"]
    values = [decomposition[k] for k in parts]
    ax.bar(parts, values, color=[GREY, GREEN, RED, BLUE])
    for i, v in enumerate(values):
        ax.text(i, v + 0.003, f"{v:.4f}", ha="center")
    ax.set_title(f"Murphy decomposition of the Brier score\nBrier = REL - RES + UNC; skill score {decomposition['skill_score']:.3f}")
    ax.set_ylabel("value (test)")

    ax = axes[2]
    labels, gaps, colors = [], [], []
    palette = {"sex": BLUE, "race": GOLD, "age band": GREEN, "education": PURPLE}
    for family, rows in subgroups.items():
        for label, row in rows.items():
            labels.append(f"{family}: {label} (n={row['n']})")
            gaps.append(row["calibration_gap"])
            colors.append(palette[family])
    yy = np.arange(len(labels))[::-1]
    ax.barh(yy, gaps, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(yy, labels, fontsize=7)
    ax.set_xlabel("observed rate - mean prediction (test)")
    ax.set_title("Calibration by subgroup in 1994: gaps of a few points at most")
    path = save(fig, "residual_analysis")
    return {"figure": path, "calibration_slope": slope, "calibration_intercept": intercept,
            "calibration_in_the_large": citl, "binned_residuals": {"bins": n_bins, "share_outside_2se": outside},
            "deviance_residuals": {"mean": float(deviance.mean()), "std": float(deviance.std()),
                                   "share_abs_above_2": float(np.mean(np.abs(deviance) > 2))},
            "murphy_decomposition": decomposition, "subgroups": subgroups}


# ---------------------------------------------------------------- 5. accuracy ceiling

def section_ceiling(test: pd.DataFrame) -> dict:
    clean = pd.read_csv(PROCESSED_DIR / "clean_adult.csv", keep_default_na=False)
    y_all = clean[TARGET].to_numpy()
    profiles = clean.groupby(TRAITS)[TARGET].agg(["size", "sum"])
    repeated = profiles[profiles["size"] >= 2]
    mixed = repeated[(repeated["sum"] > 0) & (repeated["sum"] < repeated["size"])]
    exact = {
        "records": int(len(clean)), "distinct_profiles": int(len(profiles)),
        "records_in_repeated_profiles": int(repeated["size"].sum()),
        "repeated_profiles": int(len(repeated)), "mixed_label_profiles": int(len(mixed)),
        "records_in_mixed_profiles": int(mixed["size"].sum()),
        "share_of_repeated_records_with_mixed_labels": float(mixed["size"].sum() / repeated["size"].sum()),
        "unavoidable_errors_in_repeated_profiles": int(np.minimum(repeated["sum"], repeated["size"] - repeated["sum"]).sum()),
    }
    exact["minimum_error_rate_on_repeated_profiles"] = exact["unavoidable_errors_in_repeated_profiles"] / exact["records_in_repeated_profiles"]
    largest = mixed.sort_values("size", ascending=False).head(5).reset_index()
    exact["largest_mixed_profiles"] = [{**{t: (r[t] if isinstance(r[t], str) else float(r[t])) for t in TRAITS},
                                        "size": int(r["size"]), "above_50k": int(r["sum"])} for _, r in largest.iterrows()]

    # Coarser cells: the profile a narrative character is usually described by.
    coarse = clean.assign(age_band=pd.cut(clean["age"], [16, 25, 35, 45, 55, 65, 91]).astype(str),
                          hours_band=pd.cut(clean["hours_per_week"], [0, 34, 40, 50, 99]).astype(str),
                          has_capital=(clean["capital_gain"] > 0).astype(int))
    cells = coarse.groupby(["age_band", "education", "occupation", "sex", "marital_status", "hours_band", "has_capital"])[TARGET].agg(["size", "sum"])
    coarse_error = float(np.minimum(cells["sum"], cells["size"] - cells["sum"]).sum() / len(clean))

    y, p = test[TARGET].to_numpy(), test["probability"].to_numpy()
    confidence = np.maximum(p, 1 - p)
    bayes_accuracy = float(confidence.mean())
    order = np.argsort(-confidence)
    correct = ((p >= 0.5).astype(int) == y)[order]
    coverage = np.arange(1, len(y) + 1) / len(y)
    selective_accuracy = np.cumsum(correct) / np.arange(1, len(y) + 1)
    aurc = float(np.mean(1 - selective_accuracy))
    reach = coverage[np.where(selective_accuracy >= 0.95)[0].max()]
    lower, upper = evaluator_band()
    covered = (p <= lower) | (p >= upper)
    band_point = (float(covered.mean()), float(((p >= upper).astype(int) == y)[covered].mean()))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    ax = axes[0]
    sizes = [exact["records"], exact["records_in_repeated_profiles"], exact["records_in_mixed_profiles"]]
    ax.bar(["all records", "records sharing an\nidentical 13-trait profile", "...whose profile has\nboth outcomes"], sizes, color=[GREY, BLUE, RED])
    for i, v in enumerate(sizes):
        ax.text(i, v * 1.02, f"{v:,}", ha="center")
    ax.set_yscale("log")
    ax.set_ylabel("records (log scale)")
    ax.set_title(f"Identical census profiles, different incomes: {exact['share_of_repeated_records_with_mixed_labels']:.0%}\n"
                 f"of repeated records sit in a profile with both outcomes")

    ax = axes[1]
    ax.hist(confidence, bins=50, color=BLUE, alpha=0.85)
    ax.axvline(0.95, color=RED, ls="--", label=f"max(p, 1-p) >= 0.95: {np.mean(confidence >= 0.95):.1%} of people")
    ax.axvline(bayes_accuracy, color=GREEN, lw=2, label=f"mean = best achievable accuracy {bayes_accuracy:.1%}")
    ax.set_xlabel("confidence of the calibrated model, max(p, 1 - p)")
    ax.set_ylabel("people (test)")
    ax.set_title("A calibrated model knows its own ceiling:\nno rule built on these scores can expect more than mean confidence")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(coverage, selective_accuracy, color=BLUE, lw=2, label=f"risk-coverage curve (AURC = {aurc:.4f})")
    ax.axhline(0.95, color=RED, ls="--", label="95% accuracy")
    ax.axvline(reach, color=GREY, ls=":", label=f"symmetric rule reaches 95% up to {reach:.1%} coverage")
    ax.scatter([band_point[0]], [band_point[1]], color=GOLD, s=90, zorder=3,
               label=f"deployed band: {band_point[0]:.1%} coverage, {band_point[1]:.2%}")
    ax.scatter([1.0], [selective_accuracy[-1]], color="black", s=50, zorder=3, label=f"decide everyone: {selective_accuracy[-1]:.2%}")
    ax.set_xlabel("coverage (share decided, most confident first)")
    ax.set_ylabel("accuracy on the decided share")
    ax.set_ylim(0.84, 1.005)
    ax.set_title("Selective classification: 95% is reachable only\nby abstaining on about a quarter of the people")
    ax.legend(fontsize=8, loc="lower left")
    path = save(fig, "accuracy_ceiling")
    return {"figure": path, "identical_profiles": exact,
            "coarse_profiles": {"cells": int(len(cells)), "minimum_error_rate_of_any_rule_on_these_cells": coarse_error,
                                "traits": "age band, education, occupation, sex, marital status, hours band, any capital gain"},
            "calibrated_model": {"expected_best_accuracy": bayes_accuracy, "estimated_bayes_error": 1 - bayes_accuracy,
                                 "share_with_confidence_at_least_95": float(np.mean(confidence >= 0.95)),
                                 "accuracy_at_0_5": float(selective_accuracy[-1])},
            "risk_coverage": {"aurc": aurc, "max_coverage_at_95_accuracy": float(reach),
                              "deployed_band": {"coverage": band_point[0], "accuracy": band_point[1]}}}


# ---------------------------------------------------------------- 6. the 95% guarantee

def proportion_intervals(successes: int, n: int) -> dict:
    wilson = evaluator.wilson(successes, n)
    clopper = [float(stats.beta.ppf(0.025, successes, n - successes + 1)), float(stats.beta.ppf(0.975, successes + 1, n - successes))]
    jeffreys = [float(stats.beta.ppf(0.025, successes + 0.5, n - successes + 0.5)), float(stats.beta.ppf(0.975, successes + 0.5, n - successes + 0.5))]
    posterior = stats.beta(successes + 1, n - successes + 1)
    estimate = successes / n
    margin = estimate - 0.95
    needed = int(np.ceil((1.959964 ** 2) * estimate * (1 - estimate) / margin ** 2)) if margin > 0 else None
    return {"successes": successes, "n": n, "estimate": estimate, "wilson": wilson, "clopper_pearson": clopper,
            "jeffreys": jeffreys, "posterior_probability_at_least_95": float(posterior.sf(0.95)),
            "one_sided_p_value_h0_below_95": float(stats.binom.sf(successes - 1, n, 0.95)),
            "decisions_needed_for_wilson_lower_bound_95": needed}


def conservative_band(p: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """choose_band with the Wilson lower bound, instead of the point estimate, held to 95%."""
    best = None
    for lower in evaluator.LOWER_GRID:
        neg = p <= lower
        if neg.sum() < evaluator.MIN_DECISIONS or evaluator.wilson(int((1 - y[neg]).sum()), int(neg.sum()))[0] < 0.95:
            continue
        for upper in evaluator.UPPER_GRID:
            pos = p >= upper
            if pos.sum() < evaluator.MIN_DECISIONS or evaluator.wilson(int(y[pos].sum()), int(pos.sum()))[0] < 0.95:
                continue
            coverage = (neg.sum() + pos.sum()) / len(p)
            if best is None or coverage > best[0]:
                best = (coverage, float(lower), float(upper))
    return None if best is None else (best[1], best[2])


def section_guarantee(val: pd.DataFrame, test: pd.DataFrame, deployed: str, report: dict) -> dict:
    y, p = test[TARGET].to_numpy(), test["probability"].to_numpy()
    lower, upper = evaluator_band()
    pos, neg = p >= upper, p <= lower
    decisions = {"mobility": proportion_intervals(int(y[pos].sum()), int(pos.sum())),
                 "no_mobility": proportion_intervals(int((1 - y[neg]).sum()), int(neg.sum())),
                 "all_decided": proportion_intervals(int(y[pos].sum() + (1 - y[neg]).sum()), int(pos.sum() + neg.sum()))}

    y_val, p_val = val[TARGET].to_numpy(), val[deployed].to_numpy()
    band = conservative_band(p_val, y_val)
    alternative = None
    if band is not None:
        sel = evaluator.selective_metrics(p, y, band[0], band[1])
        game = evaluator.game_accuracy(p, y, band[0], band[1])
        alternative = {"band": list(band), "test_coverage": sel["coverage"], "test_precision_mobility": sel["precision_mobility"],
                       "test_precision_no_mobility": sel["precision_no_mobility"],
                       "test_precision_no_mobility_95ci": sel["precision_no_mobility_95ci"],
                       "test_expected_game_accuracy": game["expected_accuracy_all_players"],
                       "coverage_lost_vs_deployed": report["selective_classification"]["test"]["coverage"] - sel["coverage"]}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    grid = np.linspace(0.925, 0.985, 600)
    for key, color, label in (("mobility", GOLD, "'mobility'"), ("no_mobility", BLUE, "'no mobility'"), ("all_decided", GREEN, "all decided")):
        d = decisions[key]
        density = stats.beta.pdf(grid, d["successes"] + 1, d["n"] - d["successes"] + 1)
        ax.plot(grid, density, color=color, lw=2, label=f"{label}: n = {d['n']}, P(precision >= 95%) = {d['posterior_probability_at_least_95']:.2f}")
        ax.fill_between(grid, density, where=grid >= 0.95, color=color, alpha=0.15)
    ax.axvline(0.95, color=RED, ls="--")
    ax.set_xlabel("true precision of the deterministic decision")
    ax.set_ylabel("posterior density (uniform prior, test set)")
    ax.set_title("How sure are we that each decision reaches 95%?")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = ["deployed band\n(point estimate >= 95%)", "conservative band\n(Wilson lower bound >= 95%)"]
    coverage_values = [report["selective_classification"]["test"]["coverage"], alternative["test_coverage"] if alternative else np.nan]
    game_values = [report["game_mechanic"]["test"]["expected_accuracy_all_players"],
                   alternative["test_expected_game_accuracy"] if alternative else np.nan]
    x = np.arange(2)
    ax.bar(x - 0.2, coverage_values, 0.4, color=BLUE, label="coverage (decided without luck)")
    ax.bar(x + 0.2, game_values, 0.4, color=GREEN, label="expected game accuracy")
    for i in range(2):
        ax.text(x[i] - 0.2, coverage_values[i] + 0.01, f"{coverage_values[i]:.1%}", ha="center")
        ax.text(x[i] + 0.2, game_values[i] + 0.01, f"{game_values[i]:.1%}", ha="center")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    title = "What a 'lower bound >= 95%' rule would cost (test)"
    if alternative:
        title += f"\nconservative band [{band[0]:.2f}, {band[1]:.2f}]"
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    path = save(fig, "precision_guarantee")
    return {"figure": path, "test_decisions": decisions, "conservative_alternative": alternative}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-cv", action="store_true", help="skip cross-validation and capacity analysis (sections 2 and 3)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads((ARTIFACTS_DIR / "traceability_report.json").read_text(encoding="utf-8"))
    deployed = report["model"]["name"]
    val = pd.read_parquet(ARTIFACTS_DIR / "validation_predictions.parquet")
    test = pd.read_parquet(ARTIFACTS_DIR / "test_predictions.parquet")
    X_train, y_train = read_split("train")
    X_val, y_val = read_split("validation")
    X_test, _ = read_split("test")
    model = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    if not np.allclose(model.predict_proba(X_test)[:, 1], test["probability"].to_numpy()):
        raise SystemExit("best_model.joblib does not reproduce test_predictions.parquet")

    print("Advanced evaluation")
    out = {"generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
           "producer": "src/advanced_metrics.py", "model": deployed, "random_state": SEED}
    out["discrimination"] = section_discrimination(val, test, deployed)
    if not args.skip_cv:
        out["cross_validation"] = section_cross_validation(X_train, y_train)
        out["capacity"] = section_capacity(X_train, y_train, X_val, y_val, model)
    out["residuals"] = section_residuals(X_test, test)
    out["accuracy_ceiling"] = section_ceiling(test)
    out["precision_guarantee"] = section_guarantee(val, test, deployed, report)
    previous = json.loads(REPORT_FILE.read_text(encoding="utf-8")) if REPORT_FILE.exists() and args.skip_cv else {}
    for key in ("cross_validation", "capacity"):
        if key not in out and key in previous:
            out[key] = previous[key]  # keep the last full run instead of dropping its numbers
    REPORT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved {REPORT_FILE.relative_to(PIPELINE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
