"""Complementary figures for the mathematical model: uncertainty band, systemic luck, inflation and concept drift.

The notebook 01_math_and_eda.ipynb produces the ten base figures. This script adds six figures
that make the less intuitive mechanisms of the system visible, and it persists every number it
plots in artifacts/extra_figures_report.json, so that each figure is as traceable as the rest.

Nothing is retrained. The script reads the pipeline artifacts and reuses the exact logic of the
steps that produced them:

    04_evaluator.py      choose_band, the grids and the coin-flip simulation (imported, not re-implemented)
    05_ood_testing.py    ACS loading, ACSIncome filter, 1994 recoding, CPI deflation and transform (imported)
    consumer-price-index.ts (BFF)  CPI-U table and 2.5% projection, read through 05_ood_testing.load_cpi_table

Figures (artifacts/figures/):
    model_comparison.png          AP, ECE and mean prediction of the 11 candidates and the ensemble, natural vs SMOTE
    band_anatomy.png              Precision of each deterministic decision per cut-off and the feasible (l, u) region
    systemic_luck_simulation.png  Distribution of 2000 simulated games against the closed form (Equation 13)
    inflation_adjustment.png      CPI-U, the mobility threshold per year and the effect of deflating a capital gain
    concept_drift.png             Reliability and precision per cut-off in 1994 vs ACS 2018 (CA, NY), and the subgroup breakdown
    turno_de_noche_trace.png      The six AI paths of the 'Turno de Noche' story replayed offline against the band

The ACS part needs the folktables cache in ai-service/.folktables-cache (created by 05_ood_testing.py);
without it, pass --skip-acs and the concept-drift figure is drawn from ood_results_*.json alone.

Run from the training-pipeline folder:
    .venv/Scripts/python.exe src/generate_extra_plots.py
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parents[1]
AI_SERVICE_DIR = PIPELINE_DIR.parent
ARTIFACTS_DIR = PIPELINE_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
REPORT_FILE = ARTIFACTS_DIR / "extra_figures_report.json"

# Palette of the notebook, so the new figures read as part of the same set.
BLUE, GOLD, GREEN, RED, GREY = "#5a7bb5", "#d9a441", "#2f8f4e", "#d64545", "#8c8c8c"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})

MODEL_BASE_YEAR = 1994
STORY_YEARS = (1994, 2010, 2018, 2024, 2026, 2033)
STATES = ("CA", "NY")

# 'Turno de Noche' (cms/src/seeds/turno-de-noche.ts), in the snake_case of POST /predict.
TURNO_BASE_PROFILE = {
    "age": 24, "workclass": "Self-emp-not-inc", "education": "Some-college", "education_num": 10,
    "marital_status": "Married-civ-spouse", "occupation": "Transport-moving", "relationship": "Wife",
    "race": "Other", "sex": "Female", "hours_per_week": 72, "native_country": "Columbia",
    "capital_gain": 0, "capital_loss": 0,
}
_NURSE = {"age": 7, "workclass": "Private", "occupation": "Prof-specialty", "education": "Bachelors",
          "education_num": 3, "hours_per_week": -24}
_ALONE = {"marital_status": "Married-spouse-absent", "relationship": "Unmarried"}
TURNO_PATHS = {
    "Andrés stays -> nurse": _NURSE,
    "Andrés stays -> call-centre supervisor": {"age": 7, "workclass": "Private", "occupation": "Exec-managerial", "hours_per_week": -24},
    "Andrés migrates -> nurse": {**_NURSE, **_ALONE},
    "Andrés migrates -> live-in domestic worker": {"age": 7, "workclass": "Private", "occupation": "Priv-house-serv",
                                                  "hours_per_week": -6, **_ALONE},
    "Darién -> nursing technician (Houston)": {"age": 7, "workclass": "Private", "occupation": "Tech-support",
                                               "education": "Assoc-voc", "education_num": 1, "hours_per_week": -24},
    "Darién -> night office cleaning (Houston)": {"age": 7, "workclass": "Private", "occupation": "Other-service",
                                                  "hours_per_week": -32},
}
TURNO_YEAR = 2033
# Training-domain limits applied by the BFF's context injection (Character entity).
DOMAIN_LIMITS = {"age": (17, 90), "education_num": (1, 16), "hours_per_week": (1, 99)}


def load_module(name: str, path: Path):
    """Imports a pipeline script whose file name is not a valid module name (e.g. 04_evaluator.py)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluator = load_module("evaluator", PIPELINE_DIR / "src" / "04_evaluator.py")
ood = load_module("ood_testing", AI_SERVICE_DIR / "05_ood_testing.py")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, name: str) -> str:
    path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(PIPELINE_DIR)}")
    return str(path.relative_to(PIPELINE_DIR)).replace("\\", "/")


def precision_by_cutoff(p: np.ndarray, y: np.ndarray, grid: np.ndarray, side: str) -> dict:
    """Precision and support of one deterministic decision for every cut-off of the grid."""
    precision, support = [], []
    for cut in grid:
        mask = p >= cut if side == "upper" else p <= cut
        n = int(mask.sum())
        hits = y[mask].sum() if side == "upper" else (1 - y[mask]).sum()
        precision.append(float(hits / n) if n else None)
        support.append(n)
    return {"cutoffs": [float(c) for c in grid], "precision": precision, "support": support}


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    """Mean prediction and observed rate per equal-width bin (the ECE bins of Equation 9)."""
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = [(float(p[index == b].mean()), float(y[index == b].mean()), int((index == b).sum()))
            for b in range(bins) if (index == b).any()]
    return {"mean_predicted": [r[0] for r in rows], "observed_rate": [r[1] for r in rows], "count": [r[2] for r in rows]}


# ---------------------------------------------------------------- figure 12: model comparison

def figure_model_comparison(comparison: dict, val: pd.DataFrame) -> dict:
    observed = float(val[evaluator.TARGET_COLUMN].mean())
    families = ["Logistic Regression", "HistGradientBoosting", "Random Forest", "SVM (RBF, calibrated)", "KNN"]
    by_name = {m["name"]: m for m in comparison["models"]}
    rows = []
    for family in families:
        for training_set in ("natural", "smote"):
            model = by_name.get(f"{family} [{training_set}]")
            if model:
                v = model["validation"]
                rows.append({"family": family, "training_set": training_set, "average_precision": v["average_precision"],
                             "ece": v["ece"], "mean_predicted_probability": v["mean_predicted_probability"]})
    stack_name = comparison["selected_model"]
    stack = by_name.get(stack_name, {}).get("validation") or comparison["stacking"].get("validation")
    if stack is None:  # the ensemble's validation metrics live next to the candidates or in its own section
        stack_p = val[stack_name].to_numpy()
        stack = {"average_precision": None, "ece": None, "mean_predicted_probability": float(stack_p.mean())}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(families))
    width = 0.38
    panels = (("average_precision", "average precision (validation)", "Discrimination: SMOTE never helps"),
              ("ece", "expected calibration error (validation)", "Calibration: SMOTE multiplies the error"),
              ("mean_predicted_probability", "mean predicted probability", "SMOTE inflates the chance of mobility"))
    for ax, (key, ylabel, title) in zip(axes, panels):
        natural = [next(r[key] for r in rows if r["family"] == f and r["training_set"] == "natural") for f in families]
        smote = [next(r[key] for r in rows if r["family"] == f and r["training_set"] == "smote") for f in families]
        ax.bar(x - width / 2, natural, width, color=BLUE, label="natural 76/24 (deployed)")
        ax.bar(x + width / 2, smote, width, color=GOLD, label="SMOTE 50/50")
        if stack.get(key) is not None:
            ax.axhline(stack[key], color=GREEN, ls="--", lw=1.5, label=f"stacking ensemble: {stack[key]:.4f}")
        if key == "mean_predicted_probability":
            ax.axhline(observed, color=RED, ls=":", lw=1.8, label=f"real mobility rate: {observed:.3f}")
        if key == "average_precision":
            ax.set_ylim(0.55, 0.93)
        if key == "ece":
            ax.set_ylim(0, 0.155)
        if key == "mean_predicted_probability":
            ax.set_ylim(0, 0.5)
        ax.set_xticks(x, ["LogReg", "HGB", "RandForest", "SVM", "KNN"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right", ncol=2 if key == "mean_predicted_probability" else 1)
    path = save(fig, "model_comparison")
    return {"figure": path, "observed_positive_rate_validation": observed, "candidates": rows, "stacking_validation": stack}


# ---------------------------------------------------------------- figure 13: anatomy of the band

def figure_band_anatomy(val: pd.DataFrame, test: pd.DataFrame, policy: dict, model_name: str) -> dict:
    p_val, y_val = val[model_name].to_numpy(), val[evaluator.TARGET_COLUMN].to_numpy()
    p_test, y_test = test["probability"].to_numpy(), test[evaluator.TARGET_COLUMN].to_numpy()
    lower, upper = policy["lower_bound"], policy["upper_bound"]
    recomputed = evaluator.choose_band(p_val, y_val)

    curves = {
        "validation_upper": precision_by_cutoff(p_val, y_val, evaluator.UPPER_GRID, "upper"),
        "validation_lower": precision_by_cutoff(p_val, y_val, evaluator.LOWER_GRID, "lower"),
        "test_upper": precision_by_cutoff(p_test, y_test, evaluator.UPPER_GRID, "upper"),
        "test_lower": precision_by_cutoff(p_test, y_test, evaluator.LOWER_GRID, "lower"),
    }
    # Coverage of every (l, u) pair on validation; NaN where a constraint of choose_band fails.
    coverage = np.full((len(evaluator.LOWER_GRID), len(evaluator.UPPER_GRID)), np.nan)
    for i, lo in enumerate(evaluator.LOWER_GRID):
        neg = p_val <= lo
        neg_ok = neg.sum() >= evaluator.MIN_DECISIONS and (1 - y_val[neg]).mean() >= evaluator.TARGET_PRECISION
        for j, up in enumerate(evaluator.UPPER_GRID):
            pos = p_val >= up
            pos_ok = pos.sum() >= evaluator.MIN_DECISIONS and y_val[pos].mean() >= evaluator.TARGET_PRECISION
            if neg_ok and pos_ok:
                coverage[i, j] = (neg.sum() + pos.sum()) / len(p_val)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    ax = axes[0]
    for key, color, label in (("validation_lower", BLUE, "'no mobility' (p <= l), validation"),
                              ("test_lower", BLUE, "'no mobility' (p <= l), test")):
        c = curves[key]
        ax.plot(c["cutoffs"], c["precision"], color=color, ls="-" if "validation" in key else "--", label=label)
    for key, color, label in (("validation_upper", GOLD, "'mobility' (p >= u), validation"),
                              ("test_upper", GOLD, "'mobility' (p >= u), test")):
        c = curves[key]
        ax.plot(c["cutoffs"], c["precision"], color=color, ls="-" if "validation" in key else "--", label=label)
    ax.axhline(evaluator.TARGET_PRECISION, color=RED, ls=":", lw=1.5, label="95% precision floor")
    ax.axvspan(lower, upper, color=GREY, alpha=0.15, label=f"luck band [{lower:.2f}, {upper:.2f}]")
    ax.axvline(lower, color=BLUE, lw=1)
    ax.axvline(upper, color=GOLD, lw=1)
    ax.set_xlabel("cut-off on the predicted probability")
    ax.set_ylabel("precision of the deterministic decision")
    ax.set_ylim(0.75, 1.005)
    ax.set_title("Each edge of the band is where its decision\nstops reaching 95% precision")
    ax.legend(fontsize=7.5, loc="lower center")

    ax = axes[1]
    mesh = ax.imshow(coverage.T, origin="lower", aspect="auto", cmap="viridis",
                     extent=[evaluator.LOWER_GRID[0] - 0.005, evaluator.LOWER_GRID[-1] + 0.005,
                             evaluator.UPPER_GRID[0] - 0.005, evaluator.UPPER_GRID[-1] + 0.005])
    fig.colorbar(mesh, ax=ax, label="coverage (share decided without luck)")
    ax.scatter([lower], [upper], marker="*", s=260, color=RED, edgecolor="white", zorder=3,
               label=f"chosen band: coverage {np.nanmax(coverage):.1%}")
    ax.set_xlabel("lower bound l")
    ax.set_ylabel("upper bound u")
    ax.set_title("Feasible bands on validation (blank = a decision < 95%)\nEquation 11 picks the one with most coverage")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]
    zones = {"no mobility (p <= l)": p_test <= lower, "luck band": (p_test > lower) & (p_test < upper),
             "mobility (p >= u)": p_test >= upper}
    names = list(zones)
    real_rate = [float(y_test[m].mean()) for m in zones.values()]
    mean_p = [float(p_test[m].mean()) for m in zones.values()]
    share = [float(m.mean()) for m in zones.values()]
    xx = np.arange(3)
    ax.bar(xx - 0.2, real_rate, 0.4, color=GREEN, label="real mobility rate")
    ax.bar(xx + 0.2, mean_p, 0.4, color=GREY, label="mean predicted probability")
    for k in range(3):
        ax.text(xx[k], max(real_rate[k], mean_p[k]) + 0.03, f"{share[k]:.1%} of people", ha="center", fontsize=9)
    ax.set_xticks(xx, names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("rate (test set)")
    ax.set_title("Inside the band, mobility is close to a coin toss;\noutside it, the outcome is near certain")
    ax.legend(fontsize=8, loc="upper left")
    path = save(fig, "band_anatomy")
    return {
        "figure": path,
        "choose_band_recomputed": {"lower": recomputed[0], "upper": recomputed[1], "bands_evaluated": recomputed[2]},
        "matches_traceability_report": recomputed[0] == lower and recomputed[1] == upper,
        "max_feasible_coverage_validation": float(np.nanmax(coverage)),
        "feasible_bands": int(np.isfinite(coverage).sum()),
        "precision_curves": curves,
        "test_zones": {n: {"share": s, "real_rate": r, "mean_probability": m} for n, s, r, m in zip(names, share, real_rate, mean_p)},
    }


# ---------------------------------------------------------------- figure 14: systemic luck

def figure_systemic_luck(test: pd.DataFrame, policy: dict, report: dict) -> dict:
    p, y = test["probability"].to_numpy(), test[evaluator.TARGET_COLUMN].to_numpy()
    lower, upper = policy["lower_bound"], policy["upper_bound"]
    # Same procedure and seed as 04_evaluator.game_accuracy, keeping every simulated value.
    deterministic = np.where(p >= upper, 1, np.where(p <= lower, 0, -1))
    uncertain = deterministic == -1
    correct_deterministic = (deterministic[~uncertain] == y[~uncertain]).sum()
    rng = np.random.default_rng(evaluator.RANDOM_STATE)
    simulated = np.array([(correct_deterministic + ((rng.random(int(uncertain.sum())) < p[uncertain]).astype(int)
                                                    == y[uncertain]).sum()) / len(p)
                          for _ in range(evaluator.COIN_FLIP_SIMULATIONS)])
    closed_form = evaluator.game_accuracy(p, y, lower, upper)
    variance = float((p[uncertain] * (1 - p[uncertain])).sum() / len(p) ** 2)
    reference_05 = report["reference_at_0_5"]["test"]["accuracy"]
    covered = report["selective_classification"]["test"]["accuracy_on_covered"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    ax.hist(simulated, bins=40, color=BLUE, alpha=0.85, label=f"{len(simulated)} simulated games (seed 42)")
    grid = np.linspace(simulated.min(), simulated.max(), 300)
    sd = np.sqrt(variance)
    bin_width = (simulated.max() - simulated.min()) / 40
    ax.plot(grid, len(simulated) * bin_width * np.exp(-0.5 * ((grid - closed_form["expected_accuracy_all_players"]) / sd) ** 2)
            / (sd * np.sqrt(2 * np.pi)), color=GREEN, lw=2, label=f"normal from Eq. 13: sd = {sd:.4f}")
    ax.axvline(closed_form["expected_accuracy_all_players"], color=RED, lw=2,
               label=f"closed form E[accuracy] = {closed_form['expected_accuracy_all_players']:.2%}")
    lo, hi = np.percentile(simulated, [2.5, 97.5])
    ax.axvspan(lo, hi, color=GREY, alpha=0.15, label=f"95% of games: {lo:.2%} - {hi:.2%}")
    ax.set_xlabel("share of the 9,758 test characters whose game outcome matches census reality")
    ax.set_ylabel("simulated games")
    ax.set_title("Systemic luck is random per player, predictable over many games")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = ["decided cases\n(outside band)", "every player\n(with luck)", "plain classifier\n(threshold 0.5)"]
    values = [covered, closed_form["expected_accuracy_all_players"], reference_05]
    bars = ax.bar(labels, values, color=[GREEN, BLUE, GREY])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.2%}", ha="center")
    ax.axhline(0.95, color=RED, ls=":", label="95%")
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("accuracy (test set)")
    ax.set_title("Where the 95% holds, and its deliberate cost")
    ax.legend(fontsize=8)
    path = save(fig, "systemic_luck_simulation")
    return {
        "figure": path, "simulations": int(len(simulated)), "uncertain_cases": int(uncertain.sum()),
        "closed_form_expected_accuracy": closed_form["expected_accuracy_all_players"],
        "simulated_mean": float(simulated.mean()), "simulated_95_range": [float(lo), float(hi)],
        "matches_traceability_report": [float(lo), float(hi)] == report["game_mechanic"]["test"]["simulated_accuracy_95_range"],
        "analytic_standard_deviation": float(sd), "simulated_standard_deviation": float(simulated.std()),
        "accuracy_on_covered": covered, "accuracy_at_0_5": reference_05,
    }


# ---------------------------------------------------------------- shared model access

def score(profiles: list[dict], artifacts: dict) -> np.ndarray:
    """P(Y = 1 | x) for profiles in the snake_case vocabulary of POST /predict, via the production transform."""
    return artifacts["model"].predict_proba(ood.transform(pd.DataFrame(profiles), artifacts))[:, 1]


def log_odds_decomposition(profile: dict, artifacts: dict) -> dict:
    """F(x) = b + w_hgb F_hgb(x) + w_lr F_lr(x) of the stacking ensemble (Equation 8) for one profile."""
    model = artifacts["model"]
    features = ood.transform(pd.DataFrame([profile]), artifacts)
    f_hgb = float(model.named_estimators_["hgb"].decision_function(features)[0])
    f_lr = float(model.named_estimators_["lr"].decision_function(features)[0])
    b = float(model.final_estimator_.intercept_[0])
    w_hgb, w_lr = (float(w) for w in model.final_estimator_.coef_[0])
    f = b + w_hgb * f_hgb + w_lr * f_lr
    return {"F_hgb": f_hgb, "F_lr": f_lr, "b": b, "w_hgb": w_hgb, "w_lr": w_lr, "F": f,
            "p_from_formula": float(1 / (1 + np.exp(-f))), "p_from_predict_proba": float(score([profile], artifacts)[0])}


# ---------------------------------------------------------------- figure 15: inflation

def figure_inflation(artifacts: dict) -> dict:
    table, rate = ood.load_cpi_table()
    years = np.arange(1990, 2041)
    cpi = np.array([ood.consumer_price_index(int(t), table, rate) for t in years])
    published = np.array([int(t) in table for t in years])
    cpi_1994 = ood.consumer_price_index(MODEL_BASE_YEAR, table, rate)
    threshold = 50_000 * cpi / cpi_1994
    markers = {int(t): float(50_000 * ood.consumer_price_index(t, table, rate) / cpi_1994) for t in STORY_YEARS}

    # The same character in 2026 with a growing capital gain, scored with and without deflation.
    year = 2026
    deflator = cpi_1994 / ood.consumer_price_index(year, table, rate)
    profile = {**TURNO_BASE_PROFILE, "age": 45, "workclass": "Private", "occupation": "Sales", "education": "HS-grad",
               "education_num": 9, "hours_per_week": 45, "native_country": "United-States", "race": "White"}
    # From 500: a gain of exactly 0 is its own regime in the census (>90% of the rows), not a point of the curve.
    gains = np.arange(500, 30_001, 250)
    naive = score([{**profile, "capital_gain": int(g)} for g in gains], artifacts)
    deflated = score([{**profile, "capital_gain": int(round(g * deflator))} for g in gains], artifacts)
    upper = artifacts["report"]["decision_policy"]["upper_bound"]
    # First declared gain whose prediction reaches the deterministic 'mobility' zone.
    jump_naive = float(gains[np.argmax(naive >= upper)])
    jump_deflated = float(gains[np.argmax(deflated >= upper)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.plot(years[published], cpi[published], color=BLUE, lw=2, label="CPI-U published (BLS CUUR0000SA0)")
    ax.plot(years[~published | (years == max(table))], cpi[~published | (years == max(table))], color=BLUE, ls="--", lw=2,
            label=f"projection {rate:.1%}/yr after {max(table)}")
    ax.axvline(MODEL_BASE_YEAR, color=RED, ls=":", label="1994: the model's currency")
    ax.set_xlabel("year")
    ax.set_ylabel("CPI-U (annual average)")
    ax.set_title("Price index used by the BFF and by 05_ood_testing.py")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(years, threshold / 1000, color=GOLD, lw=2)
    for t, value in markers.items():
        ax.scatter([t], [value / 1000], color=RED, zorder=3)
        ax.annotate(f"{t}: ${value:,.0f}", (t, value / 1000), textcoords="offset points", xytext=(-55, 8), fontsize=8)
    ax.set_xlabel("year in which the story takes place")
    ax.set_ylabel("mobility threshold (thousand USD of that year)")
    ax.set_title("'Mobility' = more than $50,000 of 1994,\nexpressed in the money of each year")

    ax = axes[2]
    ax.plot(gains / 1000, naive, color=GREY, lw=2, ls="--", label=f"without deflation (first p >= {upper:.2f} at \\${jump_naive:,.0f})")
    ax.plot(gains / 1000, deflated, color=GREEN, lw=2, label=f"with deflation x {deflator:.3f}, as the BFF does (at \\${jump_deflated:,.0f})")
    ax.axhspan(0.23, 0.78, color=GREY, alpha=0.12, label="luck band")
    ax.axvline(jump_naive / 1000, color=GREY, lw=0.8)
    ax.axvline(jump_deflated / 1000, color=GREEN, lw=0.8)
    ax.set_xlabel(f"capital gain declared in {year} (thousand USD of {year})")
    ax.set_ylabel("predicted probability of mobility")
    ax.set_title("Without the adjustment the same gain looks\n"
                 f"{1 / deflator:.2f}x larger and mobility comes too early")
    ax.legend(fontsize=8, loc="center right")
    path = save(fig, "inflation_adjustment")
    return {
        "figure": path, "projection_rate": rate, "last_published_year": int(max(table)), "cpi_1994": cpi_1994,
        "threshold_by_year": markers, "example_year": year, "deflator_2026_to_1994": deflator,
        "example_profile": profile,
        "first_gain_reaching_mobility_zone_2026_usd": {"without_deflation": jump_naive, "with_deflation": jump_deflated},
    }


# ---------------------------------------------------------------- figure 16: concept drift

def acs_predictions(state: str, artifacts: dict) -> pd.DataFrame:
    """Scores the full ACS 2018 sample of one state exactly as 05_ood_testing.main does."""
    raw = ood.download_acs(state, 2018)
    people = ood.acsincome_filter(raw)
    features = ood.build_features(people, ood.child_households(raw), ood.deflation_factor(2018))
    features["probability"] = score_frame(features, artifacts)
    return features


def score_frame(features: pd.DataFrame, artifacts: dict) -> np.ndarray:
    return artifacts["model"].predict_proba(ood.transform(features, artifacts))[:, 1]


def subgroup_breakdown(frame: pd.DataFrame, lower: float, upper: float) -> dict:
    """Precision of both deterministic decisions by sex and by occupation group, frozen 1994 band."""
    y, p = frame["income_above_50k"].to_numpy(), frame["probability"].to_numpy()
    positive, negative = p >= upper, p <= lower
    false_positive = positive & (y == 0)
    out = {"female_share_population": float((frame["sex"] == "Female").mean()),
           "female_share_false_mobility": float((frame["sex"][false_positive] == "Female").mean()),
           "by_sex": {}, "by_occupation": {}}
    for sex in ("Female", "Male"):
        m = (frame["sex"] == sex).to_numpy()
        out["by_sex"][sex] = {
            "precision_mobility": float(y[positive & m].mean()), "n_mobility": int((positive & m).sum()),
            "precision_no_mobility": float(1 - y[negative & m].mean()), "n_no_mobility": int((negative & m).sum()),
        }
    for occupation, group in frame.groupby("occupation"):
        m = (frame["occupation"] == occupation).to_numpy()
        n = int((positive & m).sum())
        if n >= evaluator.MIN_DECISIONS:
            out["by_occupation"][occupation] = {
                "female_share": float((group["sex"] == "Female").mean()),
                "precision_mobility": float(y[positive & m].mean()), "n_mobility": n,
                "share_of_all_mobility_decisions": n / int(positive.sum()),
            }
    return out


def figure_concept_drift(test: pd.DataFrame, policy: dict, artifacts: dict, skip_acs: bool) -> dict:
    lower, upper = policy["lower_bound"], policy["upper_bound"]
    populations = {"1994 census (test)": (test["probability"].to_numpy(), test[evaluator.TARGET_COLUMN].to_numpy())}
    frames, results = {}, {}
    for state in STATES:
        results[state] = read_json(AI_SERVICE_DIR / f"ood_results_{state}.json")
        if not skip_acs:
            frames[state] = acs_predictions(state, artifacts)
            populations[f"ACS 2018 {state}"] = (frames[state]["probability"].to_numpy(),
                                                frames[state]["income_above_50k"].to_numpy())
    colors = {"1994 census (test)": BLUE, "ACS 2018 CA": GOLD, "ACS 2018 NY": RED}
    upper_grid = np.round(np.arange(0.50, 0.99, 0.01), 2)
    lower_grid = np.round(np.arange(0.01, 0.50, 0.01), 2)
    summary = {"populations": {}}

    ncols = 4 if frames else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.2))
    if frames:
        ax = axes[0]
        ax.plot([0, 1], [0, 1], color=GREY, ls=":", label="perfect calibration")
        for name, (p, y) in populations.items():
            rel = reliability(p, y)
            ax.plot(rel["mean_predicted"], rel["observed_rate"], marker="o", color=colors[name], label=name)
            summary["populations"][name] = {"n": int(len(p)), "reliability": rel}
        ax.axvspan(lower, upper, color=GREY, alpha=0.1)
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed mobility rate")
        ax.set_title("Same model, new world: above ~0.5 it promises\nmore mobility than 2018 delivers")
        ax.legend(fontsize=8)

        ax = axes[1]
        for name, (p, y) in populations.items():
            c = precision_by_cutoff(p, y, upper_grid, "upper")
            ax.plot(c["cutoffs"], c["precision"], color=colors[name], label=name)
            summary["populations"][name]["precision_mobility_by_cutoff"] = c
        ax.axhline(0.95, color=GREY, ls=":", label="95%")
        ax.axvline(upper, color="black", lw=0.8)
        ax.set_xlabel("upper cut-off u ('mobility' if p >= u)")
        ax.set_ylabel("precision of 'mobility'")
        ax.set_title("No cut-off restores 95% for 'mobility' in 2018:\nan error of ranking, not of threshold")
        ax.legend(fontsize=8)

        ax = axes[2]
        for name, (p, y) in populations.items():
            c = precision_by_cutoff(p, y, lower_grid, "lower")
            ax.plot(c["cutoffs"], c["precision"], color=colors[name], label=name)
            summary["populations"][name]["precision_no_mobility_by_cutoff"] = c
        for state, color in (("CA", GOLD), ("NY", RED)):
            cut = results[state]["adaptation"]["one_sided_adaptation"]["lower_cutoff"]
            ax.axvline(cut, color=color, ls="--", lw=1, label=f"recalibrated cut-off {state}: {cut:.2f}")
        ax.axhline(0.95, color=GREY, ls=":")
        ax.axvline(lower, color="black", lw=0.8)
        ax.set_xlabel("lower cut-off l ('no mobility' if p <= l)")
        ax.set_ylabel("precision of 'no mobility'")
        ax.set_title("'No mobility' recovers 95% by moving only l")
        ax.legend(fontsize=8)

    ax = axes[-1]
    if frames:
        summary["subgroups"] = {s: subgroup_breakdown(frames[s], lower, upper) for s in STATES}
        labels = ["women", "men", "Adm-clerical", "Prof-specialty", "Exec-managerial"]
        x = np.arange(len(labels))
        for k, (state, color) in enumerate((("CA", GOLD), ("NY", RED))):
            sub = summary["subgroups"][state]
            vals = [sub["by_sex"]["Female"]["precision_mobility"], sub["by_sex"]["Male"]["precision_mobility"],
                    *(sub["by_occupation"].get(o, {}).get("precision_mobility", np.nan) for o in labels[2:])]
            ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, color=color, label=f"ACS 2018 {state}")
        ax.axhline(0.95, color=GREY, ls=":", label="95%")
        ax.set_xticks(x, labels, rotation=15)
        ax.set_ylabel("precision of 'mobility' (frozen band, p >= 0.78)")
        ax.set_title("The false 'mobility' promises fall on women\nand feminised occupations")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    else:
        names = ["1994", "CA 2018", "NY 2018"]
        ref = results["CA"]["in_distribution_1994_test"]
        mob = [ref["precision_mobility"]] + [results[s]["ood"]["selective_classification"]["precision_mobility"] for s in STATES]
        nomob = [ref["precision_no_mobility"]] + [results[s]["ood"]["selective_classification"]["precision_no_mobility"] for s in STATES]
        x = np.arange(3)
        axes[0].bar(x - 0.2, mob, 0.4, color=GOLD, label="'mobility'")
        axes[0].bar(x + 0.2, nomob, 0.4, color=BLUE, label="'no mobility'")
        axes[0].set_xticks(x, names)
        axes[0].axhline(0.95, color=RED, ls=":")
        axes[0].legend()
    path = save(fig, "concept_drift")
    summary["figure"] = path
    summary["source_results"] = [f"ood_results_{s}.json" for s in STATES]
    return summary


# ---------------------------------------------------------------- figure 17: a story replayed

def inject(base: dict, modifiers: dict) -> dict:
    """The BFF's context injection (context-injection.ts): integers add, strings replace, then domain clamping."""
    profile = dict(base)
    for trait, value in modifiers.items():
        if isinstance(value, str):
            profile[trait] = value
        else:
            low, high = DOMAIN_LIMITS.get(trait, (0, float("inf")))
            profile[trait] = int(min(max(profile[trait] + value, low), high))
    return profile


def figure_turno_de_noche(policy: dict, artifacts: dict) -> dict:
    lower, upper = policy["lower_bound"], policy["upper_bound"]
    table, rate = ood.load_cpi_table()
    deflator = ood.consumer_price_index(MODEL_BASE_YEAR, table, rate) / ood.consumer_price_index(TURNO_YEAR, table, rate)
    rows = [("Daniela today (base profile, 2026)", TURNO_BASE_PROFILE)]
    rows += [(name, inject(TURNO_BASE_PROFILE, mods)) for name, mods in TURNO_PATHS.items()]
    model_inputs = []
    for _, profile in rows:  # stage 2 of resolve-decision: money deflated to 1994 dollars
        model_inputs.append({**profile, "capital_gain": round(profile["capital_gain"] * deflator),
                             "capital_loss": round(profile["capital_loss"] * deflator)})
    p = score(model_inputs, artifacts)
    zones = ["mobility" if v >= upper else "no_mobility" if v <= lower else "uncertain" for v in p]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    y = np.arange(len(rows))[::-1]
    colors = [GREEN if z == "mobility" else BLUE if z == "no_mobility" else GOLD for z in zones]
    ax.axvspan(lower, upper, color=GREY, alpha=0.15, label=f"luck band [{lower:.2f}, {upper:.2f}]: weighted coin")
    ax.barh(y, p, color=colors)
    for yy, value, zone in zip(y, p, zones):
        ax.text(value + 0.01, yy, f"p = {value:.3f}  ({'coin' if zone == 'uncertain' else 'deterministic no' if zone == 'no_mobility' else 'deterministic yes'})",
                va="center", fontsize=9)
    ax.set_yticks(y, [name for name, _ in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(mobility) returned by POST /predict for the injected profile")
    ax.set_title("'Turno de Noche' replayed offline: the hidden modifiers of each decision move Daniela\n"
                 "into the luck band while the family holds together, and out of it when migration breaks it")
    ax.legend(fontsize=8, loc="lower right")
    path = save(fig, "turno_de_noche_trace")
    return {
        "figure": path, "story_year": TURNO_YEAR, "deflator_to_1994": deflator,
        "paths": [{"path": name, "model_input": model_input, "probability": float(value), "zone": zone}
                  for (name, _), model_input, value, zone in zip(rows, model_inputs, p, zones)],
        "worked_example_log_odds": log_odds_decomposition(model_inputs[1], artifacts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-acs", action="store_true", help="do not reload the ACS 2018 microdata")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report = read_json(ARTIFACTS_DIR / "traceability_report.json")
    comparison = read_json(ARTIFACTS_DIR / "model_comparison.json")
    policy = report["decision_policy"]
    model_name = report["model"]["name"]
    val = pd.read_parquet(ARTIFACTS_DIR / "validation_predictions.parquet")
    test = pd.read_parquet(ARTIFACTS_DIR / "test_predictions.parquet")
    artifacts = ood.load_artifacts()
    if artifacts["model_name"] != model_name or joblib.load(ARTIFACTS_DIR / "best_model.joblib").__class__.__name__ != report["model"]["class"]:
        raise SystemExit("best_model.joblib does not match traceability_report.json")

    print("Generating complementary figures")
    extra = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "producer": "src/generate_extra_plots.py",
        "model": model_name,
        "decision_policy": {"lower_bound": policy["lower_bound"], "upper_bound": policy["upper_bound"]},
        "model_comparison": figure_model_comparison(comparison, val),
        "band_anatomy": figure_band_anatomy(val, test, policy, model_name),
        "systemic_luck": figure_systemic_luck(test, policy, report),
        "inflation": figure_inflation(artifacts),
        "concept_drift": figure_concept_drift(test, policy, artifacts, args.skip_acs),
        "turno_de_noche": figure_turno_de_noche(policy, artifacts),
    }
    REPORT_FILE.write_text(json.dumps(extra, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved {REPORT_FILE.relative_to(PIPELINE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
