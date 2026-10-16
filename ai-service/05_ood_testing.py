"""Out-of-distribution (OOD) test of the deployed model on the ACS Income 2018 data (folktables).

The model was trained on the 1994 Adult Income census. This script measures how well it predicts
on people it has never seen, drawn from a different population, a different survey and a
different economic era: the 2018 American Community Survey (ACS) Public Use Microdata Sample,
in the formulation of Ding et al. (2021), "Retiring Adult: New Datasets for Fair Machine Learning",
available through the `folktables` package.

Protocol. The model is loaded from its artifacts and is never retrained or recalibrated.

1. Data. The ACS person records of one state are downloaded (California by default) and
   filtered with the ACSIncome rule: age > 16, personal income > 100, hours worked > 0, and a
   person weight >= 1.
2. Temporal normalisation. Incomes are first brought to 2018 dollars with the survey's ADJINC
   factor, then deflated to 1994 dollars with the CPI-U table that the BFF uses
   (bff-api/src/core/domain/services/consumer-price-index.ts, read at runtime so that the two can
   never diverge): x_1994 = x_2018 * CPI(1994) / CPI(2018). The target is computed AFTER this
   deflation, as y = 1(PINCP_1994 > 50,000), which is the Adult definition, so that both
   datasets ask the same economic question. In 2018 dollars, the threshold is about USD 84,700.
3. Feature mapping. The ACS variables are recoded into the 13 census traits the model expects,
   with the 1994 spellings (see the mapping functions below for each correspondence). ACS has no
   capital gains or losses: both are set to 0, the value of more than 90% of the Adult records.
   Interest and dividend income (INTP) is a different concept and is deliberately not used,
   because it would activate the model's capital-gain threshold spuriously.
4. Inference. The serialised imputer and encoder transform the rows exactly as the inference
   API does. A category absent from the 1994 vocabulary is encoded as all zeros, which mirrors
   production. The stacking ensemble then returns P(Y = 1 | x).
5. Metrics. The script reports discrimination (ROC-AUC, average precision), calibration
   (Brier score, ECE, mean prediction versus observed rate), the selective-classification policy
   frozen on the 1994 validation set (coverage, precision of each deterministic decision with
   Wilson intervals, behaviour inside the uncertainty band), and the game's expected accuracy.
   Each figure is set against the in-distribution test results from traceability_report.json.
   Unweighted metrics describe the sample; survey-weighted ones (PWGTP) describe the state's
   working population.

6. Adaptation (detect, quantify, correct). The model is never retrained, but the
   selective-classification band is re-estimated for the new population. The 2018 sample is
   split 50/50, stratified on the target (seed 42). On the adaptation half, the band is searched
   with exactly the rule of step 04 (04_evaluator.choose_band, imported, not re-implemented):
   maximum coverage subject to precision >= 95% for both deterministic decisions, each backed by
   at least 30 cases. The frozen 1994 band and the adapted band are then both evaluated on the
   held-out half, which takes no part in the search, so the comparison is fair and free of
   leakage. This is threshold recalibration, a standard response to distribution shift: the
   ranking learned in 1994 is kept, and only the cut-offs move to restore the precision
   guarantee in the new population. If no upper cut-off can reach 95%, the highest scores are
   mis-ranked, which no threshold can repair. The script then falls back to asymmetric
   abstention: it recalibrates the deterministic 'no mobility' decision alone and sends every
   higher score to the weighted coin.

What to expect. Discrimination should transfer reasonably well, because the relationship
between education, occupation, age, hours and income is structurally stable. Calibration
should not: the share of people above the threshold changed between 1994 nationally and 2018
in California (prior shift), and occupations and household patterns changed too (covariate
shift). A drop in calibration, and therefore in the precision of the deterministic decisions,
is the expected signature of distribution shift, not a defect of the pipeline.

Requirements: `pip install folktables` in the training-pipeline virtual environment.

Results are written to ood_results_<STATE>.json (e.g. ood_results_CA.json).

Run from the ai-service folder:
    training-pipeline/.venv/Scripts/python.exe 05_ood_testing.py --state CA
    training-pipeline/.venv/Scripts/python.exe 05_ood_testing.py --state NY --sample 50000
    training-pipeline/.venv/Scripts/python.exe 05_ood_testing.py --input-csv my_acs_extract.csv
"""

import argparse
import importlib.util
import json
import math
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

AI_SERVICE_DIR = Path(__file__).resolve().parent
REPO_DIR = AI_SERVICE_DIR.parent
ARTIFACTS_DIR = AI_SERVICE_DIR / "training-pipeline" / "artifacts"
CPI_SOURCE = REPO_DIR / "bff-api" / "src" / "core" / "domain" / "services" / "consumer-price-index.ts"
CACHE_DIR = AI_SERVICE_DIR / ".folktables-cache"
EVALUATOR_FILE = AI_SERVICE_DIR / "training-pipeline" / "src" / "04_evaluator.py"
ADAPTATION_SHARE = 0.5

MODEL_BASE_YEAR = 1994
INCOME_THRESHOLD_1994 = 50_000
RANDOM_STATE = 42
CALIBRATION_BINS = 10
WILSON_Z = 1.959964
# Training-domain limits of the Character entity (inference-api/src/domain/entities/character.py).
AGE_RANGE = (17, 90)
HOURS_RANGE = (1, 99)
# A category the 1994 vocabulary never contained: encoded as all zeros, exactly like the API.
UNSEEN = "Other"


class OODError(Exception):
    """Raised when inputs are missing or inconsistent."""


def log(message: str) -> None:
    """Writes a progress message to standard output, prefixed with the stage name.

    Args:
        message: Human-readable progress text.
    """
    print(f"[05_ood_testing] {message}", flush=True)


# ------------------------------------------------------------------ temporal normalisation

def load_cpi_table(source: Path = CPI_SOURCE) -> tuple[dict[int, float], float]:
    """Reads the CPI-U table and the projection rate from the BFF module, the single source of truth.

    Args:
        source: Path of consumer-price-index.ts.

    Returns:
        A tuple (table, projection_rate): the annual CPI-U averages by year and the annual
        inflation assumed after the last published year.

    Raises:
        OODError: If the file or its constants cannot be found.
    """
    if not source.is_file():
        raise OODError(f"CPI source not found: {source}")
    text = source.read_text(encoding="utf-8")
    block = re.search(r"CPI_U_ANNUAL_AVERAGE[^{]*\{([^}]*)\}", text)
    rate = re.search(r"PROJECTION_RATE\s*=\s*([0-9.]+)", text)
    if block is None or rate is None:
        raise OODError(f"Could not parse the CPI table or the projection rate in {source.name}")
    table = {int(year): float(value) for year, value in re.findall(r"(\d{4})\s*:\s*([0-9.]+)", block.group(1))}
    return table, float(rate.group(1))


def consumer_price_index(year: int, table: dict[int, float], projection_rate: float) -> float:
    """Returns the CPI-U of a year, mirroring consumerPriceIndex() in the BFF.

    CPI(t) is the published value for t <= last published year, and
    CPI(last) * (1 + rate)^(t - last) afterwards.

    Args:
        year: Calendar year.
        table: Published annual averages.
        projection_rate: Annual inflation assumed after the last published year.

    Returns:
        The CPI-U value.
    """
    if year in table:
        return table[year]
    last = max(table)
    return table[last] * (1 + projection_rate) ** (year - last)


def deflation_factor(year: int) -> float:
    """Factor that converts dollars of `year` into 1994 dollars: CPI(1994) / CPI(year).

    Args:
        year: The survey year.

    Returns:
        The multiplicative deflation factor.
    """
    table, rate = load_cpi_table()
    return consumer_price_index(MODEL_BASE_YEAR, table, rate) / consumer_price_index(year, table, rate)


# ------------------------------------------------------------------ data

def download_acs(state: str, year: int) -> pd.DataFrame:
    """Downloads (or reads from the local cache) the ACS 1-year person records of one state.

    Args:
        state: Two-letter state code, e.g. 'CA'.
        year: Survey year, e.g. 2018.

    Returns:
        The raw person records, with every PUMS column.

    Raises:
        OODError: If folktables is not installed.
    """
    try:
        from folktables import ACSDataSource
    except ImportError as exc:
        raise OODError("folktables is not installed: run `pip install folktables` in the training-pipeline venv") from exc
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Loading ACS {year} 1-Year person records for {state} (cache: {CACHE_DIR.name}/)")
    source = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person", root_dir=str(CACHE_DIR))
    return source.get_data(states=[state], download=True)


def child_households(raw: pd.DataFrame) -> set:
    """Identifies the households whose reference person lives with an own child.

    Computed BEFORE the ACSIncome filter, which removes the children (age <= 16) themselves.
    It is needed to tell apart the two 1994 categories for an unmarried householder:
    'Unmarried' (living with own children) and 'Not-in-family'.

    Args:
        raw: Unfiltered person records with SERIALNO and RELP (2018) or RELSHIPP (2019+).

    Returns:
        The set of household serial numbers with an own child of the reference person.
    """
    if "SERIALNO" not in raw:
        return set()
    if "RELP" in raw:
        children = raw["RELP"].isin([2, 3, 4])
    elif "RELSHIPP" in raw:
        children = raw["RELSHIPP"].isin([25, 26, 27])
    else:
        return set()
    return set(raw.loc[children, "SERIALNO"])


def acsincome_filter(raw: pd.DataFrame) -> pd.DataFrame:
    """Applies the ACSIncome population filter of folktables.

    Keeps people older than 16 who worked (hours > 0), with a personal income above 100 and a
    person weight of at least 1.

    Args:
        raw: Raw person records.

    Returns:
        The filtered records.
    """
    keep = (raw["AGEP"] > 16) & (raw["PINCP"] > 100) & (raw["WKHP"] > 0) & (raw["PWGTP"] >= 1)
    return raw.loc[keep].copy()


# ------------------------------------------------------------------ feature mapping (ACS -> 1994 census)

WORKCLASS = {1: "Private", 2: "Private", 3: "Local-gov", 4: "State-gov", 5: "Federal-gov",
             6: "Self-emp-not-inc", 7: "Self-emp-inc", 8: "Without-pay", 9: "Never-worked"}
"""COW (class of worker). Private non-profit employees (2) were 'Private' in 1994, which had no separate category."""

EDUCATION = {
    1: ("Preschool", 1), 2: ("Preschool", 1), 3: ("Preschool", 1),
    4: ("1st-4th", 2), 5: ("1st-4th", 2), 6: ("1st-4th", 2), 7: ("1st-4th", 2),
    8: ("5th-6th", 3), 9: ("5th-6th", 3), 10: ("7th-8th", 4), 11: ("7th-8th", 4),
    12: ("9th", 5), 13: ("10th", 6), 14: ("11th", 7), 15: ("12th", 8),
    16: ("HS-grad", 9), 17: ("HS-grad", 9), 18: ("Some-college", 10), 19: ("Some-college", 10),
    20: ("Assoc-voc", 11), 21: ("Bachelors", 13), 22: ("Masters", 14), 23: ("Prof-school", 15), 24: ("Doctorate", 16),
}
"""SCHL (educational attainment) -> (education, education_num). ACS does not separate vocational from
academic associate degrees; code 20 is mapped to 'Assoc-voc', the more frequent of the two in 1994."""

MARITAL = {1: "Married-civ-spouse", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never-married"}
"""MAR. ACS does not record whether the spouse is present, so every married person is 'Married-civ-spouse'."""

RACE = {1: "White", 2: "Black", 3: "Amer-Indian-Eskimo", 4: "Amer-Indian-Eskimo", 5: "Amer-Indian-Eskimo",
        6: "Asian-Pac-Islander", 7: "Asian-Pac-Islander", 8: "Other", 9: "Other"}
"""RAC1P. 'Two or more races' (9) did not exist in 1994 and is mapped to 'Other'."""

COUNTRY = {
    72: "Puerto-Rico", 60: "Outlying-US(Guam-USVI-etc)", 66: "Outlying-US(Guam-USVI-etc)",
    69: "Outlying-US(Guam-USVI-etc)", 78: "Outlying-US(Guam-USVI-etc)",
    109: "France", 110: "Germany", 116: "Greece", 117: "Hungary", 119: "Ireland", 120: "Italy",
    128: "Poland", 129: "Portugal", 139: "England", 140: "Scotland", 147: "Yugoslavia",
    206: "Cambodia", 207: "China", 209: "Hong", 210: "India", 212: "Iran", 215: "Japan", 217: "South",
    223: "Laos", 233: "Philippines", 240: "Taiwan", 242: "Thailand", 247: "Vietnam",
    301: "Canada", 303: "Mexico", 312: "El-Salvador", 313: "Guatemala", 314: "Honduras", 315: "Nicaragua",
    327: "Cuba", 329: "Dominican-Republic", 332: "Haiti", 333: "Jamaica", 341: "Trinadad&Tobago",
    364: "Columbia", 365: "Ecuador", 370: "Peru",
}
"""POBP (place of birth) -> 1994 native_country, for the countries in the 1994 vocabulary (codes of
the 2018 PUMS data dictionary). US states (1-56) map to 'United-States'. Korea is mapped to 'South',
the label the Adult data uses for it. Any other country is unseen in 1994 and is encoded as zeros."""

OCCUPATION_RANGES = [
    (10, 960, "Exec-managerial"),       # management; business and financial operations
    (1005, 1530, "Prof-specialty"),     # computer, mathematical, architecture and engineering
    (1540, 1560, "Tech-support"),       # drafters, engineering and mapping technicians
    (1600, 1860, "Prof-specialty"),     # life, physical and social scientists
    (1900, 1980, "Tech-support"),       # science technicians
    (2001, 2970, "Prof-specialty"),     # community service, legal, education, arts, media
    (3000, 3270, "Prof-specialty"),     # health diagnosing and treating practitioners
    (3300, 3550, "Tech-support"),       # health technologists and technicians
    (3601, 3655, "Other-service"),      # healthcare support
    (3700, 3960, "Protective-serv"),    # protective service
    (4000, 4160, "Other-service"),      # food preparation and serving
    (4200, 4240, "Other-service"),      # building cleaning and pest control
    (4251, 4255, "Farming-fishing"),    # grounds maintenance (under farming in 1990 census codes)
    (4330, 4655, "Other-service"),      # personal care and service
    (4700, 4965, "Sales"),              # sales and related
    (5000, 5940, "Adm-clerical"),       # office and administrative support
    (6005, 6130, "Farming-fishing"),    # farming, fishing and forestry
    (6200, 6250, "Craft-repair"),       # construction trades (supervisors to carpenters)
    (6260, 6260, "Handlers-cleaners"),  # construction laborers
    (6305, 6590, "Craft-repair"),       # construction trades
    (6600, 6600, "Handlers-cleaners"),  # helpers, construction trades
    (6660, 6950, "Craft-repair"),       # other construction and extraction
    (7000, 7640, "Craft-repair"),       # installation, maintenance and repair
    (7700, 8990, "Machine-op-inspct"),  # production
    (9005, 9600, "Transport-moving"),   # transportation and vehicle operators
    (9610, 9640, "Handlers-cleaners"),  # vehicle cleaners, laborers and freight movers, packers
    (9645, 9760, "Transport-moving"),   # other material moving
    (9800, 9830, "Armed-Forces"),       # military
    (9920, 9920, "Unknown"),            # unemployed, last worked 5+ years ago or never worked
]
"""OCCP (2018 census occupation codes) -> the 14 occupation groups of the 1994 census, by code range."""


def map_occupation(codes: pd.Series) -> pd.Series:
    """Maps OCCP codes to 1994 occupation groups by code range.

    Args:
        codes: OCCP values (numeric or numeric strings).

    Returns:
        The 1994 occupation labels; codes outside every range become 'Unknown'.
    """
    numeric = pd.to_numeric(codes, errors="coerce")
    labels = pd.Series("Unknown", index=codes.index, dtype=object)
    for low, high, label in OCCUPATION_RANGES:
        labels[(numeric >= low) & (numeric <= high)] = label
    return labels


def map_relationship(people: pd.DataFrame, with_children: set) -> pd.Series:
    """Maps the household relationship to the 1994 categories.

    The 1994 categories are: Husband / Wife (the spouse couple, by sex), Own-child,
    Other-relative, Unmarried (an unmarried partner, or an unmarried householder living with own
    children) and Not-in-family (people living alone or with non-relatives, and group quarters).

    Args:
        people: Filtered records with SEX, MAR, SERIALNO and RELP (2018) or RELSHIPP (2019+).
        with_children: Households whose reference person lives with an own child.

    Returns:
        The 1994 relationship labels.
    """
    spouse_label = np.where(people["SEX"] == 1, "Husband", "Wife")
    married = people["MAR"] == 1
    has_child = people["SERIALNO"].isin(with_children) if "SERIALNO" in people else pd.Series(False, index=people.index)
    if "RELP" in people:
        code = people["RELP"]
        reference, spouse, partner = code == 0, code == 1, code == 13
        child, relative = code.isin([2, 3, 4]), code.isin([5, 6, 7, 8, 9, 10])
    else:  # 2019+ files use RELSHIPP
        code = people["RELSHIPP"]
        reference, spouse, partner = code == 20, code.isin([21, 23]), code.isin([22, 24])
        child, relative = code.isin([25, 26, 27]), code.isin([28, 29, 30, 31, 32, 33])
    labels = pd.Series("Not-in-family", index=people.index, dtype=object)
    labels[relative] = "Other-relative"
    labels[child] = "Own-child"
    labels[partner] = "Unmarried"
    labels[reference & ~married & has_child] = "Unmarried"
    couple = spouse | (reference & married)
    labels[couple] = spouse_label[couple.to_numpy()]
    return labels


def map_country(codes: pd.Series) -> pd.Series:
    """Maps POBP birthplace codes to the 1994 native_country labels.

    Args:
        codes: POBP values.

    Returns:
        The labels: 'United-States' for the 50 states and DC, the 1994 spelling for countries
        in the 1994 vocabulary, and UNSEEN for any other country.
    """
    numeric = pd.to_numeric(codes, errors="coerce")
    labels = numeric.map(COUNTRY)
    labels[(numeric >= 1) & (numeric <= 56)] = "United-States"
    return labels.fillna(UNSEEN).astype(object)


def build_features(people: pd.DataFrame, with_children: set, deflator: float) -> pd.DataFrame:
    """Recodes the filtered ACS records into the 13 census traits plus the deflated target and the weight.

    Args:
        people: Records after the ACSIncome filter.
        with_children: Output of child_households() on the unfiltered records.
        deflator: CPI(1994) / CPI(survey year).

    Returns:
        A frame with the 1994 trait columns, `income_1994`, the binary target `income_above_50k`
        and the survey weight `weight`.
    """
    adjinc = people["ADJINC"] / 1_000_000 if "ADJINC" in people else 1.0  # survey dollars -> survey-year dollars
    income_1994 = people["PINCP"] * adjinc * deflator
    education = people["SCHL"].map(EDUCATION)
    return pd.DataFrame({
        "age": people["AGEP"].clip(*AGE_RANGE).astype(float),
        "workclass": people["COW"].map(WORKCLASS).fillna("Unknown"),
        "education": education.map(lambda e: e[0] if isinstance(e, tuple) else "Unknown"),
        "education_num": education.map(lambda e: e[1] if isinstance(e, tuple) else np.nan).astype(float),
        "marital_status": people["MAR"].map(MARITAL).fillna("Unknown"),
        "occupation": map_occupation(people["OCCP"]),
        "relationship": map_relationship(people, with_children),
        "race": people["RAC1P"].map(RACE).fillna("Other"),
        "sex": people["SEX"].map({1: "Male", 2: "Female"}),
        "hours_per_week": people["WKHP"].clip(*HOURS_RANGE).astype(float),
        "native_country": map_country(people["POBP"]),
        # ACS records no realised capital gains or losses; 0 is the value of >90% of the Adult records.
        "capital_gain": 0.0,
        "capital_loss": 0.0,
        "income_1994": income_1994,
        "income_above_50k": (income_1994 > INCOME_THRESHOLD_1994).astype(int),
        "weight": people["PWGTP"].astype(float),
    }, index=people.index)


# ------------------------------------------------------------------ inference (no retraining)

def load_artifacts() -> dict:
    """Loads the deployed model, its preprocessing artifacts and the frozen decision policy.

    Returns:
        A mapping with the model, imputer, encoder, schema, report (traceability_report.json)
        and model name.

    Raises:
        OODError: If an artifact is missing.
    """
    names = ["best_model.joblib", "imputer.joblib", "encoder.joblib", "feature_schema.json", "traceability_report.json"]
    for name in names:
        if not (ARTIFACTS_DIR / name).is_file():
            raise OODError(f"Missing artifact {name}: run the training pipeline first")
    report = json.loads((ARTIFACTS_DIR / "traceability_report.json").read_text(encoding="utf-8"))
    return {
        "model": joblib.load(ARTIFACTS_DIR / "best_model.joblib"),
        "imputer": joblib.load(ARTIFACTS_DIR / "imputer.joblib"),
        "encoder": joblib.load(ARTIFACTS_DIR / "encoder.joblib"),
        "schema": json.loads((ARTIFACTS_DIR / "feature_schema.json").read_text(encoding="utf-8")),
        "report": report,
        "model_name": report["model"]["name"],
    }


def transform(features: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Applies the exact training transform, as the inference API does (sklearn_predictor.py).

    Args:
        features: Output of build_features().
        artifacts: Output of load_artifacts().

    Returns:
        The numeric model matrix, with the columns of feature_schema.json in order.
    """
    schema = artifacts["schema"]
    numeric = pd.DataFrame(artifacts["imputer"].transform(features[schema["numeric_columns"]]),
                           columns=schema["numeric_columns"], index=features.index)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
        encoded_values = artifacts["encoder"].transform(features[schema["categorical_columns"]].astype(object))
    encoded = pd.DataFrame(encoded_values, columns=artifacts["encoder"].get_feature_names_out(schema["categorical_columns"]),
                           index=features.index)
    return pd.concat([numeric, encoded], axis=1)[schema["feature_columns"]].astype("float64")


def unseen_category_shares(features: pd.DataFrame, artifacts: dict) -> dict:
    """Measures, per categorical trait, the share of rows whose value the 1994 vocabulary never saw.

    These rows are encoded as all zeros (the API does the same), so a high share flags a weak
    mapping or a genuine novelty of the 2018 population (e.g. countries of birth absent in 1994).

    Args:
        features: Output of build_features().
        artifacts: Output of load_artifacts().

    Returns:
        A mapping from trait to {share, rows, top_values}.
    """
    shares = {}
    for column, categories in zip(artifacts["schema"]["categorical_columns"], artifacts["encoder"].categories_):
        unseen = ~features[column].isin(set(categories))
        shares[column] = {
            "share": float(unseen.mean()),
            "rows": int(unseen.sum()),
            "top_values": {str(k): int(v) for k, v in features.loc[unseen, column].value_counts().head(3).items()},
        }
    return shares


# ------------------------------------------------------------------ metrics

def wilson(successes: int, trials: int) -> list[float]:
    """Computes the Wilson score 95% interval of a binomial proportion (as in 04_evaluator.py).

    Args:
        successes: Number of successes.
        trials: Number of trials.

    Returns:
        The interval [lower, upper], or [0.0, 0.0] without trials.
    """
    if trials == 0:
        return [0.0, 0.0]
    p = successes / trials
    denominator = 1 + WILSON_Z**2 / trials
    centre = (p + WILSON_Z**2 / (2 * trials)) / denominator
    margin = WILSON_Z * math.sqrt(p * (1 - p) / trials + WILSON_Z**2 / (4 * trials**2)) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = CALIBRATION_BINS) -> float:
    """Computes the expected calibration error over equal-width bins (as in 03_trainer.py).

    Args:
        y: Binary labels.
        p: Predicted probabilities.
        bins: Number of bins.

    Returns:
        The ECE, in [0, 1].
    """
    index = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum((index == b).mean() * abs(p[index == b].mean() - y[index == b].mean())
                     for b in range(bins) if (index == b).any()))


def probabilistic_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict:
    """Computes discrimination and calibration metrics, unweighted and survey-weighted.

    Args:
        y: Binary labels.
        p: Predicted probabilities.
        w: Survey weights (PWGTP).

    Returns:
        ROC-AUC, average precision, Brier, log-loss, ECE, the mean prediction and the observed
        positive rate, plus the weighted versions of the population-level metrics.
    """
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))),
        "ece": expected_calibration_error(y, p),
        "mean_predicted_probability": float(p.mean()),
        "observed_positive_rate": float(y.mean()),
        "weighted": {
            "roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
            "average_precision": float(average_precision_score(y, p, sample_weight=w)),
            "brier": float(brier_score_loss(y, p, sample_weight=w)),
            "mean_predicted_probability": float(np.average(p, weights=w)),
            "observed_positive_rate": float(np.average(y, weights=w)),
        },
    }


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Evaluates the single-threshold decision 1(p >= t).

    Args:
        y: Binary labels.
        p: Predicted probabilities.
        threshold: Cut-off t.

    Returns:
        Accuracy, precision (with its Wilson interval), recall and the confusion counts.
    """
    flagged = p >= threshold
    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    fn, tn = int((~flagged & (y == 1)).sum()), int((~flagged & (y == 0)).sum())
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(y),
        "precision": tp / max(tp + fp, 1),
        "precision_95ci": wilson(tp, tp + fp),
        "recall": tp / max(tp + fn, 1),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def selective_metrics(y: np.ndarray, p: np.ndarray, lower: float, upper: float) -> dict:
    """Evaluates the selective-classification policy frozen on the 1994 validation set.

    Args:
        y: Binary labels.
        p: Predicted probabilities.
        lower: Lower bound of the uncertainty band.
        upper: Upper bound of the uncertainty band.

    Returns:
        Coverage, the precision of each deterministic decision with Wilson intervals, accuracy on
        decided cases, in-band calibration, the expected in-game accuracy and the decision matrix.
    """
    positive, negative = p >= upper, p <= lower
    covered, uncertain = positive | negative, ~(positive | negative)
    tp, n_pos = int((positive & (y == 1)).sum()), int(positive.sum())
    tn, n_neg = int((negative & (y == 0)).sum()), int(negative.sum())
    correct = tp + tn
    in_band = np.where(y[uncertain] == 1, p[uncertain], 1 - p[uncertain]).sum()
    return {
        "coverage": float(covered.mean()),
        "precision_mobility": tp / max(n_pos, 1),
        "precision_mobility_95ci": wilson(tp, n_pos),
        "precision_no_mobility": tn / max(n_neg, 1),
        "precision_no_mobility_95ci": wilson(tn, n_neg),
        "accuracy_on_covered": correct / max(int(covered.sum()), 1),
        "accuracy_on_covered_95ci": wilson(correct, int(covered.sum())),
        "positive_rate_inside_band": float(y[uncertain].mean()) if uncertain.any() else None,
        "mean_probability_inside_band": float(p[uncertain].mean()) if uncertain.any() else None,
        "expected_game_accuracy": float((correct + in_band) / len(y)),
        "decision_matrix": {
            "actual_no_mobility": {"predicted_no": tn, "luck": int((uncertain & (y == 0)).sum()), "predicted_yes": n_pos - tp},
            "actual_mobility": {"predicted_no": n_neg - tn, "luck": int((uncertain & (y == 1)).sum()), "predicted_yes": tp},
        },
    }


def load_evaluator():
    """Imports training-pipeline/src/04_evaluator.py, so that the band rule is the one used in 1994.

    Returns:
        The evaluator module (its file name starts with a digit, hence importlib).
    """
    spec = importlib.util.spec_from_file_location("evaluator_04", EVALUATOR_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapt_band(y: np.ndarray, p: np.ndarray, frozen: tuple[float, float]) -> dict:
    """Re-estimates the uncertainty band on half of the new population and evaluates it on the other half.

    The model is not retrained: only the two cut-offs (l, u) are searched again, with the rule of
    04_evaluator.choose_band. The frozen 1994 band and the adapted band are evaluated on the same
    held-out half, which takes no part in the search.

    Args:
        y: Binary labels of the whole OOD sample.
        p: Model probabilities of the whole OOD sample.
        frozen: The (lower, upper) band chosen on the 1994 validation set.

    Returns:
        The split sizes, the adapted band (or the reason none exists), the best precision the
        grid can reach on each side, and the held-out metrics of both bands.
    """
    evaluator = load_evaluator()
    indices = np.arange(len(y))
    fit_idx, holdout_idx = train_test_split(indices, test_size=1 - ADAPTATION_SHARE, stratify=y, random_state=RANDOM_STATE)
    y_fit, p_fit, y_out, p_out = y[fit_idx], p[fit_idx], y[holdout_idx], p[holdout_idx]

    # Best precision the grid can reach on each side, so that an impossible target is reported, not hidden.
    reach = {
        "best_precision_mobility": max((float(y_fit[p_fit >= u].mean()), float(u)) for u in evaluator.UPPER_GRID
                                       if (p_fit >= u).sum() >= evaluator.MIN_DECISIONS),
        "best_precision_no_mobility": max((float(1 - y_fit[p_fit <= l].mean()), float(l)) for l in evaluator.LOWER_GRID
                                          if (p_fit <= l).sum() >= evaluator.MIN_DECISIONS),
    }
    result = {
        "rule": "04_evaluator.choose_band: maximise coverage s.t. precision >= 95% for both deterministic decisions, "
                f"each with >= {evaluator.MIN_DECISIONS} cases (grid step 0.01)",
        "split": {"adaptation_rows": int(len(fit_idx)), "held_out_rows": int(len(holdout_idx)), "stratified": True,
                  "random_state": RANDOM_STATE},
        "grid_reach_on_adaptation_half": {k: {"precision": v[0], "at_cutoff": v[1]} for k, v in reach.items()},
        "held_out_frozen_1994_band": {"band": list(frozen), **selective_metrics(y_out, p_out, *frozen)},
    }
    try:
        lower, upper, evaluated = evaluator.choose_band(p_fit, y_fit)
    except evaluator.EvaluationError as exc:
        result["adapted_band"] = None
        result["adaptation_status"] = f"TWO_SIDED_TARGET_NOT_REACHABLE: {exc}"
        result["one_sided_adaptation"] = adapt_one_sided(y_fit, p_fit, y_out, p_out, evaluator)
        return result
    result["adapted_band"] = [lower, upper]
    result["candidate_bands_evaluated"] = int(evaluated)
    result["adaptation_status"] = "TARGET_MET_ON_ADAPTATION_HALF"
    result["held_out_adapted_band"] = {"band": [lower, upper], **selective_metrics(y_out, p_out, lower, upper)}
    return result


def adapt_one_sided(y_fit: np.ndarray, p_fit: np.ndarray, y_out: np.ndarray, p_out: np.ndarray, evaluator) -> dict:
    """Asymmetric abstention: recalibrate only the side whose precision target is still reachable.

    When no upper cut-off reaches the target, the problem lies in the ranking of the highest
    scores, which a threshold cannot repair: only retraining could. The policy keeps the
    deterministic 'no mobility' decision, re-estimated as the widest lower cut-off l with negative
    predictive value >= 95% (at least MIN_DECISIONS cases), and abstains on everything above l.
    Above l, the game resolves the outcome by the weighted coin.

    Args:
        y_fit: Labels of the adaptation half.
        p_fit: Probabilities of the adaptation half.
        y_out: Labels of the held-out half.
        p_out: Probabilities of the held-out half.
        evaluator: The 04_evaluator module (for its grid and constants).

    Returns:
        The policy, its lower cut-off, and its held-out metrics; or the reason it does not exist.
    """
    admissible = [float(l) for l in evaluator.LOWER_GRID
                  if (p_fit <= l).sum() >= evaluator.MIN_DECISIONS and (1 - y_fit[p_fit <= l].mean()) >= evaluator.TARGET_PRECISION]
    if not admissible:
        return {"status": "NOT_REACHABLE", "reason": "no lower cut-off reaches 95% negative predictive value"}
    lower = max(admissible)  # coverage grows with l, so the widest admissible cut-off maximises it
    held = selective_metrics(y_out, p_out, lower, math.inf)
    held["precision_mobility"] = held["precision_mobility_95ci"] = None  # the policy never decides "mobility"
    return {
        "status": "ADAPTED",
        "policy": "deterministic 'no mobility' for p <= lower; weighted coin for every p > lower (no deterministic 'mobility')",
        "lower_cutoff": lower,
        "held_out": {"band": [lower, None], **held},
    }


def in_distribution_reference(report: dict) -> dict:
    """Collects the 1994 test-set results from traceability_report.json, for comparison.

    Args:
        report: The parsed traceability report.

    Returns:
        The in-distribution figures, with the same keys as the OOD results.
    """
    test = report["selective_classification"]["test"]
    return {
        "roc_auc": report["calibration"]["test"]["roc_auc"],
        "brier": report["calibration"]["test"]["brier"],
        "observed_positive_rate": report["data"]["test_positive_rate"],
        "coverage": test["coverage"],
        "precision_mobility": test["precision_mobility"],
        "precision_no_mobility": test["precision_no_mobility"],
        "accuracy_on_covered": test["accuracy_on_covered"],
        "expected_game_accuracy": report["game_mechanic"]["test"]["expected_accuracy_all_players"],
    }


# ------------------------------------------------------------------ main

def parse_args() -> argparse.Namespace:
    """Parses the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Out-of-distribution test of the deployed model on ACS Income.")
    parser.add_argument("--state", default="CA", help="two-letter state code (default: CA)")
    parser.add_argument("--year", type=int, default=2018, help="ACS survey year (default: 2018)")
    parser.add_argument("--sample", type=int, default=None, help="score a random sample of N people (seed 42)")
    parser.add_argument("--input-csv", type=Path, default=None, help="read raw ACS person records from a CSV instead of downloading")
    parser.add_argument("--output", type=Path, default=None, help="results file (default: ood_results_<STATE>.json)")
    return parser.parse_args()


def main() -> int:
    """Runs the OOD test end to end and writes the results JSON.

    Returns:
        The process exit code: 0 on success, 1 on an OODError.
    """
    args = parse_args()
    if args.output is None:
        args.output = AI_SERVICE_DIR / f"ood_results_{args.state.upper()}.json"
    try:
        raw = pd.read_csv(args.input_csv) if args.input_csv else download_acs(args.state, args.year)
        with_children = child_households(raw)
        people = acsincome_filter(raw)
        if args.sample and args.sample < len(people):
            people = people.sample(args.sample, random_state=RANDOM_STATE)
        deflator = deflation_factor(args.year)
        log(f"{len(raw):,} raw records -> {len(people):,} after the ACSIncome filter; "
            f"CPI deflator {args.year}->1994 = {deflator:.4f} (threshold = USD {INCOME_THRESHOLD_1994 / deflator:,.0f} of {args.year})")

        features = build_features(people, with_children, deflator)
        artifacts = load_artifacts()
        p = artifacts["model"].predict_proba(transform(features, artifacts))[:, 1]
        y, w = features["income_above_50k"].to_numpy(), features["weight"].to_numpy()
        policy = artifacts["report"]["decision_policy"]

        results = {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": {
                "source": "folktables ACSDataSource (ACS PUMS, 1-Year, person)" if not args.input_csv else str(args.input_csv),
                "state": args.state, "survey_year": args.year,
                "raw_records": int(len(raw)), "evaluated_records": int(len(people)),
                "filter": "ACSIncome: AGEP > 16, PINCP > 100, WKHP > 0, PWGTP >= 1",
            },
            "temporal_normalisation": {
                "cpi_source": str(CPI_SOURCE.relative_to(REPO_DIR)),
                "deflator_to_1994": deflator,
                "adjinc_applied": "ADJINC" in people,
                "target": f"PINCP * ADJINC * deflator > {INCOME_THRESHOLD_1994} (1994 dollars)",
                "threshold_in_survey_year_dollars": INCOME_THRESHOLD_1994 / deflator,
                "capital_gain_and_loss": "set to 0: ACS records no realised capital gains or losses",
            },
            "model": {"name": artifacts["model_name"], "file": "training-pipeline/artifacts/best_model.joblib", "retrained": False},
            "mapping_diagnostics": {"unseen_category_shares": unseen_category_shares(features, artifacts)},
            "ood": {
                "probabilistic": probabilistic_metrics(y, p, w),
                "reference_at_0_5": threshold_metrics(y, p, 0.5),
                "deterministic_threshold": threshold_metrics(y, p, policy["deterministic_threshold"]),
                "selective_classification": {"band": [policy["lower_bound"], policy["upper_bound"]],
                                             **selective_metrics(y, p, policy["lower_bound"], policy["upper_bound"])},
            },
            "in_distribution_1994_test": in_distribution_reference(artifacts["report"]),
        }
        ood, ref = results["ood"], results["in_distribution_1994_test"]
        sel = ood["selective_classification"]
        results["shift_summary"] = {
            "delta_roc_auc": ood["probabilistic"]["roc_auc"] - ref["roc_auc"],
            "delta_brier": ood["probabilistic"]["brier"] - ref["brier"],
            "prior_shift": ood["probabilistic"]["observed_positive_rate"] - ref["observed_positive_rate"],
            "calibration_in_the_large": ood["probabilistic"]["mean_predicted_probability"] - ood["probabilistic"]["observed_positive_rate"],
            "delta_coverage": sel["coverage"] - ref["coverage"],
            "delta_precision_mobility": sel["precision_mobility"] - ref["precision_mobility"],
            "delta_precision_no_mobility": sel["precision_no_mobility"] - ref["precision_no_mobility"],
            "both_deterministic_precisions_at_least_95": sel["precision_mobility"] >= 0.95 and sel["precision_no_mobility"] >= 0.95,
        }
        results["adaptation"] = adapt_band(y, p, (policy["lower_bound"], policy["upper_bound"]))

        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        prob = ood["probabilistic"]
        log(f"ROC-AUC {prob['roc_auc']:.4f} (1994 test {ref['roc_auc']:.4f}) | Brier {prob['brier']:.4f} | ECE {prob['ece']:.4f}")
        log(f"Positive rate {prob['observed_positive_rate']:.2%} (1994 test {ref['observed_positive_rate']:.2%}) | "
            f"mean prediction {prob['mean_predicted_probability']:.2%}")
        log(f"Band [{policy['lower_bound']:.2f}, {policy['upper_bound']:.2f}]: coverage {sel['coverage']:.2%} | "
            f"precision 'mobility' {sel['precision_mobility']:.2%} | 'no mobility' {sel['precision_no_mobility']:.2%} | "
            f"game accuracy {sel['expected_game_accuracy']:.2%}")
        adaptation = results["adaptation"]
        for label, key in (("frozen 1994 band ", "held_out_frozen_1994_band"), ("adapted 2018 band", "held_out_adapted_band")):
            if key in adaptation:
                held = adaptation[key]
                log(f"Held-out half, {label} [{held['band'][0]:.2f}, {held['band'][1]:.2f}]: coverage {held['coverage']:.2%} | "
                    f"precision 'mobility' {held['precision_mobility']:.2%} | 'no mobility' {held['precision_no_mobility']:.2%}")
        if adaptation["adapted_band"] is None:
            log(f"Adaptation: {adaptation['adaptation_status'].split(':')[0]} (best 'mobility' precision on the grid: "
                f"{adaptation['grid_reach_on_adaptation_half']['best_precision_mobility']['precision']:.2%})")
            one_sided = adaptation["one_sided_adaptation"]
            if one_sided["status"] == "ADAPTED":
                held = one_sided["held_out"]
                log(f"Held-out half, one-sided band p <= {one_sided['lower_cutoff']:.2f}: coverage {held['coverage']:.2%} | "
                    f"precision 'no mobility' {held['precision_no_mobility']:.2%} "
                    f"(95% CI {held['precision_no_mobility_95ci'][0]:.2%}-{held['precision_no_mobility_95ci'][1]:.2%}) | "
                    f"game accuracy {held['expected_game_accuracy']:.2%}")
        log(f"Saved {args.output}")
    except OODError as exc:
        log(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
