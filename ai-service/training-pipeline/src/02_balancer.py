"""Step 02 of the Unfair Life training pipeline: split, encode and balance.

This module produces leakage-free training, validation and test sets, the
preprocessing artifacts with which the inference API transforms a character
profile, and a class-balanced copy of the training set for the SMOTE ablation.

1. Stratified 60 / 20 / 20 split of the clean Adult data, made BEFORE
   anything is fitted:

       train      - fits the models
       validation - selects the model, the decision threshold and the
                    selective-classification band (real rows, never balanced)
       test       - touched once, by step 04, for the final report

   Stratification on the target makes each partition reproduce the
   population prior pi = P(Y = 1) of about 0.239. This reduces the variance
   of every estimate that depends on the class mix.

2. Leakage control. The median imputer and the one-hot encoder are fitted on
   the training rows only. Fitting them on all rows would let statistics of
   the validation and test sets, such as their medians and category
   vocabularies, inform the training features. This is data leakage, and it
   biases the evaluation optimistically.

3. SMOTENC balances a COPY of the training set to 50/50. Both versions are
   saved, so that step 03 can train every model with and without SMOTE and
   document the comparison. (Result in Phase 1: SMOTE skewed the predicted
   probabilities and lowered validation average precision, so the deployed
   model is trained on the natural distribution.) The theoretical reason is
   that a classifier trained under a class prior pi' estimates
   P'(Y = 1 | x). By Bayes' theorem, this probability differs from the
   population probability P(Y = 1 | x) through the prior-ratio factor
   (pi' / (1 - pi')) / (pi / (1 - pi)) in odds space. Forcing pi' = 0.5 when
   pi is about 0.239 multiplies the odds by roughly 3.2 and inflates every
   probability.

fnlwgt is a census sampling weight, not a personal trait, so it is not a feature.

Run from the training-pipeline folder:
    .venv/Scripts/python.exe src/02_balancer.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PIPELINE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PIPELINE_DIR / "data" / "processed"
ARTIFACTS_DIR = PIPELINE_DIR / "artifacts"
INPUT_FILE = PROCESSED_DIR / "clean_adult.csv"
RAW_SPLITS_FILE = PROCESSED_DIR / "raw_splits.parquet"
BALANCING_REPORT_FILE = PROCESSED_DIR / "balancing_report.json"

IMPUTER_FILE = ARTIFACTS_DIR / "imputer.joblib"
ENCODER_FILE = ARTIFACTS_DIR / "encoder.joblib"
FEATURE_SCHEMA_FILE = ARTIFACTS_DIR / "feature_schema.json"
OBSOLETE_ARTIFACTS = [ARTIFACTS_DIR / "country_indicators.joblib"]

TARGET_COLUMN = "income_above_50k"

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
NUMERIC_COLUMNS = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
PERSON_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

VALIDATION_SHARE = 0.20
TEST_SHARE = 0.20
# Minimum dataset size required by the project specification; verified before any processing.
MIN_REQUIRED_RECORDS = 35_000
RANDOM_STATE = 42
# Size of the minority-class neighbourhood from which SMOTE interpolates (the value of the original paper).
SMOTE_NEIGHBORS = 5


class BalancingError(Exception):
    """Raised when the input data is missing, malformed, or fails a validation check."""


def log(message: str) -> None:
    """Writes a progress message to standard output, prefixed with the stage name.

    Args:
        message: Human-readable progress or diagnostic text.
    """
    print(f"[02_balancer] {message}", flush=True)


def class_summary(y: pd.Series) -> dict:
    """Summarises the class distribution of a binary target.

    Args:
        y: Binary target values.

    Returns:
        The number of rows, the count per class, and the empirical positive
        rate pi_hat = (1/n) * sum y_i.
    """
    counts = y.value_counts().sort_index()
    return {
        "rows": int(len(y)),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "positive_rate": float(y.mean()),
    }


def describe(label: str, y: pd.Series) -> None:
    """Logs the class distribution of a partition on a single line.

    Args:
        label: Partition name shown in the log.
        y: Binary target values of the partition.
    """
    summary = class_summary(y)
    counts = summary["class_counts"]
    log(f"  {label:<28} {summary['rows']:>7,} rows | class 0: {counts.get('0', 0):>6,} | "
        f"class 1: {counts.get('1', 0):>6,} | positive rate {summary['positive_rate']:.2%}")


def load_clean() -> pd.DataFrame:
    """Loads the clean table from step 01 and validates the preconditions of this stage.

    The checks are that person-level traits have no missing values (step 01
    replaced missing categories with 'Unknown'), that the target is strictly
    binary, and that the dataset meets the minimum size required by the
    project specification.

    Returns:
        The clean records restricted to the person traits and the target,
        with categorical traits as strings.

    Raises:
        BalancingError: If the file is missing or any precondition fails.
    """
    if not INPUT_FILE.is_file():
        raise BalancingError(f"Input not found: {INPUT_FILE}. Run 01_data_cleaner.py first.")
    try:
        df = pd.read_csv(INPUT_FILE, usecols=PERSON_COLUMNS + [TARGET_COLUMN], keep_default_na=False, na_values=[""])
    except ValueError as exc:
        raise BalancingError(f"{INPUT_FILE.name} is missing expected columns: {exc}") from exc
    if df[PERSON_COLUMNS].isna().any().any():
        raise BalancingError("Person-level features contain missing values; rerun 01_data_cleaner.py")
    if set(df[TARGET_COLUMN].unique()) - {0, 1}:
        raise BalancingError(f"Target '{TARGET_COLUMN}' must be binary 0/1")
    if len(df) < MIN_REQUIRED_RECORDS:
        raise BalancingError(f"Only {len(df):,} records; at least {MIN_REQUIRED_RECORDS:,} are required")
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype(str)
    log(f"Loaded {len(df):,} clean records (>= {MIN_REQUIRED_RECORDS:,} required) from {INPUT_FILE.name}")
    return df


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Partitions the data into stratified training, validation and test sets, before anything is fitted.

    The split is done in two stages. The test set (20%) is held out first.
    The remainder is then divided so that the validation set is 20% of the
    total, which means 0.20 / 0.80 = 25% of the remainder. Both stages
    stratify on the target and use the same seed, so the partitions are
    reproducible and each one preserves the population positive rate.

    Args:
        df: The validated clean records.

    Returns:
        A mapping {'train', 'validation', 'test'} to partitions with reset
        indices.
    """
    train_val, test = train_test_split(df, test_size=TEST_SHARE, stratify=df[TARGET_COLUMN], random_state=RANDOM_STATE)
    val_share_of_rest = VALIDATION_SHARE / (1 - TEST_SHARE)
    train, val = train_test_split(
        train_val, test_size=val_share_of_rest, stratify=train_val[TARGET_COLUMN], random_state=RANDOM_STATE
    )
    parts = {"train": train, "validation": val, "test": test}
    log(f"Stratified split {1 - VALIDATION_SHARE - TEST_SHARE:.0%} / {VALIDATION_SHARE:.0%} / {TEST_SHARE:.0%} "
        f"(random_state={RANDOM_STATE}):")
    for name, part in parts.items():
        describe(name, part[TARGET_COLUMN])
    return {name: part.reset_index(drop=True) for name, part in parts.items()}


def fit_preprocessors(train: pd.DataFrame) -> tuple[SimpleImputer, OneHotEncoder]:
    """Fits the preprocessing transformers on the training partition only.

    The SimpleImputer replaces a missing numeric value with the training
    median, which is robust to the heavy right tail of capital_gain, where a
    mean would be pulled upwards. The OneHotEncoder maps each categorical
    trait with K observed levels to K binary indicators. With
    handle_unknown='ignore', a category absent from training (for example a
    country not surveyed in 1994) is encoded as the all-zeros vector. The
    model then treats it as the reference profile and does not fail.

    Args:
        train: The training partition (raw, unencoded).

    Returns:
        A tuple (imputer, encoder), both fitted.
    """
    imputer = SimpleImputer(strategy="median").fit(train[NUMERIC_COLUMNS])
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.int8).fit(train[CATEGORICAL_COLUMNS])
    log(f"Fitted SimpleImputer(median) and OneHotEncoder on {len(train):,} training rows "
        f"({sum(len(c) for c in encoder.categories_)} one-hot columns)")
    return imputer, encoder


def transform(rows: pd.DataFrame, imputer: SimpleImputer, encoder: OneHotEncoder) -> pd.DataFrame:
    """Applies the exact feature transform that the inference API reproduces.

    The output is the concatenation [imputed numeric traits, one-hot
    categorical indicators], in the order recorded in feature_schema.json. The
    inference adapter (sklearn_predictor.py) applies the same two serialised
    transformers in the same order. This guarantees that a character is
    represented in production exactly as the training rows were.

    Args:
        rows: Raw person records.
        imputer: The fitted median imputer.
        encoder: The fitted one-hot encoder.

    Returns:
        The numeric feature matrix.

    Raises:
        BalancingError: If any missing value survives the transform.
    """
    numeric = pd.DataFrame(imputer.transform(rows[NUMERIC_COLUMNS]), columns=NUMERIC_COLUMNS, index=rows.index)
    encoded = pd.DataFrame(
        encoder.transform(rows[CATEGORICAL_COLUMNS]),
        columns=encoder.get_feature_names_out(CATEGORICAL_COLUMNS),
        index=rows.index,
    )
    features = pd.concat([numeric, encoded], axis=1)
    if features.isna().any().any():
        raise BalancingError("Feature matrix contains missing values after transformation")
    return features


def smote_balance(train: pd.DataFrame) -> pd.DataFrame:
    """Returns a 50/50 SMOTENC-balanced copy of the training rows (raw, unencoded).

    SMOTE (Chawla et al., 2002) synthesises minority-class samples by
    interpolation. For a minority sample x_i and one of its k nearest
    minority neighbours x_z, it creates

        x_new = x_i + lambda * (x_z - x_i),   with lambda ~ U(0, 1),

    a point on the segment between two real minority samples. SMOTENC extends
    the method to mixed data. Numeric traits are interpolated as above, while
    each categorical trait takes the most frequent level among the k
    neighbours, since a convex combination of categories is undefined.

    Numeric traits are standardised before the neighbour search, so that the
    Euclidean distance is not dominated by wide-range variables such as
    capital_gain. The synthetic numeric values are then mapped back to their
    original units, rounded and clipped to non-negative integers, as census
    quantities require.

    Args:
        train: The training partition (raw, unencoded). Only this partition
            is ever balanced.

    Returns:
        The balanced training rows (original plus synthetic) with the target
        column.

    Raises:
        BalancingError: If SMOTENC fails.
    """
    X = train[PERSON_COLUMNS].copy()
    y = train[TARGET_COLUMN]
    # Standardise so wide numeric ranges (capital_gain) do not dominate the neighbour search.
    scaler = StandardScaler()
    X[NUMERIC_COLUMNS] = scaler.fit_transform(X[NUMERIC_COLUMNS].astype("float64"))
    X[CATEGORICAL_COLUMNS] = X[CATEGORICAL_COLUMNS].astype(object)

    smote = SMOTENC(
        categorical_features=[X.columns.get_loc(c) for c in CATEGORICAL_COLUMNS],
        k_neighbors=SMOTE_NEIGHBORS,
        random_state=RANDOM_STATE,
    )
    log(f"Running SMOTENC on the training rows only (k_neighbors={SMOTE_NEIGHBORS})")
    try:
        X_res, y_res = smote.fit_resample(X, y)
    except ValueError as exc:
        raise BalancingError(f"SMOTENC failed: {exc}") from exc

    X_res = pd.DataFrame(X_res, columns=X.columns)
    X_res[NUMERIC_COLUMNS] = scaler.inverse_transform(X_res[NUMERIC_COLUMNS].astype("float64"))
    X_res[NUMERIC_COLUMNS] = X_res[NUMERIC_COLUMNS].round().clip(lower=0).astype("int64")
    for column in CATEGORICAL_COLUMNS:
        X_res[column] = X_res[column].astype(str)
    X_res[TARGET_COLUMN] = np.asarray(y_res, dtype="int64")
    log(f"  Generated {len(X_res) - len(train):,} synthetic minority rows")
    return X_res


def save(datasets: dict[str, pd.DataFrame], targets: dict[str, pd.Series], parts: dict[str, pd.DataFrame],
         imputer: SimpleImputer, encoder: OneHotEncoder, report: dict) -> None:
    """Persists the encoded partitions, the preprocessing artifacts, the feature schema and the report.

    feature_schema.json is the contract between training and inference. It
    declares the numeric and categorical inputs, the excluded columns, and
    the exact ordered list of model features. Both step 03 and the inference
    API verify their inputs against it.

    Args:
        datasets: Encoded feature matrices per partition.
        targets: Target vectors per partition.
        parts: Raw partitions, archived for exploratory analysis.
        imputer: The fitted median imputer.
        encoder: The fitted one-hot encoder.
        report: The balancing report.

    Raises:
        BalancingError: If the partitions disagree on their feature columns or
            a file cannot be written.
    """
    feature_columns = list(datasets["train"].columns)
    for name, frame in datasets.items():
        if list(frame.columns) != feature_columns:
            raise BalancingError(f"{name} features differ from the training features")
    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        for name, frame in datasets.items():
            frame.to_parquet(PROCESSED_DIR / f"X_{name}.parquet", index=False)
            targets[name].to_frame().to_parquet(PROCESSED_DIR / f"y_{name}.parquet", index=False)
            log(f"  Saved X_{name}.parquet {frame.shape} and y_{name}.parquet")
        raw = pd.concat([part.assign(split=name) for name, part in parts.items()], ignore_index=True)
        raw.to_parquet(RAW_SPLITS_FILE, index=False)
        joblib.dump(imputer, IMPUTER_FILE)
        joblib.dump(encoder, ENCODER_FILE)
        FEATURE_SCHEMA_FILE.write_text(json.dumps({
            "target": TARGET_COLUMN,
            "numeric_columns": NUMERIC_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "excluded_columns": {"fnlwgt": "census sampling weight, not a personal trait"},
            "feature_columns": feature_columns,
            "transform_steps": [
                "Apply imputer.joblib to numeric_columns (fills missing numeric values with training medians)",
                "Apply encoder.joblib to categorical_columns (unknown categories become all zeros)",
                "Concatenate [numeric_columns, encoded columns] in the order of feature_columns",
            ],
        }, indent=2), encoding="utf-8")
        BALANCING_REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
        for path in OBSOLETE_ARTIFACTS:
            if path.exists():
                path.unlink()
                log(f"  Removed obsolete artifact {path.relative_to(PIPELINE_DIR)}")
    except OSError as exc:
        raise BalancingError(f"Could not write outputs: {exc}") from exc
    log(f"  Saved {RAW_SPLITS_FILE.name}, {BALANCING_REPORT_FILE.name}, imputer, encoder and feature schema")


def main() -> int:
    """Runs the stage end to end: load, split, fit the preprocessors, balance, encode and save.

    The balanced copy is verified to be exactly 50/50 before anything is
    written.

    Returns:
        The process exit code: 0 on success, 1 on a BalancingError.
    """
    try:
        df = load_clean()
        parts = split(df)
        imputer, encoder = fit_preprocessors(parts["train"])

        smote_train = smote_balance(parts["train"])
        log("Class balance before and after SMOTE (training set only):")
        describe("train (natural)", parts["train"][TARGET_COLUMN])
        describe("train (SMOTE-balanced)", smote_train[TARGET_COLUMN])
        counts = smote_train[TARGET_COLUMN].value_counts()
        if counts.nunique() != 1:
            raise BalancingError(f"SMOTE did not balance the classes exactly: {counts.to_dict()}")

        datasets = {
            "train": transform(parts["train"], imputer, encoder),
            "train_smote": transform(smote_train, imputer, encoder),
            "validation": transform(parts["validation"], imputer, encoder),
            "test": transform(parts["test"], imputer, encoder),
        }
        targets = {
            "train": parts["train"][TARGET_COLUMN],
            "train_smote": smote_train[TARGET_COLUMN],
            "validation": parts["validation"][TARGET_COLUMN],
            "test": parts["test"][TARGET_COLUMN],
        }
        report = {
            "total_records": int(len(df)),
            "min_required_records": MIN_REQUIRED_RECORDS,
            "splits": {name: class_summary(parts[name][TARGET_COLUMN]) for name in parts},
            "train_smote": class_summary(smote_train[TARGET_COLUMN]),
            "synthetic_rows": int(len(smote_train) - len(parts["train"])),
            "smote": {"method": "SMOTENC", "k_neighbors": SMOTE_NEIGHBORS, "random_state": RANDOM_STATE},
        }
        log("Saving datasets and artifacts:")
        save(datasets, targets, parts, imputer, encoder, report)
    except BalancingError as exc:
        log(f"ERROR: {exc}")
        return 1
    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
