"""Step 01 of the Unfair Life training pipeline: clean the 1994 Adult Income census.

This module transforms the raw census extract into a clean, analysis-ready
table and records every transformation it applies, so that the data lineage
of the model can be audited. Its output is the single source of truth for all
later stages.

Data source. Only the 1994 Adult Income census (48,842 records, 15 columns)
is used. The World Bank "socialLatino" files are deliberately excluded. They
describe countries, not individuals, so joining them to persons would assign
the same aggregate value to everyone born in a country. They could moreover
be matched to only about 3% of individuals, and the remaining 97% would have
had to be imputed. Merging them would therefore have added imputed noise
rather than information (see the Phase 1 data-integrity decision).

Cleaning steps (each one is counted in data/processed/cleaning_report.json):

* Column names are normalised to snake_case, giving a stable vocabulary of
  identifiers for the rest of the pipeline.
* The census marker '?' is treated as missing. Missing categories become the
  explicit level 'Unknown' instead of having their rows deleted. Listwise
  deletion is unbiased only if values are missing completely at random, and
  non-response in a census is plausibly related to employment status. The
  explicit level keeps those rows and lets the model learn whether
  missingness itself is informative.
* The test split's trailing-period labels ('>50K.') are unified with '>50K'.
* The target is encoded as income_above_50k in {0, 1}, the Bernoulli outcome
  whose conditional probability the model estimates.
* Exact duplicate rows are removed. Duplicates would give some individuals
  double weight in the likelihood, and across partitions they would leak
  information from training into evaluation.

Run from the training-pipeline folder:
    .venv/Scripts/python.exe src/01_data_cleaner.py
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PIPELINE_DIR / "data" / "raw" / "income-dataset.csv"
PROCESSED_DIR = PIPELINE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "clean_adult.csv"
REPORT_FILE = PROCESSED_DIR / "cleaning_report.json"

ENCODINGS = ("utf-8", "cp1252")
MISSING_MARKER = "?"
UNKNOWN_CATEGORY = "Unknown"
TARGET_COLUMN = "income_above_50k"
VALID_LABELS = {"<=50K", ">50K"}

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
# fnlwgt is kept in the clean file for EDA, but it is a census sampling weight,
# not a personal trait, so step 02 does not use it as a model feature.
NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


class CleaningError(Exception):
    """Raised when the raw file is missing, unreadable, or structurally invalid."""


def log(message: str) -> None:
    """Writes a progress message to standard output, prefixed with the stage name.

    Args:
        message: Human-readable progress or diagnostic text.
    """
    print(f"[01_data_cleaner] {message}", flush=True)


def to_snake_case(name: str) -> str:
    """Normalises a column name to snake_case.

    Every run of non-alphanumeric characters is replaced by a single
    underscore, leading and trailing underscores are removed, and the result
    is lower-cased. The mapping is deterministic, so the same raw header
    always yields the same identifier.

    Args:
        name: The raw column header, e.g. 'education-num'.

    Returns:
        The normalised identifier, e.g. 'education_num'.
    """
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()


def read_raw() -> pd.DataFrame:
    """Reads the raw census file and verifies its structure.

    The text encoding is detected by trial decoding: UTF-8 first, then
    Windows-1252. This guarantees that no character is silently corrupted. The
    census '?' marker is parsed as missing at read time, and leading spaces
    after the delimiters are stripped.

    Returns:
        The raw records with snake_case column names.

    Raises:
        CleaningError: If the file is missing, cannot be decoded or parsed,
            or lacks an expected column.
    """
    if not RAW_FILE.is_file():
        raise CleaningError(f"Raw dataset not found: {RAW_FILE}")
    raw = RAW_FILE.read_bytes()
    for encoding in ENCODINGS:
        try:
            raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise CleaningError(f"Could not decode {RAW_FILE.name} with any of {ENCODINGS}")

    log(f"Loading {RAW_FILE.name} (encoding={encoding})")
    try:
        df = pd.read_csv(RAW_FILE, encoding=encoding, skipinitialspace=True, na_values=[MISSING_MARKER])
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise CleaningError(f"Failed to parse {RAW_FILE.name}: {exc}") from exc

    df.columns = [to_snake_case(column) for column in df.columns]
    missing = [column for column in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + ["income"] if column not in df.columns]
    if missing:
        raise CleaningError(f"{RAW_FILE.name} is missing expected columns: {missing}")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Applies the cleaning steps and measures each one.

    The steps run in a fixed order:

    1. Categorical text and labels are stripped of surrounding whitespace.
    2. Missing values are counted per column, before any of them is treated.
    3. Numeric columns are coerced to numbers, and rows whose numeric traits
       or label are missing are dropped. A row without a label cannot
       contribute to supervised learning. In this census no row is dropped.
    4. Missing categories are replaced by the explicit level 'Unknown'.
    5. Labels are normalised and validated against VALID_LABELS, then encoded
       as the binary target.
    6. Exact duplicate rows are removed.

    The function also records the class counts and the positive rate
    pi = P(Y = 1). That rate is the prior that stratified splitting preserves
    in step 02, and the reference against which the calibration of every
    model is judged.

    Args:
        df: The raw records returned by read_raw().

    Returns:
        A tuple (clean_df, report): the cleaned records and a dictionary with
        the count produced by each step.

    Raises:
        CleaningError: If an income label outside VALID_LABELS is found.
    """
    report: dict = {"raw_rows": int(len(df)), "raw_columns": int(df.shape[1])}
    log(f"  Raw data: {len(df):,} rows x {df.shape[1]} columns")

    for column in CATEGORICAL_COLUMNS + ["income"]:
        df[column] = df[column].astype("string").str.strip()

    missing_by_column = {column: int(count) for column, count in df.isna().sum().items() if count > 0}
    report["missing_values_by_column"] = missing_by_column
    log(f"  Missing values ('?' or blank): {missing_by_column}")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    rows_before = len(df)
    df = df.dropna(subset=NUMERIC_COLUMNS + ["income"])
    report["rows_dropped_missing_numeric_or_label"] = int(rows_before - len(df))

    df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].fillna(UNKNOWN_CATEGORY)

    # The census test split marks labels with a trailing period ('>50K.').
    df["income"] = df["income"].str.rstrip(".")
    unexpected = set(df["income"].unique()) - VALID_LABELS
    if unexpected:
        raise CleaningError(f"Unexpected income labels: {sorted(unexpected)}")
    df[TARGET_COLUMN] = (df["income"] == ">50K").astype("int64")
    df = df.drop(columns="income")

    rows_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicate_rows_removed"] = int(rows_before - len(df))
    log(f"  Removed {report['duplicate_rows_removed']:,} exact duplicate rows")

    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].astype("int64")
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    report["clean_rows"] = int(len(df))
    report["class_counts"] = {str(label): int(count) for label, count in counts.items()}
    report["positive_rate"] = float(df[TARGET_COLUMN].mean())
    log(f"  Clean data: {len(df):,} rows | class counts {report['class_counts']} | "
        f"positive rate {report['positive_rate']:.2%}")
    return df, report


def save(df: pd.DataFrame, report: dict) -> None:
    """Persists the clean table and its cleaning report.

    Args:
        df: The cleaned records.
        report: The per-step counts returned by clean().

    Raises:
        CleaningError: If an output file cannot be written.
    """
    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as exc:
        raise CleaningError(f"Could not write outputs to {PROCESSED_DIR}: {exc}") from exc
    log(f"Saved {OUTPUT_FILE.relative_to(PIPELINE_DIR)} ({OUTPUT_FILE.stat().st_size / 1_048_576:.1f} MB) "
        f"and {REPORT_FILE.relative_to(PIPELINE_DIR)}")


def main() -> int:
    """Runs the stage end to end: read, clean and save.

    Returns:
        The process exit code: 0 on success, 1 on a CleaningError.
    """
    try:
        df, report = clean(read_raw())
        save(df, report)
    except CleaningError as exc:
        log(f"ERROR: {exc}")
        return 1
    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
