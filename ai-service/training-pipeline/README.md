# Unfair Life – AI Training Pipeline

Offline data science environment for the Unfair Life AI Prediction Microservice.

It cleans the 1994 Adult Income census, splits it into train / validation / test sets, documents a SMOTE balancing comparison, trains and compares candidate classifiers (Logistic Regression, KNN, SVM, Random Forest and HistGradientBoosting), deploys a stacking ensemble (HistGradientBoosting + Logistic Regression, combined in log-odds space by a logistic meta-learner), chooses a selective-classification decision policy, and exports the artifacts the inference API serves.

The exported model is consumed by the FastAPI server in `/ai-service/inference-api`, which loads it at startup to serve the `/predict` endpoint. This pipeline never runs in production; it only produces the artifacts.

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 01 | `src/01_data_cleaner.py` | Cleans the Adult census only (the SocialLatino country files are excluded). Writes `clean_adult.csv` and `cleaning_report.json`. |
| 02 | `src/02_balancer.py` | Stratified 60/20/20 split, fits the imputer and encoder on training rows, and saves a SMOTENC-balanced copy of the training set for comparison. `fnlwgt` (a census sampling weight) is not a feature. |
| 03 | `src/03_trainer.py` | Trains every model on the natural and the SMOTE data and selects by **validation** average precision. SMOTE lowered average precision in every family and skewed probabilities, so the deployed model uses the natural distribution. |
| 04 | `src/04_evaluator.py` | Chooses the uncertainty band and a 95%-precision threshold on validation, then reports once on the untouched test set with confidence intervals. |
| — | `notebooks/01_math_and_eda.ipynb` | Reads the artifacts and writes figures (`artifacts/figures/`) and LaTeX equations and tables (`artifacts/latex/`) for the academic report. |

## Directory Structure

| Path              | Purpose                                                     |
|-------------------|-------------------------------------------------------------|
| `data/raw/`       | Original, untouched source CSVs.                            |
| `data/processed/` | Clean data, split and encoded datasets, cleaning and balancing reports. |
| `notebooks/`      | The math, EDA and traceability notebook.                    |
| `src/`            | Pipeline steps 01–04.                                        |
| `artifacts/`      | Model, preprocessing objects, decision policy, reports, figures and LaTeX. |

## Setup

Requires Python 3.10+.

```bash
cd ai-service/training-pipeline
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the pipeline, then execute the notebook:

```bash
python src/01_data_cleaner.py
python src/02_balancer.py
python src/03_trainer.py
python src/04_evaluator.py
jupyter nbconvert --to notebook --execute --inplace notebooks/01_math_and_eda.ipynb
```

## Dependencies

- `pandas`, `pyarrow`: data loading, cleaning and parquet storage
- `scikit-learn`: preprocessing, models and metrics
- `imbalanced-learn`: SMOTENC for the balancing comparison
- `shap`, `matplotlib`: explainability and figures for the notebook
- `jupyter`: notebook environment
