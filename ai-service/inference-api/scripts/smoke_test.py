"""Smoke test: load the Phase 1 artifacts and run predictions through the use case.

Run from the inference-api folder:
    .venv/Scripts/python.exe scripts/smoke_test.py
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.application.predict_mobility import PredictMobilityUseCase  # noqa: E402
from src.domain.entities.character import Character  # noqa: E402
from src.domain.exceptions import InvalidCharacterError  # noqa: E402
from src.infrastructure.adapters.ml.sklearn_predictor import SklearnPredictor  # noqa: E402

CHARACTERS = {
    "Rural farmhand, Guatemala": Character(
        age=23, workclass="Private", education="7th-8th", education_num=4,
        marital_status="Never-married", occupation="Farming-fishing", relationship="Own-child",
        race="Amer-Indian-Eskimo", sex="Male", hours_per_week=50, native_country="Guatemala",
    ),
    "Office clerk, Mexico": Character(
        age=34, workclass="Private", education="HS-grad", education_num=9,
        marital_status="Married-civ-spouse", occupation="Adm-clerical", relationship="Wife",
        race="White", sex="Female", hours_per_week=40, native_country="Mexico",
    ),
    "Surgeon with investments, United States": Character(
        age=48, workclass="Self-emp-inc", education="Prof-school", education_num=15,
        marital_status="Married-civ-spouse", occupation="Prof-specialty", relationship="Husband",
        race="White", sex="Male", hours_per_week=60, native_country="United-States",
        capital_gain=15024,
    ),
    "Engineer, unseen country (Chile)": Character(
        age=39, workclass="Private", education="Masters", education_num=14,
        marital_status="Married-civ-spouse", occupation="Exec-managerial", relationship="Husband",
        race="White", sex="Male", hours_per_week=45, native_country="Chile",
    ),
}


def main() -> int:
    """Loads the artifacts, scores four reference characters and checks a domain rejection.

    The reference profiles span the probability range: a deterministic "no",
    a deterministic "yes", and a country absent from the training data. A
    single run therefore exercises the whole serving path (artifact loading,
    consistency checks, transform, inference and policy) without starting
    the HTTP server.

    Returns:
        The process exit code: 0 if every check passes, 1 if an invalid
        character is accepted.
    """
    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(name)s: %(message)s")

    start = time.perf_counter()
    predictor = SklearnPredictor()
    use_case = PredictMobilityUseCase(predictor)
    print(f"\nLoaded artifacts in {time.perf_counter() - start:.2f}s | model={predictor.model_name} "
          f"| band=[{predictor.policy.lower_bound:.2f}, {predictor.policy.upper_bound:.2f}] "
          f"| threshold={predictor.threshold:.4f}\n")

    print(f"{'CHARACTER':<42}{'PROBABILITY':>12}{'ZONE':>14}{'LATENCY':>11}")
    print("-" * 79)
    for label, character in CHARACTERS.items():
        start = time.perf_counter()
        result = use_case(character)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"{label:<42}{result.probability:>12.4f}{result.zone.value:>14}{elapsed_ms:>9.1f}ms")
    print("-" * 79)

    try:
        Character(**{**CHARACTERS["Office clerk, Mexico"].to_features(), "age": 12})
        print("FAIL: invalid character was accepted")
        return 1
    except InvalidCharacterError as exc:
        print(f"Invalid input rejected by the domain: {exc}")

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
