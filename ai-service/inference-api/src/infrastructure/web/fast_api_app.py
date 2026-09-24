"""FastAPI application and composition root.

The model artifacts are loaded exactly once, in the lifespan handler, and the
wired use case is stored in app.state for every request to share. If the
artifacts cannot be loaded, startup fails instead of serving a broken API.

Environment variables:
    AI_ARTIFACTS_DIR   override the Phase 1 artifacts folder
                       (default: <ai-service>/training-pipeline/artifacts)
    BFF_URL            allowed browser origin(s), comma-separated (see cors.py)
    BFF_SECRET_TOKEN   shared secret the BFF sends in X-BFF-Token (see security.py)
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.application.predict_mobility import PredictMobilityUseCase
from src.infrastructure.adapters.ml.sklearn_predictor import SklearnPredictor
from src.infrastructure.web.controllers import router
from src.infrastructure.web.cors import cors_origins
from src.infrastructure.web.security import resolve_expected_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Loads the model once at start-up and releases it at shutdown.

    Loading in the lifespan, instead of per request, keeps the latency of a
    prediction to the cost of inference alone. It also makes start-up
    fail-fast: an ArtifactLoadError aborts the server before it accepts any
    traffic.

    Args:
        app: The FastAPI application, whose state receives the wired use case,
            the model name and the decision policy.

    Yields:
        Control to the running application.

    Raises:
        ArtifactLoadError: If the artifacts are missing or inconsistent.
    """
    artifacts_dir = os.environ.get("AI_ARTIFACTS_DIR") or None
    logger.info("Loading prediction model (artifacts_dir=%s)", artifacts_dir or "default")
    predictor = SklearnPredictor(artifacts_dir=artifacts_dir)

    app.state.predict_mobility = PredictMobilityUseCase(predictor)
    app.state.model_name = predictor.model_name
    app.state.policy = predictor.policy
    logger.info(
        "Inference API ready: model=%s band=[%.2f, %.2f] threshold=%.4f", predictor.model_name,
        predictor.policy.lower_bound, predictor.policy.upper_bound, predictor.policy.threshold,
    )
    try:
        yield
    finally:
        app.state.predict_mobility = None
        logger.info("Inference API shut down; model released")


app = FastAPI(
    title="Unfair Life - AI Inference API",
    description="Predicts a character's social mobility (income above 50K) for the NestJS BFF.",
    version="1.0.0",
    lifespan=lifespan,
)
allowed_origins = cors_origins(os.environ)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
    max_age=600,
)
logger.info("CORS allowed for %s", ", ".join(allowed_origins))
resolve_expected_token()  # log a missing-token problem at startup, not on the first request
app.include_router(router)
