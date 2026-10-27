"""HTTP controllers: translate between JSON and the application use case.

The controllers form the inbound adapter of the hexagonal architecture.
They validate transport input (through the Pydantic schemas), map it to
domain entities, invoke the use case, and translate domain errors into HTTP
status codes:

- 422 for an invalid profile;
- 500 for an internal inconsistency;
- 503 when the model is not loaded.

They hold no business logic.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.application.predict_mobility import PredictMobilityUseCase
from src.domain.exceptions import DomainError, InvalidCharacterError
from src.infrastructure.web.schemas import CharacterRequest, ErrorResponse, HealthResponse, PredictionResponse
from src.infrastructure.web.security import require_bff_token

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_use_case(request: Request) -> PredictMobilityUseCase:
    """Retrieves the use case wired at start-up by the lifespan handler.

    Args:
        request: The incoming request, whose application state holds the use
            case.

    Returns:
        The shared PredictMobilityUseCase.

    Raises:
        HTTPException: With status 503 if the model is not loaded (for
            example, during shutdown).
    """
    use_case = getattr(request.app.state, "predict_mobility", None)
    if use_case is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Prediction model is not loaded")
    return use_case


# Declared with plain `def`: scoring is CPU-bound, so FastAPI runs it in its
# worker thread pool instead of blocking the event loop.
@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict a character's social mobility",
    dependencies=[Depends(require_bff_token)],
    responses={
        status.HTTP_403_FORBIDDEN: {"model": ErrorResponse, "description": "Missing or invalid X-BFF-Token header"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Request or domain validation failed"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse, "description": "Model failed to score"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
def predict(payload: CharacterRequest, request: Request) -> PredictionResponse:
    """Estimates the social-mobility probability of one character (POST /predict).

    Access is restricted to the BFF by the require_bff_token dependency,
    which runs before this handler.

    Args:
        payload: The character profile, already validated by Pydantic.
        request: The incoming request, used to reach the shared use case.

    Returns:
        The probability, its decision zone and the policy that produced it.

    Raises:
        HTTPException: 422 if a domain rule rejects the profile; 500 if the
            prediction violates a domain invariant or the model fails; 503 if
            the model is not loaded.
    """
    use_case = _get_use_case(request)

    try:
        character = payload.to_domain()
    except InvalidCharacterError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc

    try:
        result = use_case.execute(character)
    except DomainError as exc:
        logger.error("Domain rule violated while predicting: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed due to an internal error"
        ) from exc

    logger.info(
        "Prediction: country=%s education=%s probability=%.4f has_mobility=%s",
        character.native_country, character.education, result.probability, result.has_mobility,
    )
    return PredictionResponse.from_domain(result)


@router.get("/health", response_model=HealthResponse, summary="Liveness and model status")
def health(request: Request) -> HealthResponse:
    """Reports liveness together with the identity of the served model (GET /health).

    The endpoint is public on purpose: hosting platforms probe it
    unauthenticated, and it reveals no data beyond the published policy.

    Args:
        request: The incoming request, used to reach the application state.

    Returns:
        The model name and its decision policy.

    Raises:
        HTTPException: With status 503 if the model is not loaded.
    """
    _get_use_case(request)
    return HealthResponse(
        status="ok",
        model=request.app.state.model_name,
        lower_bound=request.app.state.policy.lower_bound,
        upper_bound=request.app.state.policy.upper_bound,
        threshold=request.app.state.policy.threshold,
    )
