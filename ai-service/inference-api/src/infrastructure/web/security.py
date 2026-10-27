"""Service-to-service authentication: only the NestJS BFF may call /predict.

The BFF sends the shared secret in the X-BFF-Token header; it must match the
BFF_SECRET_TOKEN environment variable (the same value the BFF and the CMS use).

Fallback: when BFF_SECRET_TOKEN is not set, 'dev-secret-token' is accepted in
development only (mirrors cms/src/policies/is-bff-server.ts). On Render (the
RENDER variable is injected automatically) or with ENVIRONMENT=production a
missing token fails closed, so a publicly known default never unlocks the API.
"""

import hashlib
import hmac
import logging
import os
from collections.abc import Mapping

from fastapi import Header, HTTPException, Request, status

logger = logging.getLogger(__name__)

BFF_TOKEN_HEADER = "X-BFF-Token"
DEV_FALLBACK_BFF_TOKEN = "dev-secret-token"

_warned_about_fallback = False


def _is_production(env: Mapping[str, str]) -> bool:
    """Detects a production deployment.

    Render sets the RENDER variable automatically. ENVIRONMENT=production
    covers any other host.

    Args:
        env: The process environment.

    Returns:
        True when the service runs in production.
    """
    return bool(env.get("RENDER")) or env.get("ENVIRONMENT", "").strip().lower() == "production"


def resolve_expected_token(env: Mapping[str, str] = os.environ) -> str | None:
    """Determines the token that callers must present.

    The policy fails closed in production. Without a configured secret, the
    function returns None, so every request is rejected, rather than accepting
    the publicly known development fallback.

    Args:
        env: The process environment (defaults to os.environ).

    Returns:
        The expected token, the development fallback outside production, or
        None when every request must be rejected.
    """
    global _warned_about_fallback
    configured = env.get("BFF_SECRET_TOKEN", "").strip()
    if configured:
        return configured
    if _is_production(env):
        logger.error("BFF_SECRET_TOKEN is not set in production; rejecting all /predict requests.")
        return None
    if not _warned_about_fallback:
        logger.warning("BFF_SECRET_TOKEN is not set; using the development fallback '%s'.", DEV_FALLBACK_BFF_TOKEN)
        _warned_about_fallback = True
    return DEV_FALLBACK_BFF_TOKEN


def _constant_time_equals(received: str, expected: str) -> bool:
    """Compares two secrets in time independent of their content.

    A naive comparison returns at the first differing character, which lets
    an attacker recover the secret one character at a time by measuring
    response times (a timing side channel). Hashing both values with SHA-256
    yields equal-length digests, and hmac.compare_digest then compares them
    in constant time.

    Args:
        received: The token presented by the caller.
        expected: The configured secret.

    Returns:
        True if the two tokens are equal.
    """
    # Hashing both values gives equal-length digests, so timing does not leak the token length.
    return hmac.compare_digest(
        hashlib.sha256(received.encode()).digest(), hashlib.sha256(expected.encode()).digest()
    )


def require_bff_token(
    request: Request, x_bff_token: str | None = Header(default=None, alias=BFF_TOKEN_HEADER)
) -> None:
    """FastAPI dependency that answers 403 unless the X-BFF-Token header matches BFF_SECRET_TOKEN.

    A missing header and a wrong token receive the same response, so the API
    does not reveal which of the two occurred. Every rejection is logged with
    the client address for auditing.

    Args:
        request: The incoming request, used for the audit log.
        x_bff_token: The value of the X-BFF-Token header, if present.

    Raises:
        HTTPException: With status 403 if the token is missing or invalid,
            or if no secret is configured in production.
    """
    expected = resolve_expected_token()
    if expected is None or x_bff_token is None or not _constant_time_equals(x_bff_token, expected):
        client = request.client.host if request.client else "unknown"
        logger.warning("Rejected %s %s from %s: missing or invalid %s", request.method, request.url.path, client,
                       BFF_TOKEN_HEADER)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
