"""Allowed browser origins for the inference API.

BFF_URL lists the BFF's public origin(s), comma-separated, e.g.
https://unfair-life-bff.onrender.com. Without it, only the local BFF
(localhost / 127.0.0.1 on port 3000) is allowed, for development.

CORS is enforced by browsers only. The BFF calls this API server to server,
so these rules neither block nor allow those calls; they only govern any
browser page that tries to call the API.
"""

from collections.abc import Mapping
from urllib.parse import urlsplit

LOCAL_BFF_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


def cors_origins(env: Mapping[str, str]) -> list[str]:
    """Parses the allowed browser origins from BFF_URL, or falls back to the local BFF origins.

    Each entry must be a bare origin (scheme and host, optionally a port, but
    no path). A browser's Origin header has exactly that form, so any other
    value could never match it, and accepting one would hide a
    misconfiguration. Duplicates are removed and order is preserved.

    Args:
        env: The process environment (or any mapping with the same keys).

    Returns:
        The list of allowed origins.

    Raises:
        ValueError: If an entry is not a bare http(s) origin.
    """
    raw = env.get("BFF_URL", "") or ""
    origins = [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
    for origin in origins:
        parts = urlsplit(origin)
        if parts.scheme not in ("http", "https") or not parts.netloc or parts.path or parts.query or parts.fragment:
            raise ValueError(f"BFF_URL entries must be bare origins like 'https://unfair-life-bff.onrender.com', got '{origin}'")
    return list(dict.fromkeys(origins)) if origins else list(LOCAL_BFF_ORIGINS)
