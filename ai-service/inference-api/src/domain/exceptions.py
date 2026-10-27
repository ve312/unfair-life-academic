"""Domain-level errors. Infrastructure maps these to transport errors (e.g. HTTP 422)."""


class DomainError(Exception):
    """Base class for every error raised by the domain layer."""


class InvalidCharacterError(DomainError, ValueError):
    """A character profile violates a domain rule (missing field, out-of-range value)."""


class InvalidPredictionError(DomainError, ValueError):
    """A prediction outcome is internally inconsistent (e.g. probability outside [0, 1])."""
