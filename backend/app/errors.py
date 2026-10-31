"""Typed provider errors. The API layer turns these into clear 503 messages."""

from typing import Literal

ErrorKind = Literal[
    "timeout", "network", "http_error", "rate_limited", "malformed", "not_configured"
]


class SourceUnavailable(Exception):
    """A live data source could not give us a trustworthy answer. Never replaced by guesses."""

    def __init__(
        self,
        source: str,
        kind: ErrorKind,
        reason: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(f"{source}: {kind}: {reason}")
        self.source = source
        self.kind: ErrorKind = kind
        self.reason = reason
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        return self.kind in {"timeout", "network", "rate_limited", "http_error"}

    def user_message(self) -> str:
        if self.kind == "not_configured":
            return f"{self.source} is not configured: {self.reason}."
        return (
            f"Live data from {self.source} could not be retrieved ({self.reason}). "
            "No substitute values are shown."
        )


class MalformedPayload(SourceUnavailable):
    """The source answered, but not in the shape its documentation promises."""

    def __init__(self, source: str, reason: str) -> None:
        super().__init__(source, "malformed", reason)


class NotConfigured(SourceUnavailable):
    """A required credential or setting is missing."""

    def __init__(self, source: str, reason: str) -> None:
        super().__init__(source, "not_configured", reason)
