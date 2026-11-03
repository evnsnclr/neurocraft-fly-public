"""Protocol error types with safe, non-secret diagnostics."""

from __future__ import annotations


class ProtocolError(ValueError):
    """Base class for a rejected protocol or trace value."""

    code = "MALFORMED"


class OversizeFrameError(ProtocolError):
    code = "OVERSIZE"


class UnknownFieldError(ProtocolError):
    code = "UNKNOWN_FIELD"


class IdentityMismatchError(ProtocolError):
    code = "IDENTITY_MISMATCH"


class SequenceError(ProtocolError):
    code = "SEQUENCE"


class StaleMessageError(ProtocolError):
    code = "STALE"


class FutureMessageError(ProtocolError):
    code = "FUTURE"


class AuthenticationError(ProtocolError):
    code = "UNAUTHORIZED"


class ReceiptMismatchError(ProtocolError):
    code = "RECEIPT_MISMATCH"


class TraceIntegrityError(ProtocolError):
    code = "TRACE_INTEGRITY"
