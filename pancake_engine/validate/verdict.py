"""ValidationVerdict + ValidationError + ValidationWarning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ValidationError", "ValidationWarning", "ValidationVerdict"]


@dataclass(frozen=True)
class ValidationError:
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


@dataclass(frozen=True)
class ValidationWarning:
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


@dataclass
class ValidationVerdict:
    """Aggregate result of all validation checks.

    ``ok`` is true iff ``errors`` is empty. ``warnings`` never affect ``ok``
    — they surface for the caller to display but never block the run.
    """

    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def merge(self, other: "ValidationVerdict") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def add_error(self, code: str, message: str, **context: Any) -> None:
        self.errors.append(ValidationError(code=code, message=message, context=dict(context)))

    def add_warning(self, code: str, message: str, **context: Any) -> None:
        self.warnings.append(
            ValidationWarning(code=code, message=message, context=dict(context))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }
