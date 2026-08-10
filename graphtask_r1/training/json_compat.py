from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def to_json_compatible(value: object) -> object:
    """Recursively replace Arrow/NumPy containers and scalars with JSON-native values."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [to_json_compatible(item) for item in value]

    as_py: Any = getattr(value, "as_py", None)
    if callable(as_py):
        return to_json_compatible(as_py())
    tolist: Any = getattr(value, "tolist", None)
    if callable(tolist):
        return to_json_compatible(tolist())
    item: Any = getattr(value, "item", None)
    if callable(item):
        return to_json_compatible(item())
    raise TypeError(f"unsupported JSON value: {type(value).__module__}.{type(value).__name__}")
