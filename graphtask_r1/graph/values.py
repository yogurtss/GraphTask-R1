from __future__ import annotations

from datetime import date


def format_attribute_value(value: str, datatype: str, unit: str | None) -> str:
    if datatype == "quantity":
        return value if unit in {None, "", "1"} else f"{value} {unit}"
    if datatype == "year":
        return str(int(value))
    if datatype == "date":
        text = value.replace("-", "/")
        sign = -1 if text.startswith("/") else 1
        if sign < 0:
            text = text[1:]
        parts = text.split("/")
        if len(parts) == 3:
            year, month, day = (int(part) for part in parts)
            year *= sign
            if year >= 1:
                return date(year, month, day).isoformat()
            return f"{year:05d}-{month:02d}-{day:02d}"
    return value


def attribute_sort_key(
    value: str, datatype: str, unit: str | None
) -> tuple[int, str, float, int, int]:
    if datatype == "quantity":
        return (0, unit or "1", float(value), 0, 0)
    if datatype == "year":
        return (1, "", float(int(value)), 0, 0)
    if datatype == "date":
        text = value.replace("-", "/")
        sign = -1 if text.startswith("/") else 1
        if sign < 0:
            text = text[1:]
        parts = text.split("/")
        if len(parts) != 3:
            raise ValueError(f"invalid date attribute: {value!r}")
        year, month, day = (int(part) for part in parts)
        year *= sign
        return (1, "", float(year), month, day)
    raise ValueError(f"attribute type {datatype!r} is not orderable")
