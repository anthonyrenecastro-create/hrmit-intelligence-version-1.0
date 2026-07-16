from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FieldSchema:
    name: str
    dtype: str
    required: bool
    nullable: bool
    categorical_values: tuple[str, ...] | None = None
    numeric_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class StructuredSchema:
    schema_id: str
    fields: tuple[FieldSchema, ...]
    version: str

    def field_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def get_field(self, name: str) -> FieldSchema:
        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(f"Field {name} not defined in schema")
