from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from hrm.multimodal.structured.schema import StructuredSchema, FieldSchema
from hrm.multimodal.types import DecodedModality, ModalityInput


@dataclass(frozen=True)
class StructuredDecoder:
    schema: StructuredSchema

    def decode(self, source: Any, source_id: str, timestamp: float | None = None) -> DecodedModality:
        if isinstance(source, str):
            text = source
        elif isinstance(source, (bytes, bytearray)):
            text = source.decode("utf-8")
        else:
            raise ValueError("Structured source must be a text string or bytes")

        if text.lstrip().startswith("{") or text.lstrip().startswith("["):
            payload = json.loads(text)
        else:
            reader = csv.DictReader(io.StringIO(text))
            payload = [row for row in reader]

        records = self._validate_payload(payload)
        tensor, mask = self._encode_records(records)
        metadata = {"schema_id": self.schema.schema_id, "version": self.schema.version, "record_count": len(records)}
        return DecodedModality(
            modality="structured",
            source_id=source_id,
            tensor=tensor,
            mask=mask,
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            timestamp=timestamp,
            metadata=metadata,
        )

    def _validate_payload(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("Structured payload must be a JSON object, JSON array, or CSV table")
        records: list[dict[str, Any]] = []
        for idx, record in enumerate(payload):
            if not isinstance(record, dict):
                raise ValueError("Each record must be an object")
            normalized: dict[str, Any] = {}
            for field in self.schema.fields:
                if field.name not in record:
                    if field.required:
                        raise ValueError(f"Missing required field: {field.name}")
                    normalized[field.name] = None
                    continue
                value = record[field.name]
                if value is None:
                    if not field.nullable:
                        raise ValueError(f"Field not nullable: {field.name}")
                    normalized[field.name] = None
                    continue
                if field.dtype == "numeric":
                    numeric = float(value)
                    if field.numeric_range is not None:
                        if not field.numeric_range[0] <= numeric <= field.numeric_range[1]:
                            raise ValueError(f"Numeric range violation for {field.name}")
                    normalized[field.name] = numeric
                elif field.dtype == "categorical":
                    string_value = str(value)
                    if field.categorical_values is not None and string_value not in field.categorical_values:
                        raise ValueError(f"Unknown categorical value for {field.name}")
                    normalized[field.name] = string_value
                else:
                    normalized[field.name] = str(value)
            records.append(normalized)
        return records

    def _encode_records(self, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        rows = []
        masks = []
        for record in records:
            row_values: list[float] = []
            row_mask: list[float] = []
            for field in self.schema.fields:
                value = record[field.name]
                if value is None:
                    row_values.append(0.0)
                    row_mask.append(0.0)
                elif field.dtype == "numeric":
                    row_values.append(float(value))
                    row_mask.append(1.0)
                elif field.dtype == "categorical":
                    index = field.categorical_values.index(str(value)) if field.categorical_values else 0
                    row_values.append(float(index))
                    row_mask.append(1.0)
                else:
                    row_values.append(float(len(str(value))))
                    row_mask.append(1.0)
            rows.append(row_values)
            masks.append(row_mask)
        return np.asarray(rows, dtype=np.float32), np.asarray(masks, dtype=np.float32)
