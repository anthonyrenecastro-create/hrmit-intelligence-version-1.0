from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from hrm.multimodal.structured.schema import FieldSchema, StructuredSchema
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
        metadata = {
            "schema_id": self.schema.schema_id,
            "version": self.schema.version,
            "record_count": len(records),
            "fields": [field.name for field in self.schema.fields],
        }
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
                if value is None or value == "":
                    if not field.nullable:
                        raise ValueError(f"Field not nullable: {field.name}")
                    normalized[field.name] = None
                    continue
                if field.dtype == "numeric":
                    numeric = float(value)
                    if field.numeric_range is not None and not field.numeric_range[0] <= numeric <= field.numeric_range[1]:
                        raise ValueError(f"Numeric range violation for {field.name}")
                    normalized[field.name] = numeric
                elif field.dtype == "categorical":
                    string_value = str(value)
                    if field.categorical_values is not None and string_value not in field.categorical_values:
                        raise ValueError(f"Unknown categorical value for {field.name}")
                    normalized[field.name] = string_value
                elif field.dtype == "timestamp":
                    normalized[field.name] = self._parse_timestamp(value)
                else:
                    normalized[field.name] = str(value)
            records.append(normalized)
        return records

    def _parse_timestamp(self, value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(datetime.fromisoformat(value).timestamp())
        raise ValueError(f"Invalid timestamp value: {value}")

    def _encode_records(self, records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        rows = []
        masks = []
        timestamps: list[float] = []
        for record in records:
            row_values: list[float] = []
            row_mask: list[float] = []
            for field in self.schema.fields:
                value = record[field.name]
                if value is None:
                    row_values.append(0.0)
                    row_mask.append(0.0)
                    continue
                if field.dtype == "numeric":
                    numeric = float(value)
                    low, high = field.numeric_range or (0.0, 1.0)
                    norm = (numeric - low) / max(high - low, 1e-6)
                    row_values.append(float(np.clip(norm, 0.0, 1.0)))
                    row_mask.append(1.0)
                elif field.dtype == "categorical":
                    categories = field.categorical_values or ()
                    one_hot = [0.0] * len(categories)
                    if str(value) in categories:
                        one_hot[categories.index(str(value))] = 1.0
                    row_values.extend(one_hot)
                    row_mask.extend([1.0] * len(categories))
                elif field.dtype == "timestamp":
                    timestamp = float(value)
                    row_values.append(timestamp)
                    row_mask.append(1.0)
                    timestamps.append(timestamp)
                else:
                    row_values.append(float(len(str(value))))
                    row_mask.append(1.0)
            rows.append(row_values)
            masks.append(row_mask)

        tensor = np.asarray(rows, dtype=np.float32)
        mask_array = np.asarray(masks, dtype=np.float32)
        if tensor.ndim == 2 and tensor.shape[0] > 1 and tensor.shape[1] > 0:
            if timestamps:
                min_time = min(timestamps)
                time_offset = [(t - min_time) if t > 0 else 0.0 for t in timestamps]
                tensor[:, -1] = np.asarray(time_offset, dtype=np.float32)[: tensor.shape[0]]
        return tensor, mask_array
