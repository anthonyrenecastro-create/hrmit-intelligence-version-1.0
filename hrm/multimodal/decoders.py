from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .types import DecodedModality, ModalityInput

Image.MAX_IMAGE_PIXELS = 25_000_000


class DecodeError(ValueError):
    pass


class ImageDecoder:
    formats = {"PNG", "JPEG"}

    def decode(self, value: ModalityInput) -> DecodedModality:
        source = value.payload
        try:
            image = Image.open(io.BytesIO(source) if isinstance(source, bytes) else Path(source))
            image.verify()
            image = Image.open(io.BytesIO(source) if isinstance(source, bytes) else Path(source))
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise DecodeError(f"Malformed image {value.source_id}: {error}") from error
        if image.format not in self.formats:
            raise DecodeError(f"Unsupported image format: {image.format}")
        original_mode, original_size = image.mode, image.size
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        metadata = {**value.metadata, "format": image.format, "original_mode": original_mode,
                    "original_dimensions": original_size, "channel_layout": "HWC"}
        return DecodedModality("vision", value.source_id, array, None, array.shape,
                               str(array.dtype), value.timestamp, metadata)


class AudioDecoder:
    def decode(self, value: ModalityInput) -> DecodedModality:
        source = value.payload
        try:
            stream = io.BytesIO(source) if isinstance(source, bytes) else str(Path(source))
            with wave.open(stream, "rb") as wav:
                channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
                if width not in (1, 2, 4):
                    raise DecodeError(f"Unsupported PCM sample width: {width}")
                raw = wav.readframes(frames)
        except (wave.Error, OSError, EOFError) as error:
            raise DecodeError(f"Malformed WAV {value.source_id}: {error}") from error
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if width == 1:
            audio = (audio - 128.0) / 128.0
        else:
            audio /= float(2 ** (8 * width - 1))
        audio = audio.reshape(-1, channels).mean(axis=1)
        metadata = {**value.metadata, "sample_rate": rate, "channels": channels,
                    "sample_width": width, "frame_count": frames, "format": "WAV"}
        return DecodedModality("audio", value.source_id, audio, None, audio.shape,
                               str(audio.dtype), value.timestamp, metadata)


class StructuredDecoder:
    def decode(self, value: ModalityInput) -> DecodedModality:
        if not isinstance(value.payload, dict) or not value.payload:
            raise DecodeError("Structured payload must be a non-empty mapping")
        schema = value.metadata.get("schema")
        if not isinstance(schema, dict):
            raise DecodeError("Structured input requires metadata.schema")
        fields, values, mask = [], [], []
        for name, definition in schema.items():
            fields.append(name)
            raw = value.payload.get(name)
            mask.append(raw is not None)
            kind = definition.get("type", "number")
            if raw is None:
                values.append(0.0)
            elif kind == "number":
                values.append(float(raw))
            elif kind == "boolean":
                values.append(float(bool(raw)))
            elif kind == "category":
                choices = definition.get("choices", [])
                if raw not in choices:
                    raise DecodeError(f"Unknown category for {name}: {raw}")
                values.append(float(choices.index(raw)) / max(1, len(choices) - 1))
            else:
                raise DecodeError(f"Unsupported field type: {kind}")
        tensor, validity = np.asarray(values, np.float32), np.asarray(mask, bool)
        metadata = {**value.metadata, "field_order": tuple(fields), "raw_values": dict(value.payload)}
        return DecodedModality("structured", value.source_id, tensor, validity, tensor.shape,
                               str(tensor.dtype), value.timestamp, metadata)
