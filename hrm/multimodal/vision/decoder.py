from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from hrm.multimodal.types import DecodedModality, ModalityInput

MAX_IMAGE_PIXELS = 1280 * 1280
SUPPORTED_FORMATS = {"PNG", "JPEG", "JPG"}
SUPPORTED_MODES = {"RGB", "L", "RGBA", "LA", "P"}


@dataclass(frozen=True)
class ImageDecoder:
    max_pixels: int = MAX_IMAGE_PIXELS
    supported_formats: frozenset[str] = frozenset(SUPPORTED_FORMATS)

    def decode(self, source: Any, source_id: str, timestamp: float | None = None) -> DecodedModality:
        raw_bytes = self._read_source(source)
        image = self._open_image(raw_bytes)
        if image.format is None or image.format.upper() not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {image.format}")
        if image.mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported image mode: {image.mode}")

        original_mode = image.mode
        if image.mode in {"RGBA", "LA", "P"}:
            image = image.convert("RGB")
        width, height = image.size
        if width * height > self.max_pixels:
            raise ValueError("Image exceeds safe maximum pixel count")

        tensor = np.asarray(image, dtype=np.float32)
        if tensor.ndim == 2:
            tensor = tensor[..., None]
        metadata = {
            "format": image.format,
            "original_mode": original_mode,
            "converted_mode": image.mode,
            "width": width,
            "height": height,
            "pixel_count": width * height,
        }
        return DecodedModality(
            modality="vision",
            source_id=source_id,
            tensor=tensor,
            mask=None,
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            timestamp=timestamp,
            metadata=metadata,
        )

    def _read_source(self, source: Any) -> bytes:
        if isinstance(source, (bytes, bytearray)):
            return bytes(source)
        if isinstance(source, str):
            with open(source, "rb") as handle:
                return handle.read()
        raise ValueError("Image source must be bytes or file path")

    def _open_image(self, raw_bytes: bytes) -> Image.Image:
        try:
            Image.MAX_IMAGE_PIXELS = self.max_pixels
            with Image.open(io.BytesIO(raw_bytes)) as image:
                image_format = image.format
                image.load()
                image_copy = image.copy()
                image_copy.format = image_format
                return image_copy
        except UnidentifiedImageError as error:
            raise ValueError("Malformed image bytes") from error
        except Exception as error:
            raise ValueError("Unable to decode image") from error
