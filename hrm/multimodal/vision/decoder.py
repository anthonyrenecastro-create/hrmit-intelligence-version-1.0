from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from hrm.multimodal.types import DecodedModality, ModalityInput

MAX_IMAGE_PIXELS = 1280 * 1280
SUPPORTED_FORMATS = {"PNG", "JPEG", "JPG"}


@dataclass(frozen=True)
class ImageDecoder:
    def decode(self, source: Any, source_id: str, timestamp: float | None = None) -> DecodedModality:
        image = self._open_image(source)
        mode = image.mode
        if mode not in {"RGB", "L"}:
            if mode == "RGBA":
                image = image.convert("RGB")
            else:
                raise ValueError(f"Unsupported image mode: {mode}")
        if image.format is None or image.format.upper() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {image.format}")
        width, height = image.size
        metadata = {
            "format": image.format,
            "mode": image.mode,
            "width": width,
            "height": height,
        }
        tensor = np.asarray(image, dtype=np.float32)
        if tensor.ndim == 2:
            tensor = tensor[..., None]
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

    def _open_image(self, source: Any) -> Image.Image:
        if isinstance(source, (bytes, bytearray)):
            try:
                image = Image.open(io.BytesIO(source))
            except Exception as error:
                raise ValueError("Malformed image bytes") from error
        elif isinstance(source, str):
            try:
                image = Image.open(source)
            except Exception as error:
                raise ValueError("Unable to open image file") from error
        else:
            raise ValueError("Image source must be bytes or file path")
        image.load()
        image.verify()
        with Image.open(io.BytesIO(source if isinstance(source, (bytes, bytearray)) else open(source, "rb").read())) as verified:
            if verified.format and verified.size[0] * verified.size[1] > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds safe maximum pixel count")
        return Image.open(io.BytesIO(source if isinstance(source, (bytes, bytearray)) else open(source, "rb").read()))
