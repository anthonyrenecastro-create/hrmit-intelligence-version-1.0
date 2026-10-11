"""Character <-> HRM field codec.

Maps text to/from the HRM's (height, width, channels) field representation.
Each character becomes a field frame; the HRM processes sequences of frames.
"""
from __future__ import annotations

import numpy as np


class CharFieldCodec:
    """Byte-level codec between text and HRM field frames."""

    def __init__(self, height: int = 8, width: int = 8, channels: int = 4):
        self.height = height
        self.width = width
        self.channels = channels
        # 256 byte values + 1 EOS token
        self.vocab_size = 257
        self.eos_id = 256
        # Fixed random projection: byte -> field pattern (deterministic)
        rng = np.random.RandomState(777)
        self.byte_to_field = rng.randn(256, height, width, channels).astype(np.float32) * 0.3

    def encode(self, text: str) -> np.ndarray:
        """Text -> (seq_len, height, width, channels) field frames."""
        data = text.encode("utf-8", errors="replace")
        frames = np.stack([self.byte_to_field[b] for b in data], axis=0)
        return frames.astype(np.float32)

    def encode_ids(self, text: str) -> list[int]:
        data = text.encode("utf-8", errors="replace")
        return list(data)

    def decode_ids(self, ids: list[int]) -> str:
        filtered = [i for i in ids if 0 <= i < 256]
        return bytes(filtered).decode("utf-8", errors="replace")
