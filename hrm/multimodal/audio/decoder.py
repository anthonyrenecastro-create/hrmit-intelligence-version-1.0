from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np

from hrm.multimodal.types import DecodedModality, ModalityInput


@dataclass(frozen=True)
class AudioDecoder:
    max_duration_seconds: float = 10.0
    supported_channels: tuple[int, ...] = (1, 2)
    supported_sample_rates: tuple[int, ...] = (8000, 16000, 22050, 32000, 44100)

    def decode(self, source: Any, source_id: str, timestamp: float | None = None) -> DecodedModality:
        if isinstance(source, (bytes, bytearray)):
            buffer = io.BytesIO(source)
        elif isinstance(source, str):
            buffer = open(source, "rb")
        else:
            raise ValueError("Audio source must be bytes or file path")

        try:
            with wave.open(buffer, "rb") as wav:
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                frames = wav.getnframes()
                duration = frames / float(sample_rate)
                if sample_rate not in self.supported_sample_rates:
                    raise ValueError(f"Unsupported sample rate: {sample_rate}")
                if channels not in self.supported_channels:
                    raise ValueError(f"Unsupported channel count: {channels}")
                if duration > self.max_duration_seconds:
                    raise ValueError("Audio exceeds maximum duration")
                raw = wav.readframes(frames)
        except wave.Error as error:
            raise ValueError("Malformed WAV audio") from error

        dtype = np.int16 if sample_width == 2 else np.uint8
        waveform = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if channels == 2:
            waveform = waveform.reshape(-1, 2).mean(axis=1)
        metadata = {
            "sample_rate": sample_rate,
            "channels": channels,
            "duration": duration,
            "sample_count": int(frames),
            "dtype": str(dtype),
        }
        return DecodedModality(
            modality="audio",
            source_id=source_id,
            tensor=waveform,
            mask=None,
            shape=waveform.shape,
            dtype=str(waveform.dtype),
            timestamp=timestamp,
            metadata=metadata,
        )
