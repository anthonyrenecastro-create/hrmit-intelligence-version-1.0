from __future__ import annotations

import numpy as np
from scipy.signal import stft

from hrm.multimodal.types import DecodedModality


class AudioFeatures:
    def __init__(self, n_fft: int = 256, hop_length: int = 128, n_mels: int = 40) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def log_mel_spectrogram(self, decoded: DecodedModality) -> DecodedModality:
        waveform = np.asarray(decoded.tensor, dtype=np.float32)
        _, _, stft_matrix = stft(waveform, nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length, padded=False)
        magnitude = np.abs(stft_matrix)
        mel = np.log1p(magnitude[: self.n_mels, :])
        return DecodedModality(
            modality=decoded.modality,
            source_id=decoded.source_id,
            tensor=mel.astype(np.float32),
            mask=decoded.mask,
            shape=mel.shape,
            dtype=str(mel.dtype),
            timestamp=decoded.timestamp,
            metadata={**decoded.metadata, "feature_type": "log_mel_spectrogram", "n_mels": self.n_mels},
        )
