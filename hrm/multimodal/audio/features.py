from __future__ import annotations

import numpy as np
from scipy.signal import stft

from hrm.multimodal.types import DecodedModality


class AudioFeatures:
    def __init__(self, n_fft: int = 256, hop_length: int = 128, n_mels: int = 40, fmin: float = 20.0, fmax: float | None = None) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def log_mel_spectrogram(self, decoded: DecodedModality) -> DecodedModality:
        waveform = np.asarray(decoded.tensor, dtype=np.float32).ravel()
        sample_rate = int(decoded.metadata.get("sample_rate", 16000))
        fmax = self.fmax or sample_rate / 2.0
        _, _, stft_matrix = stft(
            waveform,
            fs=sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            boundary=None,
            padded=False,
        )
        magnitude = np.abs(stft_matrix)
        mel_basis = self._mel_filterbank(sample_rate, self.n_fft, self.n_mels, self.fmin, fmax)
        mel_spec = np.matmul(mel_basis, magnitude)
        log_mel = np.log1p(mel_spec)
        return DecodedModality(
            modality=decoded.modality,
            source_id=decoded.source_id,
            tensor=log_mel.astype(np.float32),
            mask=decoded.mask,
            shape=log_mel.shape,
            dtype=str(log_mel.dtype),
            timestamp=decoded.timestamp,
            metadata={
                **decoded.metadata,
                "feature_type": "log_mel_spectrogram",
                "n_mels": self.n_mels,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "fmin": self.fmin,
                "fmax": fmax,
            },
        )

    def _mel_filterbank(self, sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
        def hz_to_mel(hz: float) -> float:
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel: float) -> float:
            return 700.0 * (10 ** (mel / 2595.0) - 1.0)

        mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_frequencies = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
        filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

        for m in range(1, n_mels + 1):
            lower = hz_points[m - 1]
            center = hz_points[m]
            upper = hz_points[m + 1]
            lower_bin = np.searchsorted(bin_frequencies, lower)
            center_bin = np.searchsorted(bin_frequencies, center)
            upper_bin = np.searchsorted(bin_frequencies, upper)
            if lower_bin == center_bin:
                center_bin = lower_bin + 1
            if center_bin == upper_bin:
                upper_bin = center_bin + 1
            for i in range(lower_bin, center_bin):
                filterbank[m - 1, i] = (bin_frequencies[i] - lower) / max(center - lower, 1e-6)
            for i in range(center_bin, upper_bin):
                filterbank[m - 1, i] = (upper - bin_frequencies[i]) / max(upper - center, 1e-6)
        return filterbank
