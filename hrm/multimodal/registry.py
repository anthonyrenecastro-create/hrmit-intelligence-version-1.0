from __future__ import annotations

from typing import Any

from hrm.multimodal.types import ModalityInput


class ModalityRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, Any] = {}

    def register(self, modality: str, adapter: Any) -> None:
        self._adapters[modality] = adapter

    def get(self, modality: str) -> Any:
        if modality not in self._adapters:
            raise ValueError(f"Unsupported modality: {modality}")
        return self._adapters[modality]

    def available_modalities(self) -> tuple[str, ...]:
        return tuple(self._adapters.keys())

    def adapt(self, input_data: ModalityInput) -> Any:
        adapter = self.get(input_data.modality)
        return adapter.decode(input_data)
