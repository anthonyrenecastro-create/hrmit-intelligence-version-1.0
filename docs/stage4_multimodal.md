# Stage 4 Multimodal Processing

Stage 4 accepts real PNG/JPEG bytes or paths, PCM WAV bytes or paths, and schema-bearing structured mappings. Every input has a source identifier, optional timestamp, and metadata. Decoders preserve file and schema information; encoders compute modality-specific 32-dimensional representations.

Image tensors are decoded as `H × W × C` uint8 RGB. The vision encoder converts pixels to float32 `[0, 1]` internally and derives spatial pooling, color, variance, and gradient features. Audio is decoded to mono float32 `[-1, 1]` and represented using spectral bands, RMS, and zero-crossing features. Structured fields retain schema order, raw values, and a validity mask.

Fusion is a confidence-normalized weighted sum. Its result exposes weights, confidences, missing modalities, pairwise contradictions, source provenance, and diagnostics. `HRMStateProjector` applies a deterministic bounded residual update to the cognitive state; the fusion input never silently replaces the state.

The encoders are deterministic computed baselines, not pretrained foundation models. The benchmark demonstrates functional multimodal complementarity; it does not demonstrate general image, audio, or video understanding. Video remains explicitly experimental and is not registered as a completed modality.
