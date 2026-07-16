# HRM Intelligence Framework Integration

This repository now includes a foundational integration of the TPU HRM baseline notebook into a structured HRM Theory pipeline.

Files added/updated:
- `hrm/baseline.py`: import-safe, module-based version of the TPU-Parallel HRM baseline.
- `hrm/theory.py`: a staged HRM Theory integration scaffold with Stage 1 execution and placeholder definitions for Stages 2-6.
- `hrm/runner.py`: command-line entrypoint for running HRM stages.
- `hrm/__init__.py`: package exports.

## Usage

Run the baseline stage:

```bash
python3 hrm/runner.py --stage 1 --output hrm_baseline_outputs --seed 0
```

> Note: `hrm/runner.py` handles package path resolution automatically, so direct invocation should work without setting `PYTHONPATH`.
> If JAX is unavailable, Stage 1 will fall back to a placeholder baseline artifact.

Run Stage 2 with planning:

```bash
python3 hrm/runner.py --stage 2 --plan-query "Analyze baseline recovery and propose improvements"
```

Run Stage 3 tool verification:

```bash
python3 hrm/runner.py --stage 3
```

Run Stage 4 multimodal perception:

```bash
python3 hrm/runner.py --stage 4 --modality-query "Inspect sensory inputs" --modalities "vision,audio,structured"
```

Run Stage 5 distributed cognition:

```bash
python3 hrm/runner.py --stage 5 --consensus-query "Coordinate distributed HRM reasoning" --agent-roles "safety,efficiency,planning,recovery"
```

Run Stage 6 learning systems:

```bash
python3 hrm/runner.py --stage 6 --learning-rate 0.05 --starting-preferences '{"exploration": 0.2, "safety": 0.3, "efficiency": 0.5, "bias": 0.0}'
```

## Stage structure

Stage 1: HRM reasoning core + basic memory
Stage 2: Long-term memory, knowledge retrieval, planning
Stage 3: Tool use, code execution, external APIs, self verification
Stage 4: Multimodal perception (text, images, audio, video)
Stage 5: Distributed cognition (multiple reasoning agents)
Stage 6: Learning systems (continual adaptation, preference optimization, memory refinement)

## Next steps

- Expand Stage 2 with a retrieval index and planning layer.
- Add tool adapters and verification loops in Stage 3.
- Extend the empirical Stage 4 vision/audio/structured baseline; video remains experimental.
- Build multi-agent coordination in Stage 5.
- Add learning signal capture and memory refinement in Stage 6.


## Empirical completion boundaries

Stage 4 now decodes real PNG/JPEG and PCM WAV inputs, preserves structured schemas, performs confidence-aware fusion, and applies a bounded projection into HRM cognitive state. Its deterministic benchmark demonstrates multimodal complementarity. This is not a claim of general visual, audio, or video understanding.

Stage 6 now includes immutable feedback and experience records, persistence and replay, bounded candidate parameter updates, held-out and regression gates, promotion, rollback, calibration metrics, and provenance. The legacy deterministic preference update remains available only as a compatibility baseline and carries `completion_claim: false`.
