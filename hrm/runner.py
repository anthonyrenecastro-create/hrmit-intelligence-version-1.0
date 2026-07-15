from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ is None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))

from hrm.theory import HRMTheory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HRM Theory integration pipeline.")
    parser.add_argument("--stage", type=int, default=1, help="Stage number to run (1-6)")
    parser.add_argument("--output", type=Path, default=Path("hrm_baseline_outputs"), help="Output directory for stage artifacts")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the Stage 1 baseline run")
    parser.add_argument("--plan-query", type=str, default="Improve HRM safety and recovery", help="Query for Stage 2 planning")
    parser.add_argument("--api-endpoint", type=str, default="status", help="Endpoint name for Stage 3 API verification")
    parser.add_argument("--api-payload", type=str, default='{"health": true}', help="JSON payload for Stage 3 API verification")
    parser.add_argument("--modality-query", type=str, default="Text", help="Optional text description for Stage 4 perception")
    parser.add_argument("--modalities", type=str, default="text,image,audio,video", help="Comma-separated modalities to include in Stage 4")
    parser.add_argument("--consensus-query", type=str, default="Coordinate distributed HRM reasoning and planning", help="Query used by Stage 5 agent coordination")
    parser.add_argument("--agent-roles", type=str, default="safety,efficiency,planning,recovery", help="Comma-separated agent roles for Stage 5")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for Stage 6 preference adaptation")
    parser.add_argument("--starting-preferences", type=str, default='{"exploration": 0.2, "safety": 0.3, "efficiency": 0.5, "bias": 0.0}', help="JSON string for Stage 6 starting preference weights")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    theory = HRMTheory()
    if args.stage == 1:
        result = theory.run_stage(1, seed=args.seed, output_dir=args.output)
    elif args.stage == 2:
        result = theory.run_stage(2, baseline_record=None, plan_query=args.plan_query)
    elif args.stage == 3:
        result = theory.run_stage(
            3,
            baseline_record=None,
            api_endpoint=args.api_endpoint,
            api_payload=json.loads(args.api_payload),
        )
    elif args.stage == 4:
        modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
        result = theory.run_stage(
            4,
            modality_query=args.modality_query,
            include_modalities=modalities,
        )
    elif args.stage == 5:
        agent_roles = [item.strip() for item in args.agent_roles.split(",") if item.strip()]
        result = theory.run_stage(
            5,
            baseline_record=None,
            consensus_query=args.consensus_query,
            agent_roles=agent_roles,
        )
    elif args.stage == 6:
        try:
            starting_preferences = json.loads(args.starting_preferences)
        except ValueError:
            starting_preferences = {"exploration": 0.2, "safety": 0.3, "efficiency": 0.5, "bias": 0.0}
        result = theory.run_stage(
            6,
            baseline_record=None,
            learning_rate=args.learning_rate,
            starting_preferences=starting_preferences,
        )
    else:
        result = theory.run_stage(args.stage)

    print("HRM Theory stage result:")
    print(result)


if __name__ == "__main__":
    main()
