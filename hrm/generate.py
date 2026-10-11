"""Unified HRM content generation entrypoint.

Ties the stages together for end-to-end generation:
  Stage 1 (HRM core) -> Stage 4 (vision) -> Stage 2 (text description)

Usage:
    python3 hrm/generate.py --mode image --seed 42 --output ./out
    python3 hrm/generate.py --mode text --prompt "The HRM" --output ./out
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate content with the HRM.")
    parser.add_argument("--mode", choices=["image", "text"], default="image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="The HRM")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--steps", type=int, default=48, help="HRM dynamics steps for image gen")
    parser.add_argument("--output", type=Path, default=Path("hrm_generated"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "image":
        from hrm.generation.vision_generator import HRMImageGenerator
        gen = HRMImageGenerator(seed=args.seed, out_size=128)
        img = gen.generate(seed=args.seed, steps=args.steps, creativity=0.15)
        out_path = args.output / f"hrm_image_{args.seed}.png"
        gen.save_image(img, out_path)
        print(f"Image saved: {out_path} ({img.shape[1]}x{img.shape[0]})")
    elif args.mode == "text":
        from hrm.generation.text_generator import HRMTextGenerator
        gen = HRMTextGenerator(seed=args.seed)
        # Try loading trained weights
        weights = args.output / "hrm_textgen.npz"
        if weights.exists():
            gen.load(args.output)
            print(f"Loaded trained weights from {weights}")
        text = gen.generate(prompt=args.prompt, max_tokens=args.max_tokens,
                            temperature=args.temperature, seed=args.seed)
        out_path = args.output / f"hrm_text_{args.seed}.txt"
        out_path.write_text(text)
        print(f"Text saved: {out_path}")
        print("---")
        print(text[:500])


if __name__ == "__main__":
    main()
