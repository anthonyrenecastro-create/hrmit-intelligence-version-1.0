from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
    )


def _write_gate_status_svg(report: dict[str, Any], out_path: Path) -> None:
    checks: dict[str, bool] = report.get("checks", {})
    gate_names = [
        "G1_topology_mutation_ok",
        "G2_transport_map_ok",
        "ablation_integrity_ok",
        "structural_ledger_gate_ok",
    ]
    width = 920
    row_h = 52
    top = 56
    height = top + row_h * len(gate_names) + 30
    x_label = 24
    x_bar = 350
    bar_w = 520
    bar_h = 28

    lines: list[str] = [_svg_header(width, height)]
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#fcfcfd"/>\n')
    lines.append('<text x="24" y="34" font-size="22" font-family="DejaVu Sans, sans-serif" fill="#222">Validation Gates</text>\n')

    for i, name in enumerate(gate_names):
        ok = bool(checks.get(name, False))
        y = top + i * row_h
        fill = "#1f9d55" if ok else "#d64545"
        label = "PASS" if ok else "FAIL"
        lines.append(
            f'<text x="{x_label}" y="{y + 21}" font-size="16" font-family="DejaVu Sans Mono, monospace" fill="#333">{name}</text>\n'
        )
        lines.append(
            f'<rect x="{x_bar}" y="{y}" width="{bar_w}" height="{bar_h}" rx="6" fill="#eceff3"/>\n'
        )
        lines.append(
            f'<rect x="{x_bar}" y="{y}" width="{bar_w if ok else int(bar_w * 0.35)}" height="{bar_h}" rx="6" fill="{fill}"/>\n'
        )
        lines.append(
            f'<text x="{x_bar + bar_w - 56}" y="{y + 20}" font-size="14" font-family="DejaVu Sans, sans-serif" fill="#fff">{label}</text>\n'
        )

    lines.append("</svg>\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def _ring_points(cx: float, cy: float, r: float, n: int) -> list[tuple[float, float]]:
    import math

    if n <= 0:
        return []
    points: list[tuple[float, float]] = []
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        points.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    return points


def _write_topology_svg(report: dict[str, Any], out_path: Path) -> None:
    probe = report.get("topology_probe", {})
    before_nodes = int(probe.get("before_nodes", 0))
    after_nodes = int(probe.get("after_nodes", 0))
    before_version = int(probe.get("before_topology_version", 0))
    after_version = int(probe.get("after_topology_version", 0))
    events = probe.get("accepted_topology_events", [])

    width, height = 1020, 420
    lines: list[str] = [_svg_header(width, height)]
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>\n')
    lines.append('<text x="24" y="36" font-size="24" font-family="DejaVu Sans, sans-serif" fill="#222">Topology Before/After</text>\n')

    left_cx, right_cx, cy = 280.0, 740.0, 210.0
    radius = 120.0
    before_pts = _ring_points(left_cx, cy, radius, max(before_nodes, 1))
    after_pts = _ring_points(right_cx, cy, radius, max(after_nodes, 1))

    lines.append(f'<circle cx="{left_cx}" cy="{cy}" r="{radius}" fill="none" stroke="#6b7280" stroke-width="2"/>\n')
    lines.append(f'<circle cx="{right_cx}" cy="{cy}" r="{radius}" fill="none" stroke="#6b7280" stroke-width="2"/>\n')

    for x, y in before_pts:
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="#3b82f6"/>\n')

    for idx, (x, y) in enumerate(after_pts):
        fill = "#f97316" if idx >= before_nodes else "#10b981"
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{fill}"/>\n')

    lines.append(
        f'<text x="{left_cx - 100}" y="72" font-size="16" font-family="DejaVu Sans Mono, monospace" fill="#333">before: nodes={before_nodes}, v={before_version}</text>\n'
    )
    lines.append(
        f'<text x="{right_cx - 100}" y="72" font-size="16" font-family="DejaVu Sans Mono, monospace" fill="#333">after: nodes={after_nodes}, v={after_version}</text>\n'
    )
    lines.append(
        '<line x1="420" y1="210" x2="600" y2="210" stroke="#111827" stroke-width="2" marker-end="url(#arrow)"/>\n'
    )
    lines.append(
        f'<text x="430" y="196" font-size="14" font-family="DejaVu Sans, sans-serif" fill="#111827">events: {", ".join(events) if events else "none"}</text>\n'
    )

    lines.insert(
        1,
        '<defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#111827"/></marker></defs>\n',
    )
    lines.append("</svg>\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def _write_transport_table(report: dict[str, Any], out_path: Path) -> None:
    probe = report.get("topology_probe", {})
    mapping = probe.get("phi_transport_new_to_old") or []
    rows = ["# Transport Map Table", "", "| new_index | old_index |", "|---:|:---|"]
    for new_idx, old_idx in enumerate(mapping):
        rows.append(f"| {new_idx} | {old_idx if old_idx is not None else 'introduced'} |")
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_pack_readme(report: dict[str, Any], out_path: Path) -> None:
    run_id = report.get("run_id", "unknown-run")
    checks = report.get("checks", {})
    lines = [
        f"# Compact Visualization Pack ({run_id})",
        "",
        "Generated artifacts:",
        "- gate_status.svg",
        "- topology_before_after.svg",
        "- transport_map_table.md",
        "",
        "Gate summary:",
    ]
    for key in [
        "G1_topology_mutation_ok",
        "G2_transport_map_ok",
        "ablation_integrity_ok",
        "structural_ledger_gate_ok",
    ]:
        lines.append(f"- {key}: {checks.get(key)}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_pack(report_path: Path, output_dir: Path | None) -> Path:
    report = _load_report(report_path)
    if output_dir is None:
        artifact_root = Path(report.get("artifact_root", "validation/artifacts"))
        output_dir = artifact_root / "visualization_pack"
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_gate_status_svg(report, output_dir / "gate_status.svg")
    _write_topology_svg(report, output_dir / "topology_before_after.svg")
    _write_transport_table(report, output_dir / "transport_map_table.md")
    _write_pack_readme(report, output_dir / "README.md")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate compact validation visualization pack.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("validation/FINAL_VALIDATION_REPORT.json"),
        help="Path to validation report JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <artifact_root>/visualization_pack.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = generate_pack(args.report, args.output_dir)
    print(json.dumps({"status": "ok", "output_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()