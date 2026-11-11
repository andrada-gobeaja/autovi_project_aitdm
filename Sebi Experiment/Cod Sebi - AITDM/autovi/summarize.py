"""Comparison table of the AutoVI experiments: mean +- std over the seeds.

    python autovi/summarize.py                       # every experiment in outputs/autovi/
    python autovi/summarize.py outputs/autovi/fedavg outputs/autovi/centralised

Reads <experiment>/seed<s>/results.json (written by autovi/evaluate.py) and
writes summary.md / summary.json next to the experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SPRO_LIMITS = ("0.05", "0.3")  # false-positive-rate limits shown in the table


def collect(exp_dir: Path) -> dict | None:
    """{metric: {column: [value of every seed]}} of one experiment."""
    runs = [json.loads(p.read_text()) for p in sorted(exp_dir.glob("seed*/results.json"))]
    if not runs:
        return None
    columns = [*runs[0]["per_category"], "mean"]
    table = {"seeds": len(runs), "columns": columns, "metrics": {}}

    def values(metric: str, column: str, limit: str | None = None):
        out = []
        for run in runs:
            entry = run["mean"] if column == "mean" else run["per_category"][column]
            if limit is None:
                out.append(entry[metric])
            elif "auc_spro" in entry and entry["auc_spro"].get(limit) is not None:
                out.append(entry["auc_spro"][limit])
        return out

    for metric in ("auroc", "ap", "f1_max"):
        table["metrics"][metric] = {c: values(metric, c) for c in columns}
    for limit in SPRO_LIMITS:
        per_column = {c: values("auc_spro", c, limit) for c in columns}
        if all(per_column.values()):
            table["metrics"][f"auc_spro@{limit}"] = per_column
    return table


def cell(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("experiments", type=Path, nargs="*",
                        help="experiment folders (default: all in outputs/autovi/)")
    args = parser.parse_args()
    root = REPO_ROOT / "outputs" / "autovi"
    exp_dirs = args.experiments or sorted(p for p in root.iterdir() if p.is_dir())

    tables = {d.name: t for d in exp_dirs if (t := collect(d.resolve())) is not None}
    if not tables:
        raise SystemExit("No results.json found. Run autovi/evaluate.py first.")
    titles = {"auroc": "Image-level AUROC", "ap": "Image-level average precision",
              "f1_max": "Image-level max. F1 (oracle threshold)",
              "auc_spro@0.05": "Pixel-level AU-sPRO, FPR <= 0.05 (official script)",
              "auc_spro@0.3": "Pixel-level AU-sPRO, FPR <= 0.3 (official script)"}

    # Experiments on different sets of categories get tables of their own (their
    # "mean" columns are not comparable either).
    groups: dict[tuple, dict] = {}
    for name, table in tables.items():
        groups.setdefault(tuple(table["columns"]), {})[name] = table

    lines = []
    for columns, group in groups.items():
        if len(groups) > 1:
            lines += [f"## Categories: {', '.join(columns[:-1])}", ""]
        metrics = [m for m in ("auroc", "auc_spro@0.05", "auc_spro@0.3", "ap", "f1_max")
                   if all(m in t["metrics"] for t in group.values())]
        for metric in metrics:
            lines += [f"### {titles[metric]}", "",
                      "| experiment | seeds | " + " | ".join(columns) + " |",
                      "|---|---|" + "---|" * len(columns)]
            for name, table in group.items():
                lines.append(f"| {name} | {table['seeds']} | "
                             + " | ".join(cell(table["metrics"][metric][c]) for c in columns) + " |")
            lines.append("")
    text = "\n".join(lines)
    print(text)
    out_dir = exp_dirs[0].resolve().parent
    (out_dir / "summary.md").write_text(text)
    (out_dir / "summary.json").write_text(json.dumps(tables, indent=2))
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
