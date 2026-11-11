"""One table comparing AutoVI runs (e.g. the splits of scripts/run_autovi_splits.sh).

    python experiments/summarize_autovi.py outputs/splits/<tag>

Reads, for every run folder inside: metrics.json (written by the Flower server)
and autovi_eval/summary.json (official evaluation, experiments/evaluate_autovi.py).
Prints a Markdown table (paste it in the report) and writes summary.csv.

Columns:
  AUROC <object>   official image-level AUROC, all defects vs good
  AUROC mean       mean of the per-object AUROCs (main number)
  sPRO@0.05/0.3    official localisation, mean over objects
  F1               at the threshold the clients agreed on (no test labels used)
  val loss         clients' final validation loss (good images only)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_run(run: Path) -> dict | None:
    metrics_file = run / "metrics.json"
    if not metrics_file.exists():
        return None
    metrics = json.loads(metrics_file.read_text())
    cfg = metrics.get("run_config", {})
    meta_file = Path(str(cfg.get("partitions-dir", ""))) / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    settings = meta.get("settings", {})

    split = settings.get("strategy", "?")
    if split in ("dirichlet", "quantity"):
        split += f" a={settings.get('alpha')}"
    if settings.get("feature_skew"):
        split += " + camera skew"
    if meta.get("num_clients") == 1:
        split = "centralised (all data, 1 client)"

    evals = metrics.get("evaluate_metrics_clientapp", {})
    last = evals[max(evals, key=int)] if evals else {}
    test = metrics.get("anomaly_detection_test", {})
    row = {
        "run": run.name,
        "split": split,
        "clients": meta.get("num_clients", "?"),
        "f1": test.get("f1"),
        "val_loss": last.get("eval_loss"),
    }

    official_file = run / "autovi_eval" / "summary.json"
    objects = meta.get("objects", [])
    if official_file.exists():
        official = json.loads(official_file.read_text())
        objects = list(official)
        aurocs = [official[o]["classification"]["auc_roc_all_defects"] for o in objects]
        for o, a in zip(objects, aurocs):
            row[f"auroc_{o}"] = a
        row["auroc_mean"] = sum(aurocs) / len(aurocs)
        for x in ("0.05", "0.3"):
            vals = [official[o]["localization"]["auc_spro"][x] for o in objects]
            row[f"spro_{x}"] = sum(vals) / len(vals)
        row["source"] = "official"
    else:  # not evaluated yet: the server's own AUROC (same definition)
        for o in objects:
            if f"auroc_{o}" in test:
                row[f"auroc_{o}"] = test[f"auroc_{o}"]
        row["auroc_mean"] = test.get("auroc")
        row["source"] = "server only"
    row["objects"] = objects
    return row


def fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder that contains the run folders")
    parser.add_argument("--order", nargs="*", help="run names in the order to show them")
    args = parser.parse_args()

    runs = [r for r in (load_run(p) for p in sorted(args.folder.iterdir()) if p.is_dir()) if r]
    if not runs:
        raise SystemExit(f"No finished runs (metrics.json) in {args.folder}")
    if args.order:
        rank = {name: i for i, name in enumerate(args.order)}
        runs.sort(key=lambda r: rank.get(r["run"], len(rank)))

    objects = []
    for r in runs:
        objects += [o for o in r["objects"] if o not in objects]
    header = (["Run", "Split", "Clients", "AUROC mean"] + [f"AUROC {o}" for o in objects]
              + ["sPRO@0.05", "sPRO@0.3", "F1", "val loss"])
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in runs:
        cells = ([r["run"], r["split"], str(r["clients"]), fmt(r.get("auroc_mean"))]
                 + [fmt(r.get(f"auroc_{o}")) for o in objects]
                 + [fmt(r.get("spro_0.05")), fmt(r.get("spro_0.3")),
                    fmt(r.get("f1")), fmt(r.get("val_loss"))])
        print("| " + " | ".join(cells) + " |")
    pending = [r["run"] for r in runs if r["source"] != "official"]
    if pending:
        print(f"\n(no official evaluation yet for: {', '.join(pending)}; AUROC from the server)")

    keys = (["run", "split", "clients", "auroc_mean"] + [f"auroc_{o}" for o in objects]
            + ["spro_0.05", "spro_0.3", "f1", "val_loss", "source"])
    with open(args.folder / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    print(f"\nSaved {args.folder / 'summary.csv'}")


if __name__ == "__main__":
    main()
