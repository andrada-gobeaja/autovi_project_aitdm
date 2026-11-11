"""Run the OFFICIAL AutoVI evaluation (Renault / eVantage) on a finished run.

At the end of an AutoVI run, the server writes one anomaly map per test image to
<run>/anomaly_maps/<object>/test/<defect>/<id>.png. This script feeds them to the
official code (github.com/phcarval/autovi_evaluation_code), object by object:

    python experiments/evaluate_autovi.py --run outputs/sim-autovi_group-<date>

It imports the official functions directly, so the official code needs NO changes
(its evaluate_experiment.py computes the image-level AUROC but never writes it).

Metrics:
    AUC-sPRO@x       pixel-level localisation, integrated up to a false-positive rate x
    AUROC all        image level: every defective image vs. every good one
    AUROC mean       image level: one AUROC per defect type, then the mean
                     (the official definition; noisy when a type has 1-2 images)
    image score = maximum of the anomaly map, as in the official code
Results: <run>/autovi_eval/<object>/metrics.json and <run>/autovi_eval/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPRO_LIMITS = ["0.01", "0.05", "0.1", "0.3", "1.0"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="outputs/<run> folder")
    parser.add_argument("--eval-code", type=Path, default=Path.home() / "autovi_evaluation_code",
                        help="clone of github.com/phcarval/autovi_evaluation_code")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data" / "raw" / "autovi",
                        help="the extracted AutoVI dataset (with its ground_truth/ folders)")
    parser.add_argument("--workers", type=int, default=None,
                        help="CPU processes for the sPRO curves (default: none)")
    parser.add_argument("--curve-max-distance", type=float, default=0.001,
                        help="precision of the sPRO curves; keep the official 0.001 for "
                             "reported numbers (larger = faster, for quick checks only)")
    args = parser.parse_args()

    maps_root = args.run / "anomaly_maps"
    if not maps_root.is_dir():
        sys.exit(f"ERROR: {maps_root} not found. Is this an AutoVI run that finished?")
    if not (args.eval_code / "evaluate_experiment.py").exists():
        sys.exit(f"ERROR: official code not found in {args.eval_code}. Clone it:\n"
                 "  git clone https://github.com/phcarval/autovi_evaluation_code.git ~/autovi_evaluation_code")

    sys.path.insert(0, str(args.eval_code.resolve()))
    from evaluate_experiment import (get_auc_spros_for_metrics,  # noqa: E402
                                     get_image_level_detection_metrics,
                                     get_image_level_detection_metrics_per_image,
                                     read_maps, set_resize_shape)
    from src.aggregation import MetricsAggregator  # noqa: E402
    from src.image import DefectsConfig  # noqa: E402
    from src.util import compute_classification_auc_roc  # noqa: E402

    out_root = args.run / "autovi_eval"
    summary = {}
    for obj_dir in sorted(p for p in maps_root.iterdir() if p.is_dir()):
        obj = obj_dir.name
        print(f"\n>>> {obj}: official evaluation (a few minutes for the large objects)")
        with open(args.dataset / obj / "defects_config.json") as f:
            defects_config = DefectsConfig.create_from_list(json.load(f))
        # The official README says to pass the base folder, but the code reads
        # <dir>/<defect>/<id>.png, so it needs <maps>/<object>/test
        gt_maps, anomaly_maps = read_maps(
            gt_dir=str(args.dataset / obj / "ground_truth"),
            anomaly_maps_test_dir=str(obj_dir / "test"),
            defects_config=defects_config,
            resize_shape=set_resize_shape(obj))

        metrics = MetricsAggregator(gt_maps=gt_maps, anomaly_maps=anomaly_maps,
                                    parallel_workers=args.workers,
                                    parallel_niceness=19).run(
                                        curve_max_distance=args.curve_max_distance)
        auc_spro = {str(k): float(v) for k, v in get_auc_spros_for_metrics(metrics).items()}
        per_type = get_image_level_detection_metrics(gt_maps, anomaly_maps)["auc_roc"]
        per_image = get_image_level_detection_metrics_per_image(gt_maps, anomaly_maps)
        good = [r["anomaly_scores"]["max"] for r in per_image if r["image_type"] == "good"]
        bad = [r["anomaly_scores"]["max"] for r in per_image if r["image_type"] != "good"]

        result = {
            "localization": {"auc_spro": auc_spro},
            "classification": {
                "auc_roc_all_defects": float(compute_classification_auc_roc(good, bad)),
                "auc_roc_mean_over_types": float(per_type["mean"]),
                "auc_roc_per_defect_type": {k: float(v) for k, v in per_type.items()
                                            if k != "mean"},
            },
        }
        (out_root / obj).mkdir(parents=True, exist_ok=True)
        (out_root / obj / "metrics.json").write_text(json.dumps(result, indent=2))
        summary[obj] = result

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    header = (["Object", "AUROC all", "AUROC mean"]
              + [f"AUC-sPRO@{x}" for x in SPRO_LIMITS])
    print("\n| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    for obj, r in summary.items():
        c, s = r["classification"], r["localization"]["auc_spro"]
        row = [obj, f"{c['auc_roc_all_defects']:.3f}", f"{c['auc_roc_mean_over_types']:.3f}"]
        print("| " + " | ".join(row + [f"{s[x]:.3f}" for x in SPRO_LIMITS]) + " |")
    print(f"\nSaved to {out_root}")


if __name__ == "__main__":
    main()
