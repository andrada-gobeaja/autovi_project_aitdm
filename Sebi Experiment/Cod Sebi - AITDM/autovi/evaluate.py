"""Shared evaluation of a finished run: ONE function for the federated runs and
for the centralised / local-only baselines, so all are measured the same way.

    python autovi/evaluate.py --run outputs/autovi/fedavg/seed1 \\
        --partitions data/partitions/autovi_384

For every category of the test set it computes

* image level (is the image defective?): AUROC, average precision, max. F1
* pixel level (where is the defect?): AU-sPRO at several false-positive limits,
  with the OFFICIAL AutoVI evaluation code (github.com/phcarval/autovi_evaluation_code).
  That code reads one anomaly map per test image from
  <maps>/<category>/test/<defect>/<image>.png, which this script writes first.

and stores results.json (+ example heat maps) in the run folder.

A run folder contains either final_model.pt (one model for all categories:
federated, centralised) or final_model_<category>.pt (local-only training).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from autovi_fl.task import (
    detection_metrics,
    get_device,
    iter_anomaly_maps,
    load_meta,
    load_student,
    load_test_data,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Layout created by autovi/wsl_sync.sh; override with the environment variables
DATASET_ROOT = Path(os.environ.get("AUTOVI_DATASET_ROOT", REPO_ROOT.parent / "autovi"))
EVAL_CODE = Path(os.environ.get("AUTOVI_EVAL_CODE", REPO_ROOT.parent / "autovi_evaluation_code"))
# Pixels (test images x width x height) the official script may hold at once; the
# default fits 15 GB of RAM (WSL2 on a 32 GB machine). Raise it with more memory.
MEMORY_BUDGET = float(os.environ.get("AUTOVI_EVAL_PIXELS", 650e6))


def student_of_run(run_dir: Path, category: str) -> torch.nn.Module:
    per_category = run_dir / f"final_model_{category}.pt"
    return load_student(per_category if per_category.exists() else run_dir / "final_model.pt")


def predict_maps(student, x: np.ndarray, device) -> np.ndarray:
    """(N, S, S) float32 anomaly maps at the model's input resolution."""
    return np.concatenate([m.float().cpu().numpy() for m in iter_anomaly_maps(student, x, device)])


def write_anomaly_maps(maps: np.ndarray, defects, names, original_size, out_dir: Path) -> None:
    """Store the maps as 16-bit PNGs at the ORIGINAL image size, as the official
    evaluation expects. All maps of a category share one scale (the metrics
    compare scores across images)."""
    width, height = original_size
    scale = 65535.0 / max(float(maps.max()), 1e-12)
    for defect in set(defects):
        (out_dir / defect).mkdir(parents=True, exist_ok=True)

    def save(i: int) -> None:
        m = torch.from_numpy(maps[i])[None, None]
        m = F.interpolate(m, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
        array = (m.numpy() * scale).clip(0, 65535).astype(np.uint16)
        Image.fromarray(array).save(out_dir / defects[i] / f"{names[i]}.png", compress_level=1)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(save, range(len(maps))))


def official_auc_spro(category: str, maps_test_dir: Path, out_dir: Path, workers: int,
                      timeout: float = 3600) -> dict:
    """Run the official evaluation script and return {fpr limit: AU-sPRO}."""
    command = [sys.executable, "evaluate_experiment.py",
               "--object_name", category,
               "--dataset_base_dir", str(DATASET_ROOT),
               "--anomaly_maps_dir", str(maps_test_dir),
               "--output_dir", str(out_dir)]
    if workers > 1:
        command += ["--num_parallel_workers", str(workers)]
    log = out_dir.parent / f"{category}.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / "metrics.json"
    result_file.unlink(missing_ok=True)
    # The official script never shuts its worker pools down, and now and then it
    # does not exit after it has written its results (seen: 35 minutes asleep).
    # So do not wait for the exit: once metrics.json is there, give the script
    # 30 s to leave, then stop it and its workers.
    with open(log, "w") as out:
        process = subprocess.Popen(command, cwd=EVAL_CODE, stdout=out, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        start, written_at = time.time(), None
        while process.poll() is None:
            time.sleep(2)
            if written_at is None and result_file.exists():
                written_at = time.time()
            hung = written_at is not None and time.time() - written_at > 30
            if hung or time.time() - start > timeout:
                os.killpg(process.pid, signal.SIGKILL)  # own session: group id = pid
                process.wait()
    try:
        return json.loads(result_file.read_text())["localization"]["auc_spro"]
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(f"Official evaluation failed for {category}, see {log}") from error


def save_examples(x, y, maps, category: str, path: Path, n: int) -> None:
    """Heat maps of the first n defective and n good test images (for the report)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = [*np.flatnonzero(y == 1)[:n], *np.flatnonzero(y == 0)[:n]]
    vmax = float(np.quantile(maps, 0.999))
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.4), squeeze=False)
    for col, i in enumerate(picks):
        axes[0, col].imshow(x[i])
        axes[0, col].set_title(f"{'defective' if y[i] else 'good'}\nscore {maps[i].max():.3f}",
                               fontsize=9)
        axes[1, col].imshow(x[i])
        axes[1, col].imshow(maps[i], cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{category}: test images (top) and anomaly maps (bottom)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate_run(run_dir: Path, partitions_dir: Path, spro: bool = True, keep_maps: bool = False,
                 workers: int = 0, examples: int = 4) -> dict:
    """Evaluate the model(s) in run_dir and write run_dir/results.json."""
    meta = load_meta(partitions_dir)
    test = load_test_data(partitions_dir)
    device = get_device()
    if spro and not ((EVAL_CODE / "evaluate_experiment.py").exists() and DATASET_ROOT.is_dir()):
        print(f"WARNING: official evaluation code ({EVAL_CODE}) or dataset ({DATASET_ROOT}) "
              "not found: AU-sPRO is skipped. Set AUTOVI_EVAL_CODE / AUTOVI_DATASET_ROOT.")
        spro = False

    maps_root = run_dir / "anomaly_maps"
    categories = meta["categories"]
    if workers <= 0:  # share the CPUs between the categories that run in parallel
        workers = max(2, (os.cpu_count() or 4) // min(len(categories), 3))

    # --- 1. GPU, seconds: anomaly maps and image-level metrics ------------------
    per_category, maps_of, masks = {}, {}, {}
    for index, category in enumerate(categories):
        masks[category] = mask = test["category"] == index
        x, y = test["x"][mask], test["y"][mask]
        maps_of[category] = maps = predict_maps(student_of_run(run_dir, category), x, device)
        per_category[category] = {"good": int((y == 0).sum()), "defective": int((y == 1).sum()),
                                  **detection_metrics(y, maps.reshape(len(maps), -1).max(axis=1))}
        if examples > 0:
            save_examples(x, y, maps, category, run_dir / f"examples_{category}.png", examples)
        print(f"{category}: AUROC {per_category[category]['auroc']:.4f}", flush=True)
    test.pop("x")  # the images are not needed any more: leave the memory to step 2
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- 2. CPU, minutes: official AU-sPRO, several categories at the same time --
    # The official script keeps all full-size maps of a category in memory: about
    # 11 GB for 620 M pixels (measured). MEMORY_BUDGET limits what runs together,
    # e.g. two of the 1000 x 750 categories, or one of them and the 400 x 400 ones.
    cost = {c: (per_category[c]["good"] + per_category[c]["defective"])
            * meta["test"][c]["original_size"][0] * meta["test"][c]["original_size"][1]
            for c in categories}
    gate, in_use = threading.Condition(), [0.0]

    def pixel_level(category: str) -> dict:
        need = min(cost[category], MEMORY_BUDGET)
        with gate:
            while in_use[0] + need > MEMORY_BUDGET:
                gate.wait()
            in_use[0] += need
        try:
            mask, maps_test_dir = masks[category], maps_root / category / "test"
            write_anomaly_maps(maps_of.pop(category), test["defect"][mask], test["name"][mask],
                               meta["test"][category]["original_size"], maps_test_dir)
            auc = official_auc_spro(category, maps_test_dir, run_dir / "spro" / category, workers)
            if not keep_maps:
                shutil.rmtree(maps_root / category)
            return auc
        finally:
            with gate:
                in_use[0] -= need
                gate.notify_all()

    if spro:
        with ThreadPoolExecutor(len(categories)) as pool:
            for category, auc in zip(categories, pool.map(pixel_level, categories)):
                per_category[category]["auc_spro"] = auc
                print(f"{category}: AU-sPRO@0.05 {auc['0.05']:.4f}", flush=True)

    mean = {k: float(np.mean([per_category[c][k] for c in categories]))
            for k in ("auroc", "ap", "f1_max")}
    if spro:
        mean["auc_spro"] = {limit: float(np.mean([per_category[c]["auc_spro"][limit]
                                                  for c in categories]))
                            for limit in per_category[categories[0]]["auc_spro"]}
        if not keep_maps:
            shutil.rmtree(maps_root, ignore_errors=True)
    results = {"image_size": meta["image_size"], "per_category": per_category, "mean": mean}
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        privacy = json.loads(metrics_file.read_text()).get("privacy")
        if privacy:
            results["privacy"] = privacy
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a finished AutoVI run (image-level metrics + official AU-sPRO).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="folder with final_model*.pt")
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--no-spro", action="store_true", help="image-level metrics only (fast)")
    parser.add_argument("--keep-maps", action="store_true",
                        help="keep the anomaly-map PNGs (up to 1 GB per run)")
    parser.add_argument("--workers", type=int, default=0,
                        help="processes of the official evaluation PER CATEGORY (the categories "
                             "run at the same time); 0 = share all CPUs between them")
    parser.add_argument("--examples", type=int, default=4, help="heat-map examples per class (0 = none)")
    args = parser.parse_args()
    evaluate_run(args.run.resolve(), args.partitions.resolve(), spro=not args.no_spro,
                 keep_maps=args.keep_maps, workers=args.workers, examples=args.examples)


if __name__ == "__main__":
    main()
