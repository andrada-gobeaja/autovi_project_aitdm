"""The two non-federated reference points, with the SAME student, optimiser and
evaluation as the federated runs:

* centralised: one student trained on the pooled training images of all
  clients (what you could do if the stations were allowed to share images)
* local:       one student per client, trained on that client's images only
  (no collaboration at all; the usual one-model-per-category protocol of the
  anomaly-detection literature, e.g. PatchCore)

    python autovi/baselines.py --mode centralised --partitions data/partitions/autovi_384 --epochs 100
    python autovi/baselines.py --mode local       --partitions data/partitions/autovi_384 --epochs 100

Use epochs = num-server-rounds x local-epochs of the federated run, so every
image is seen equally often. Each seed is saved like a federated run
(outputs/autovi/<name>/seed<s>/ with final_model*.pt, metrics.json, results.json).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate_run  # noqa: E402

from autovi_fl.task import (  # noqa: E402
    ImageDataset,
    build_student,
    client_files,
    get_device,
    get_weights,
    load_images,
    load_meta,
    set_seed,
    train,
    validation_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def fit(x_train: np.ndarray, args, seed: int, tag: str) -> tuple[torch.nn.Module, list[float]]:
    """Train a fresh student on x_train; returns it and the loss of every epoch."""
    set_seed(seed)
    student = build_student()
    loader = DataLoader(ImageDataset(x_train), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    losses = []

    def log(epoch: int, loss: float) -> None:
        losses.append(loss)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"  [{tag}] epoch {epoch}/{args.epochs}: train loss {loss:.4f}", flush=True)

    train(student, loader, epochs=args.epochs, lr=args.lr, device=get_device(),
          momentum=args.momentum, weight_decay=args.weight_decay, amp=not args.no_amp,
          on_epoch_end=log)
    return student, losses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Centralised and local-only baselines for AutoVI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", choices=["centralised", "local"], required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--name", help="experiment name (default: the mode)")
    parser.add_argument("--no-spro", action="store_true", help="skip the official AU-sPRO evaluation")
    parser.add_argument("--force", action="store_true", help="re-run seeds that are done")
    args = parser.parse_args()

    partitions = args.partitions.resolve()
    meta = load_meta(partitions)
    if args.mode == "local" and meta.get("strategy", "category") != "category":
        parser.error("--mode local needs one client per category (prepare_data.py --strategy "
                     "category): with mixed clients there is no 'the model of this category'")
    exp_dir = REPO_ROOT / "outputs" / "autovi" / (args.name or args.mode)
    device = get_device()
    files = client_files(partitions)
    train_sets = [load_images(f, "train") for f in files]
    val_loaders = [DataLoader(ImageDataset(load_images(f, "val")), batch_size=args.batch_size)
                   for f in files]

    for seed in args.seeds:
        run_dir = exp_dir / f"seed{seed}"
        if (run_dir / "results.json").exists() and not args.force:
            print(f">>> {exp_dir.name} seed {seed}: done already, skipped")
            continue
        if (run_dir / "metrics.json").exists() and not args.force:  # written after training
            print(f">>> {exp_dir.name} seed {seed}: trained already, evaluating only", flush=True)
            evaluate_run(run_dir, partitions, spro=not args.no_spro)
            continue
        print(f">>> {exp_dir.name} seed {seed}", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        history = {"run_config": {"mode": args.mode, "partitions-dir": str(partitions),
                                  "seed": seed, "epochs": args.epochs, "learning-rate": args.lr,
                                  "momentum": args.momentum, "weight-decay": args.weight_decay,
                                  "batch-size": args.batch_size, "amp": not args.no_amp},
                   "train_loss": {}, "val": {}}

        if args.mode == "centralised":
            student, losses = fit(np.concatenate(train_sets), args, seed, "pooled")
            torch.save(get_weights(student), run_dir / "final_model.pt")
            history["train_loss"]["pooled"] = losses
            students = {c: student for c in meta["categories"]}
        else:
            students = {}
            for category, x_train in zip(meta["categories"], train_sets):
                students[category], losses = fit(x_train, args, seed, category)
                torch.save(get_weights(students[category]), run_dir / f"final_model_{category}.pt")
                history["train_loss"][category] = losses

        # Each client's held-out good images, with the model that client would use
        for category, loader in zip(meta["categories"], val_loaders):
            history["val"][category] = validation_metrics(students[category], loader, device)
        history["train_seconds"] = round(time.time() - start, 1)
        (run_dir / "metrics.json").write_text(json.dumps(history, indent=2))
        evaluate_run(run_dir, partitions, spro=not args.no_spro)


if __name__ == "__main__":
    main()
