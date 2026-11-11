"""Turn the AutoVI dataset (Renault Group, Zenodo 10459003) into federated client
partitions in the same format as split_data.py, so the Flower app can read them.

AutoVI is an UNSUPERVISED anomaly-detection benchmark: train/ only contains GOOD
images; test/ has good and defective images (+ pixel masks in ground_truth/).
So, unlike split_data.py:

  * clients only receive train/good images (each client = a production station);
  * test.npz is the OFFICIAL test set, untouched: y_test = 0 (good) / 1 (defect).
    No client ever sees a defect, which is the realistic industrial setting.

Non-IID splits (the "label" used for partitioning is the object type):
  --strategy group      one object per client (station 0 = engine_wiring, ...):
                        strong feature / domain skew. num-clients <= #objects
  --strategy dirichlet  every client gets a skewed MIX of objects (--alpha);
                        lets you use 3-5 clients with only 3 objects
  --strategy quantity   same object mix, very different client sizes (--alpha)
  --strategy iid        same object mix everywhere (reference point)
  --feature-skew        (any strategy) a different "camera" per client: darker /
                        brighter, noisier images. The test set is never modified

Example (from the repository root, venv active):

    python data_splitting/prepare_autovi.py --root data/raw/autovi \\
        --objects engine_wiring pipe_clip underbody_screw \\
        --size 256 --strategy group --out data/partitions/autovi_group

Outputs (in --out):
    client_<i>.npz   x_train, y_train, x_val, y_val (+ obj_train, obj_val)
    test.npz         x_test, y_test (0 good / 1 defect), obj_test, defect_test,
                     path_test ("<object>/test/<defect>/<id>", to write anomaly maps
                     in the folder layout the official AutoVI evaluation script expects)
    meta.json        settings, objects, original image sizes, per-client counts
    object_distribution.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from partitioning import (  # noqa: E402
    apply_feature_skew,
    dirichlet_split,
    group_split,
    iid_split,
    quantity_split,
)
from split_data import plot_distribution, print_report, train_val_split  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoVI -> federated partitions (train: good only, test: official).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "raw" / "autovi",
                        help="folder that contains one sub-folder per AutoVI object")
    parser.add_argument("--objects", nargs="+",
                        help="objects to use (default: every sub-folder of --root with train/good)")
    parser.add_argument("--size", type=int, default=256,
                        help="images are resized to size x size (a multiple of 32; 256 as in RD4AD)")
    parser.add_argument("--strategy", choices=["group", "dirichlet", "quantity", "iid"], default="group")
    parser.add_argument("--num-clients", type=int,
                        help="default: number of objects (group) or 3 (other strategies)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration (dirichlet / quantity)")
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="fraction of each client's GOOD images kept for local validation")
    parser.add_argument("--feature-skew", action="store_true",
                        help="different camera per client (brightness / noise), as in split_data.py")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, help="default: data/partitions/autovi_<strategy>")
    args = parser.parse_args()

    if args.objects is None:
        args.objects = sorted(p.name for p in args.root.iterdir()
                              if (p / "train" / "good").is_dir()) if args.root.is_dir() else []
    if not args.objects:
        parser.error(f"no AutoVI objects found in {args.root} (expected <object>/train/good/)")
    if args.num_clients is None:
        args.num_clients = len(args.objects) if args.strategy == "group" else 3
    if args.out is None:
        args.out = REPO_ROOT / "data" / "partitions" / f"autovi_{args.strategy}"
    return args


def load_image(path: Path, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """RGB uint8 (size, size, 3). Some objects are stored as RGBA: alpha is dropped."""
    with Image.open(path) as image:
        original_size = image.size  # (width, height)
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8), original_size


def load_object(root: Path, name: str, size: int) -> dict:
    folder = root / name
    train_files = sorted((folder / "train" / "good").glob("*.png"))
    if not train_files:
        sys.exit(f"ERROR: no images in {folder / 'train' / 'good'}")
    x_train, sizes = zip(*(load_image(p, size) for p in train_files))

    x_test, y_test, defects, rel_paths = [], [], [], []
    for defect_dir in sorted(d for d in (folder / "test").iterdir() if d.is_dir()):
        for path in sorted(defect_dir.glob("*.png")):
            x_test.append(load_image(path, size)[0])
            y_test.append(0 if defect_dir.name == "good" else 1)
            defects.append(defect_dir.name)
            rel_paths.append(f"{name}/test/{defect_dir.name}/{path.stem}")
    print(f"{name:>16}: {len(train_files)} train (good) | {len(x_test)} test "
          f"({y_test.count(0)} good, {y_test.count(1)} defect) | original size {sizes[0]}")
    return {"x_train": np.stack(x_train), "original_size": list(sizes[0]),
            "x_test": np.stack(x_test), "y_test": np.array(y_test, dtype=np.int64),
            "defect_test": np.array(defects), "path_test": np.array(rel_paths)}


def main() -> None:
    args = get_arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    objects = {name: load_object(args.root, name, args.size) for name in args.objects}

    # Pool of good training images, "label" = object id (used only for partitioning)
    x_pool = np.concatenate([o["x_train"] for o in objects.values()])
    obj_pool = np.concatenate([np.full(len(o["x_train"]), i)
                               for i, o in enumerate(objects.values())])

    if args.strategy == "group":
        parts = group_split(obj_pool, args.num_clients, seed=args.seed)
    elif args.strategy == "dirichlet":
        parts = dirichlet_split(obj_pool, args.num_clients, alpha=args.alpha,
                                min_samples=args.min_samples, seed=args.seed)
    elif args.strategy == "quantity":
        parts = quantity_split(obj_pool, args.num_clients, alpha=args.alpha,
                               min_samples=args.min_samples, seed=args.seed)
    else:
        parts = iid_split(obj_pool, args.num_clients, seed=args.seed)

    counts, client_info = [], []
    for cid, indices in enumerate(parts):
        train_idx, val_idx = train_val_split(indices, args.val_fraction, args.seed + cid)
        x_train, x_val = x_pool[train_idx], x_pool[val_idx]
        if args.feature_skew:  # the same "camera" takes the client's train and val images
            x_train = apply_feature_skew(x_train, cid, args.num_clients, args.seed)
            x_val = apply_feature_skew(x_val, cid, args.num_clients, args.seed)
        np.savez_compressed(
            args.out / f"client_{cid}.npz",
            x_train=x_train, y_train=np.zeros(len(train_idx), dtype=np.int64),
            x_val=x_val, y_val=np.zeros(len(val_idx), dtype=np.int64),
            obj_train=obj_pool[train_idx], obj_val=obj_pool[val_idx],
        )
        per_object = [int((obj_pool[train_idx] == i).sum()) for i in range(len(objects))]
        counts.append(per_object)
        client_info.append({"client": cid, "train": int(len(train_idx)),
                            "val": int(len(val_idx)), "train_object_counts": per_object})

    # Official test set, untouched
    test = {key: np.concatenate([o[key] for o in objects.values()])
            for key in ("x_test", "y_test", "defect_test", "path_test")}
    test["obj_test"] = np.concatenate([np.full(len(o["y_test"]), i)
                                       for i, o in enumerate(objects.values())])
    np.savez_compressed(args.out / "test.npz", **test)

    test_breakdown = {
        name: {d: int((o["defect_test"] == d).sum()) for d in np.unique(o["defect_test"])}
        for name, o in objects.items()}
    meta = {
        "source": "AutoVI (Zenodo 10459003)",
        "data_type": "images",
        "task": "unsupervised_anomaly_detection",
        "strategy": args.strategy,
        "num_clients": args.num_clients,
        "num_classes": 2,                      # test labels: 0 good, 1 defect
        "class_names": ["good", "defect"],
        "objects": list(objects),
        "object_original_size": {n: o["original_size"] for n, o in objects.items()},
        "sample_shape": [args.size, args.size, 3],
        "test_samples": int(len(test["y_test"])),
        "test_breakdown": test_breakdown,
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "clients": client_info,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    counts = np.array(counts)
    print_report(counts, list(objects))
    plot_distribution(counts, list(objects),
                      f"AutoVI, {args.strategy} split: objects (stations) per client",
                      args.out / "object_distribution.png")
    print(f"Test set: {len(test['y_test'])} images "
          f"({int((test['y_test'] == 0).sum())} good, {int(test['y_test'].sum())} defect)")
    print(f"Partitions written to {args.out}")


if __name__ == "__main__":
    main()
