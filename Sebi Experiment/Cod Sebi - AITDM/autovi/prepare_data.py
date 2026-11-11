"""AutoVI -> federated partitions for unsupervised anomaly detection.

    # one client per category (default): the extreme non-IID case
    python autovi/prepare_data.py --dataset-root ~/aitdm/autovi \\
        --categories tank_screw pipe_staple underbody_pipes --size 384

    # every client gets a Dirichlet(alpha) mix of the categories
    python autovi/prepare_data.py --dataset-root ~/aitdm/autovi --strategy dirichlet \\
        --alpha 0.3 --out data/partitions/autovi_384_dirichlet0.3

The training data of AutoVI contains only defect-free images, so the usual
label skew (good vs. defective) does not exist. What differs between clients
is WHICH OBJECT they photograph. Using the category as the label:

* --strategy category   client i owns category i: no overlap at all (the limit
                        alpha -> 0 of the Dirichlet split), plus quantity skew
* --strategy dirichlet  the starter kit's dirichlet_split() on the category
                        label: clients hold different mixtures of the objects.
                        Small alpha = close to "category", large alpha = IID.

Both strategies use the SAME training and validation images in total (the
validation images are drawn first, per category, with --seed), so runs on the
two splits and the centralised baseline are directly comparable.

Output (default data/partitions/autovi_<size>/), same layout as split_data.py:

    client_<i>.npz   x_train, x_val : good images, (N, S, S, 3) uint8
                     category_train, category_val, name_train, name_val
    test.npz         x_test, y_test (0 good / 1 defective), category_test,
                     defect_test, name_test : the official test images of all categories
    meta.json        categories, sizes, counts per client and category, settings
    label_distribution.png   categories per client

All images are resized to S x S (no cropping: a defect may be anywhere). The
test set is the official one and is never used for training or tuning.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORIES = ["tank_screw", "pipe_staple", "underbody_pipes"]


def category_dir(root: Path, category: str) -> Path:
    """<root>/<category>/ or, as the zips unpack, <root>/<category>/<category>/."""
    for candidate in (root / category, root / category / category):
        if (candidate / "train" / "good").is_dir():
            return candidate
    raise FileNotFoundError(f"No train/good folder for '{category}' under {root}")


def load_image(path: Path, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """RGB uint8 (size, size, 3) and the original (width, height).
    pipe_clip / pipe_staple are stored as RGBA: the alpha channel is dropped."""
    with Image.open(path) as im:
        original = im.size
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8), original


def load_folder(folder: Path, size: int, pool: ThreadPoolExecutor):
    files = sorted(folder.glob("*.png"))
    loaded = list(pool.map(lambda p: load_image(p, size), files))
    images = np.stack([img for img, _ in loaded]) if loaded else np.zeros((0, size, size, 3), np.uint8)
    return images, [p.stem for p in files], {orig for _, orig in loaded}


def deal_validation(val_cat: np.ndarray, train_parts: list[np.ndarray], train_cat: np.ndarray,
                    rng: np.random.Generator) -> list[np.ndarray]:
    """Give every client validation images with the same category mixture as its
    training images: the validation images of a category are dealt in proportion
    to the clients' shares of that category's training images."""
    num_clients = len(train_parts)
    parts: list[list[int]] = [[] for _ in range(num_clients)]
    for c in np.unique(val_cat):
        idx = rng.permutation(np.flatnonzero(val_cat == c))
        shares = np.array([(train_cat[p] == c).sum() for p in train_parts], dtype=float)
        cuts = (np.cumsum(shares / shares.sum()) * len(idx)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(idx, cuts)):
            parts[cid] += chunk.tolist()
    for cid in range(num_clients):  # nobody is left without validation data
        if not parts[cid]:
            donor = max(range(num_clients), key=lambda i: len(parts[i]))
            parts[cid].append(parts[donor].pop())
    return [np.sort(np.array(p, dtype=int)) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoVI -> federated client partitions (good images only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="folder that contains one sub-folder per category")
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES,
                        help="with --strategy category: client 0, 1, 2, ... in this order")
    parser.add_argument("--size", type=int, default=384, help="images are resized to size x size")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="part of every category's good training images held out for validation")
    parser.add_argument("--strategy", choices=["category", "dirichlet"], default="category")
    parser.add_argument("--num-clients", type=int, default=3, help="[dirichlet] number of clients")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="[dirichlet] concentration: smaller = closer to one category per client")
    parser.add_argument("--min-samples", type=int, default=20,
                        help="[dirichlet] minimum training images per client")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed of the train / validation split and of the Dirichlet split")
    parser.add_argument("--out", type=Path, help="output folder (default data/partitions/autovi_<size>)")
    args = parser.parse_args()

    out = args.out or REPO_ROOT / "data" / "partitions" / f"autovi_{args.size}"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pool = ThreadPoolExecutor(16)
    categories = args.categories
    num_clients = len(categories) if args.strategy == "category" else args.num_clients
    meta = {"dataset": "AutoVI v1.0.0", "strategy": args.strategy, "num_clients": num_clients,
            "categories": categories, "image_size": args.size,
            "val_fraction": args.val_fraction, "split_seed": args.seed,
            "alpha": args.alpha if args.strategy == "dirichlet" else None,
            "clients": {}, "test": {}}
    train = {"x": [], "category": [], "name": []}
    val = {"x": [], "category": [], "name": []}
    test_x, test_y, test_cat, test_defect, test_name = [], [], [], [], []

    for index, category in enumerate(categories):
        base = category_dir(args.dataset_root.expanduser(), category)

        # --- official training images (good only) -> train / validation ---------
        images, names, sizes = load_folder(base / "train" / "good", args.size, pool)
        names = np.array([f"{category}/{n}" for n in names])
        order = rng.permutation(len(images))
        n_val = max(1, int(round(args.val_fraction * len(images))))
        for part, idx in ((val, np.sort(order[:n_val])), (train, np.sort(order[n_val:]))):
            part["x"].append(images[idx])
            part["category"].append(np.full(len(idx), index, dtype=np.int64))
            part["name"].append(names[idx])
        width, height = sorted(sizes)[0]

        # --- official test images (good + one folder per defect type) -----------
        counts = {}
        for folder in sorted(p for p in (base / "test").iterdir() if p.is_dir()):
            images, names, _ = load_folder(folder, args.size, pool)
            counts[folder.name] = len(images)
            test_x.append(images)
            test_y.append(np.full(len(images), int(folder.name != "good"), dtype=np.int64))
            test_cat.append(np.full(len(images), index, dtype=np.int64))
            test_defect += [folder.name] * len(images)
            test_name += names
        meta["test"][category] = {"original_size": [width, height], "counts": counts,
                                  "good": counts.get("good", 0),
                                  "defective": sum(v for k, v in counts.items() if k != "good")}
        print(f"{category}: {len(train['x'][-1])} train + {len(val['x'][-1])} val good images "
              f"({width}x{height} -> {args.size}x{args.size}), test {counts}")

    train = {k: np.concatenate(v) for k, v in train.items()}
    val = {k: np.concatenate(v) for k, v in val.items()}

    # --- who owns what ---------------------------------------------------------
    if args.strategy == "category":
        train_parts = [np.flatnonzero(train["category"] == c) for c in range(len(categories))]
        val_parts = [np.flatnonzero(val["category"] == c) for c in range(len(categories))]
    else:
        sys.path.insert(0, str(REPO_ROOT / "data_splitting"))
        from partitioning import dirichlet_split  # the starter kit's implementation

        train_parts = dirichlet_split(train["category"], num_clients, alpha=args.alpha,
                                      min_samples=args.min_samples, seed=args.seed)
        val_parts = deal_validation(val["category"], train_parts, train["category"], rng)

    counts = []
    for cid, (tr, va) in enumerate(zip(train_parts, val_parts)):
        np.savez(out / f"client_{cid}.npz",
                 x_train=train["x"][tr], x_val=val["x"][va],
                 category_train=train["category"][tr], category_val=val["category"][va],
                 name_train=train["name"][tr], name_val=val["name"][va])
        per_category = [int((train["category"][tr] == c).sum()) for c in range(len(categories))]
        counts.append(per_category)
        meta["clients"][str(cid)] = {
            "train": int(len(tr)), "val": int(len(va)),
            "train_per_category": dict(zip(categories, per_category)),
            "val_per_category": {c: int((val["category"][va] == i).sum())
                                 for i, c in enumerate(categories)}}
        if args.strategy == "category":
            meta["clients"][str(cid)]["category"] = categories[cid]
        print(f"client {cid}: {len(tr)} train ({dict(zip(categories, per_category))}), {len(va)} val")

    np.savez(out / "test.npz", x_test=np.concatenate(test_x), y_test=np.concatenate(test_y),
             category_test=np.concatenate(test_cat), defect_test=np.array(test_defect),
             name_test=np.array(test_name))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    try:  # the starter kit's plot: categories per client
        sys.path.insert(0, str(REPO_ROOT / "data_splitting"))
        from split_data import plot_distribution

        title = "AutoVI good training images | one client per category"
        if args.strategy == "dirichlet":
            title = f"AutoVI good training images | Dirichlet over the category (alpha={args.alpha})"
        plot_distribution(np.array(counts), categories, title, out / "label_distribution.png")
    except ImportError as error:  # matplotlib / pandas missing: the data is still complete
        print(f"(no plot: {error})")
    print(f"\nWrote {num_clients} clients and {len(test_name)} test images to {out}")


if __name__ == "__main__":
    main()
