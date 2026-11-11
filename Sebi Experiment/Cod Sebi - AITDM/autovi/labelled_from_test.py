"""SIDE EXPERIMENT: a labelled dataset built from the AutoVI *test* folders, so
that the starter kit's label-skew splits (Dirichlet, ...) and its classifier app
(fl_app/) can be used unchanged.

    python autovi/labelled_from_test.py --dataset-root ~/aitdm/autovi --size 224
    python data_splitting/split_data.py --input data/raw/autovi_test_labelled_224.npz \\
        --class-names "<printed by this script>" --strategy dirichlet --alpha 0.3 \\
        --num-clients 3 --out data/partitions/autovi_sup_dirichlet

Why the test folders: the training folders of AutoVI contain only good images,
so they have a single label and cannot be split by label. The test folders
contain good AND defective images.

What this changes, and what to say in the report:

* The task becomes SUPERVISED classification (the model sees defects during
  training), which is not how AutoVI is meant to be used: it assumes the
  defects are known in advance.
* The official test set is consumed: split_data.py holds a random part of it out
  as the new test set. The numbers are therefore NOT comparable with the
  unsupervised results (fl_app_autovi/) or with published AutoVI results.

Classes: <category>/good and <category>/defective for every category (all
defect types of a category are merged, some have only 1-3 images).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORIES = ["tank_screw", "pipe_staple", "underbody_pipes"]


def category_dir(root: Path, category: str) -> Path:
    for candidate in (root / category, root / category / category):
        if (candidate / "test" / "good").is_dir():
            return candidate
    raise FileNotFoundError(f"No test/good folder for '{category}' under {root}")


def load_image(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Labelled dataset (good / defective per category) from the AutoVI test folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--size", type=int, default=224, help="images are resized to size x size")
    parser.add_argument("--out", type=Path,
                        help="output file (default data/raw/autovi_test_labelled_<size>.npz)")
    args = parser.parse_args()
    out = args.out or REPO_ROOT / "data" / "raw" / f"autovi_test_labelled_{args.size}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    images, labels, category_ids, defects, class_names = [], [], [], [], []
    with ThreadPoolExecutor(16) as pool:
        for cid, category in enumerate(args.categories):
            class_names += [f"{category}/good", f"{category}/defective"]
            test_dir = category_dir(args.dataset_root.expanduser(), category) / "test"
            for folder in sorted(p for p in test_dir.iterdir() if p.is_dir()):
                files = sorted(folder.glob("*.png"))
                images += list(pool.map(lambda p: load_image(p, args.size), files))
                labels += [2 * cid + int(folder.name != "good")] * len(files)
                category_ids += [cid] * len(files)
                defects += [folder.name] * len(files)

    labels = np.array(labels, dtype=np.int64)
    np.savez(out, images=np.stack(images), labels=labels,
             category=np.array(category_ids, dtype=np.int64), defect=np.array(defects))
    print(f"Wrote {len(labels)} images ({args.size}x{args.size}) to {out}")
    for index, name in enumerate(class_names):
        print(f"  class {index}: {name:<28} {int((labels == index).sum())} images")
    print(f'--class-names "{",".join(class_names)}"')


if __name__ == "__main__":
    main()
