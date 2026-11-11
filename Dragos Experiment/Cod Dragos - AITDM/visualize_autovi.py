"""Figures for a finished AutoVI run (Reverse Distillation + Flower).

    python experiments/visualize_autovi.py --run outputs/<run-folder>

Uses only what the run saved (metrics.json, anomaly_maps/) and the dataset
(test images and ground-truth masks). Writes to <run>/figures/:

  loss_curves.png            training and validation loss per round
  roc_curves.png             image-level ROC per object (AUROC = official "AUROC all")
  score_distributions.png    image scores of good vs defective test images
  auroc_per_defect_type.png  AUROC of each defect type vs good, with image counts
  examples_<object>.png      test image | ground truth | anomaly map, per defect type,
                             plus a typical good image and the good image with the
                             highest score (the worst false alarm)

Image score = maximum of the anomaly map (as in the official evaluation). The maps
were saved divided by each object's maximum, so scores are comparable within an
object, not across objects. "above X% of good" = share of that object's good test
images with a lower score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_map(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float32) / 65535.0


def collect_scores(maps_root: Path) -> dict[str, list[dict]]:
    """object -> [{defect, id, score, path}] for every saved anomaly map."""
    scores = {}
    for obj_dir in sorted(p for p in maps_root.iterdir() if p.is_dir()):
        items = []
        for defect_dir in sorted(p for p in (obj_dir / "test").iterdir() if p.is_dir()):
            for png in sorted(defect_dir.glob("*.png")):
                items.append({"defect": defect_dir.name, "id": png.stem,
                              "score": float(read_map(png).max()), "path": png})
        scores[obj_dir.name] = items
        print(f"  {obj_dir.name}: {len(items)} anomaly maps")
    return scores


def labels_and_scores(items):
    y = np.array([it["defect"] != "good" for it in items], dtype=int)
    s = np.array([it["score"] for it in items])
    return y, s


def plot_losses(metrics: dict, out: Path) -> None:
    def series(key, name):
        rounds = sorted(metrics.get(key, {}), key=int)
        return ([int(r) for r in rounds],
                [float(metrics[key][r][name]) for r in rounds if name in metrics[key][r]])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for key, name, label in [("train_metrics_clientapp", "train_loss", "training loss"),
                             ("evaluate_metrics_clientapp", "eval_loss",
                              "validation loss (good images)")]:
        rounds, values = series(key, name)
        if values:
            ax.plot(rounds[:len(values)], values, marker=".", label=label)
    ax.set_xlabel("federated round")
    ax.set_ylabel("distillation loss (1 - cosine)")
    ax.set_title("Loss per round (average over clients)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_rocs(scores: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for obj, items in scores.items():
        y, s = labels_and_scores(items)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{obj} (AUROC {roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("false positive rate (good images flagged)")
    ax.set_ylabel("true positive rate (defects found)")
    ax.set_title("Image-level ROC per object")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_distributions(scores: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(scores), figsize=(4.2 * len(scores), 3.5), squeeze=False)
    for ax, (obj, items) in zip(axes[0], scores.items()):
        y, s = labels_and_scores(items)
        bins = np.linspace(s.min(), s.max(), 30)
        ax.hist(s[y == 0], bins=bins, alpha=0.6, label=f"good (n={int((y == 0).sum())})")
        ax.hist(s[y == 1], bins=bins, alpha=0.6, label=f"defect (n={int((y == 1).sum())})")
        ax.set_title(obj)
        ax.set_xlabel("image score (relative to object max)")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("number of test images")
    fig.suptitle("Score distributions: the less they overlap, the better")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_per_type(scores: dict, out: Path) -> None:
    rows = []
    for obj, items in scores.items():
        good = [it["score"] for it in items if it["defect"] == "good"]
        for defect in sorted({it["defect"] for it in items} - {"good"}):
            bad = [it["score"] for it in items if it["defect"] == defect]
            y = np.r_[np.zeros(len(good)), np.ones(len(bad))]
            rows.append((obj, defect, len(bad), roc_auc_score(y, np.r_[good, bad])))
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.2))
    names = [f"{o} / {d} (n={n})" for o, d, n, _ in rows]
    colors = [f"C{list(scores).index(o)}" for o, *_ in rows]
    ax.barh(names, [a for *_, a in rows], color=colors)
    for i, (*_, n, a) in enumerate(rows):
        ax.text(a + 0.01, i, f"{a:.2f}" + ("  (very few images)" if n < 10 else ""),
                va="center", fontsize=8)
    ax.axvline(0.5, color="k", ls="--", lw=0.8)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC (defect type vs all good images of the object)")
    ax.set_title("Detection per defect type")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def gt_mask(dataset: Path, obj: str, defect: str, image_id: str, shape) -> np.ndarray | None:
    folder = dataset / obj / "ground_truth" / defect / image_id
    if defect == "good" or not folder.is_dir():
        return None
    mask = np.zeros(shape, dtype=bool)
    for png in folder.glob("*.png"):
        m = np.asarray(Image.open(png))
        if m.ndim == 3:  # RGB or RGBA (pipe_clip): ignore the alpha channel
            m = m[..., :3].max(axis=-1)
        mask |= m > 0
    return mask


def plot_examples(obj: str, items: list[dict], dataset: Path, out: Path,
                  per_type: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    good = sorted((it for it in items if it["defect"] == "good"), key=lambda it: it["score"])
    good_scores = np.array([it["score"] for it in good])
    chosen = []
    for defect in sorted({it["defect"] for it in items} - {"good"}):
        pool = [it for it in items if it["defect"] == defect]
        for i in rng.choice(len(pool), size=min(per_type, len(pool)), replace=False):
            chosen.append((defect, pool[i]))
    if good:
        chosen.append(("good (typical)", good[len(good) // 2]))
        chosen.append(("good (highest score)", good[-1]))

    fig, axes = plt.subplots(len(chosen), 3, figsize=(10, 3.2 * len(chosen)), squeeze=False,
                             layout="constrained")
    for row, (title, it) in zip(axes, chosen):
        image = np.asarray(Image.open(dataset / obj / "test" / it["defect"] / f"{it['id']}.png")
                           .convert("RGB"))
        amap = read_map(it["path"])
        mask = gt_mask(dataset, obj, it["defect"], it["id"], image.shape[:2])
        above = 100 * float((good_scores < it["score"]).mean()) if len(good_scores) else 0
        row[0].imshow(image)
        row[0].set_title(f"{title}  #{it['id']}", fontsize=9)
        row[1].imshow(image)
        if mask is not None:
            row[1].imshow(np.ma.masked_where(~mask, mask), cmap="autumn", alpha=0.5)
            row[1].set_title("ground truth", fontsize=9)
        else:
            row[1].set_title("ground truth: no defect", fontsize=9)
        row[2].imshow(image)
        heat = row[2].imshow(amap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        if mask is not None:
            row[2].contour(mask, levels=[0.5], colors="white", linewidths=0.8)
        row[2].set_title(f"anomaly map\nscore {it['score']:.2f}, above {above:.0f}% of good",
                         fontsize=9)
        for ax in row:
            ax.axis("off")
    fig.colorbar(heat, ax=axes[:, 2].tolist(), shrink=0.5, label="anomaly (relative)")
    fig.suptitle(f"{obj}: test examples (white outline = true defect)")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="outputs/<run> folder")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data" / "raw" / "autovi")
    parser.add_argument("--out", type=Path, help="default: <run>/figures")
    parser.add_argument("--examples-per-type", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="which examples are drawn")
    args = parser.parse_args()

    out = args.out or args.run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((args.run / "metrics.json").read_text())

    plot_losses(metrics, out / "loss_curves.png")
    print("Reading anomaly maps...")
    scores = collect_scores(args.run / "anomaly_maps")
    plot_rocs(scores, out / "roc_curves.png")
    plot_distributions(scores, out / "score_distributions.png")
    plot_per_type(scores, out / "auroc_per_defect_type.png")
    for obj, items in scores.items():
        print(f"  examples for {obj}")
        plot_examples(obj, items, args.dataset, out / f"examples_{obj}.png",
                      args.examples_per_type, args.seed)
    print(f"Figures saved to {out}:")
    for f in sorted(out.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
