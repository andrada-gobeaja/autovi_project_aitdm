"""Model, data loading, training and anomaly scoring for AutoVI – everything that
is *not* Flower-specific lives here, so the centralised and local-only baselines
(autovi/baselines.py) and the evaluation (autovi/evaluate.py) reuse it.

The task is UNSUPERVISED anomaly detection: the training data contains only
defect-free ("good") images, so there are no labels and no classifier.

Model: student-teacher feature pyramid matching (STFPM, Wang et al., BMVC 2021).

* teacher = ResNet-18 pre-trained on ImageNet, frozen. It is public knowledge:
  every participant downloads the same weights, they are never sent around.
* student = the same architecture, randomly initialised. It is trained to
  reproduce the teacher's feature maps (layers 1-3) on good images.
  ONLY THE STUDENT IS FEDERATED.
* On a defect the student has never learnt to imitate the teacher, so the two
  disagree: the per-pixel disagreement is the anomaly map, its maximum the
  image-level anomaly score.
"""

from __future__ import annotations

import json
import random
import zlib
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FeaturePyramid(torch.nn.Module):
    """ResNet-18 up to layer3. Returns the feature maps of layers 1, 2 and 3
    (64, 128, 256 channels at 1/4, 1/8, 1/16 of the input resolution).
    layer4 and the classification head are not needed, which leaves 2.8 M
    parameters (11 MB) to exchange instead of 11.7 M."""

    def __init__(self, pretrained: bool):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        self.conv1, self.bn1, self.relu, self.maxpool = net.conv1, net.bn1, net.relu, net.maxpool
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]


def build_student() -> torch.nn.Module:
    """The trainable (and federated) network, randomly initialised: call
    set_seed() first to get the same initial model."""
    return FeaturePyramid(pretrained=False)


@lru_cache(maxsize=2)
def get_teacher(device: torch.device) -> torch.nn.Module:
    """The frozen ImageNet teacher (cached: built once per process and device)."""
    teacher = FeaturePyramid(pretrained=True).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def get_weights(student: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    """What a client sends to the server: the student's weights and BatchNorm
    statistics. The integer step counters of BatchNorm are left out (they are
    not weights, and averaging them makes no sense)."""
    return OrderedDict((k, v.detach().cpu()) for k, v in student.state_dict().items()
                       if not k.endswith("num_batches_tracked"))


def set_weights(student: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    result = student.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.endswith("num_batches_tracked")]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"Weights do not fit the student: missing {missing}, "
                           f"unexpected {result.unexpected_keys}")


def load_student(model_path: Path | str) -> torch.nn.Module:
    """Student with the weights saved by a run (final_model*.pt)."""
    student = build_student()
    set_weights(student, torch.load(model_path, map_location="cpu"))
    return student


# ---------------------------------------------------------------------------
# Reproducibility (same helpers as fl_app/aitdm_fl/task.py)
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def derive_seed(seed: int, *parts) -> int:
    """A different but reproducible seed for every (client, round, ...) combination."""
    text = ":".join(str(p) for p in (seed, *parts))
    return zlib.crc32(text.encode())


# ---------------------------------------------------------------------------
# Data (files written by autovi/prepare_data.py)
# ---------------------------------------------------------------------------
def load_meta(partitions_dir: Path | str) -> dict:
    return json.loads((Path(partitions_dir) / "meta.json").read_text())


def load_images(path: Path, split: str) -> np.ndarray:
    """x_<split> of an .npz file: (N, H, W, 3) uint8. Kept as uint8 (4x smaller
    than float32); preprocess() converts one batch at a time, on the GPU."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run autovi/prepare_data.py first and pass the ABSOLUTE "
            "output folder with --run-config \"partitions-dir='...'\"")
    with np.load(path) as data:
        return data[f"x_{split}"]


class ImageDataset(Dataset):
    """Images only: there are no labels in unsupervised anomaly detection.
    No augmentation: a flipped or rotated assembly is itself an anomaly."""

    def __init__(self, x: np.ndarray):
        self.x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W) uint8

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


def load_client_data(data_path: Path, batch_size: int,
                     seed: int | None = None) -> tuple[DataLoader, DataLoader]:
    """Train / validation loaders of ONE client (both contain good images only)."""
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_loader = DataLoader(ImageDataset(load_images(data_path, "train")),
                              batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(ImageDataset(load_images(data_path, "val")), batch_size=batch_size)
    return train_loader, val_loader


def client_files(partitions_dir: Path | str) -> list[Path]:
    return sorted(Path(partitions_dir).glob("client_*.npz"),
                  key=lambda p: int(p.stem.split("_")[1]))


def load_test_data(partitions_dir: Path | str) -> dict[str, np.ndarray]:
    """The held-out test images of all categories (server / analysis only):
    x (N, H, W, 3) uint8, y (0 = good, 1 = defective), category index, defect
    type and file name of every image. Use it ONCE, for the final numbers."""
    path = Path(partitions_dir) / "test.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run autovi/prepare_data.py first.")
    with np.load(path) as data:
        return {"x": data["x_test"], "y": data["y_test"], "category": data["category_test"],
                "defect": data["defect_test"], "name": data["name_test"]}


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 (B, 3, H, W) -> normalised float32 on the device (ImageNet statistics,
    because the teacher was trained with them)."""
    x = batch.to(device, non_blocking=True).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Loss and anomaly maps
# ---------------------------------------------------------------------------
def layer_distances(teacher_feats, student_feats) -> list[torch.Tensor]:
    """For every layer: 0.5 * || t/|t| - s/|s| ||^2 at every position, (B, h, w).
    Features are L2-normalised over the channels, so only their direction counts."""
    return [0.5 * (F.normalize(t.float(), dim=1) - F.normalize(s.float(), dim=1)).pow(2).sum(1)
            for t, s in zip(teacher_feats, student_feats)]


def distillation_loss(teacher_feats, student_feats) -> torch.Tensor:
    """STFPM loss of ONE batch: per image, the distance averaged over the positions
    and summed over the layers; summed over the images (as in the paper, whose
    learning rate of 0.4 assumes this)."""
    return sum(d.mean(dim=(1, 2)).sum() for d in layer_distances(teacher_feats, student_feats))


def anomaly_map(teacher_feats, student_feats, size: tuple[int, int]) -> torch.Tensor:
    """(B, H, W) anomaly maps: the layer distances, upsampled to the image size and
    multiplied (STFPM). The cube root (geometric mean) keeps the same ranking of
    pixels but a much smaller dynamic range, which matters when the maps are
    stored as 16-bit images for the official evaluation."""
    product = None
    for d in layer_distances(teacher_feats, student_feats):
        d = F.interpolate(d[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]
        product = d if product is None else product * d
    return product.clamp_min(0).pow(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def train(student, loader, epochs: int, lr: float, device, momentum: float = 0.9,
          weight_decay: float = 1e-4, proximal_mu: float = 0.0, amp: bool = True,
          on_epoch_end=None) -> float:
    """Train the student on good images and return the average loss per image.

    With proximal_mu > 0 this is FedProx: a penalty keeps the local student close
    to the global one received from the server (helps with non-IID clients)."""
    teacher = get_teacher(device)
    student.to(device).train()
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    global_params = [p.detach().clone() for p in student.parameters()]
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    total_loss, total_images = 0.0, 0
    for epoch in range(epochs):
        epoch_loss, epoch_images = 0.0, 0
        for batch in loader:
            x = preprocess(batch, device)
            with torch.autocast(device.type, enabled=use_amp):
                with torch.no_grad():
                    teacher_feats = teacher(x)
                student_feats = student(x)
            loss = distillation_loss(teacher_feats, student_feats)
            if proximal_mu > 0:
                prox = sum(((p - g) ** 2).sum() for p, g in zip(student.parameters(), global_params))
                loss = loss + (proximal_mu / 2) * prox
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "The training loss is not finite. Lower the learning rate or "
                    "switch mixed precision off (amp=false / --no-amp).")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            epoch_images += len(batch)
        total_loss += epoch_loss
        total_images += epoch_images
        if on_epoch_end is not None:
            on_epoch_end(epoch + 1, epoch_loss / max(epoch_images, 1))
    return total_loss / max(total_images, 1)


@torch.no_grad()
def iter_anomaly_maps(student, x: np.ndarray, device, batch_size: int = 32):
    """Yield (B, H, W) float32 anomaly maps (on the device) for uint8 images
    (N, H, W, 3), batch by batch. The student is in eval mode.

    Always full precision: in the first rounds the student's BatchNorm statistics
    lag behind its weights, activations get large and half precision overflows
    (NaN scores). Inference is cheap anyway."""
    teacher = get_teacher(device)
    student.to(device).eval()
    for i in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[i:i + batch_size]).permute(0, 3, 1, 2)
        inputs = preprocess(batch, device)
        teacher_feats, student_feats = teacher(inputs), student(inputs)
        yield anomaly_map(teacher_feats, student_feats, size=tuple(inputs.shape[-2:]))


def image_scores(student, x: np.ndarray, device, batch_size: int = 32) -> np.ndarray:
    """Image-level anomaly score = maximum of the anomaly map."""
    scores = [m.flatten(1).max(dim=1).values.cpu().numpy()
              for m in iter_anomaly_maps(student, x, device, batch_size)]
    return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)


@torch.no_grad()
def validation_metrics(student, loader, device) -> dict[str, float]:
    """On held-out GOOD images: the distillation loss per image (how well the
    student imitates the teacher on normal data it has not trained on) and the
    mean / maximum image-level anomaly score (should stay low).
    Full precision, see iter_anomaly_maps()."""
    teacher = get_teacher(device)
    student.to(device).eval()
    total_loss, scores = 0.0, []
    for batch in loader:
        x = preprocess(batch, device)
        teacher_feats, student_feats = teacher(x), student(x)
        total_loss += distillation_loss(teacher_feats, student_feats).item()
        maps = anomaly_map(teacher_feats, student_feats, size=tuple(x.shape[-2:]))
        scores.append(maps.flatten(1).max(dim=1).values.cpu().numpy())
    if not scores:
        return {"loss": 0.0, "score_mean": 0.0, "score_max": 0.0}
    scores = np.concatenate(scores)
    return {"loss": total_loss / len(scores), "score_mean": float(scores.mean()),
            "score_max": float(scores.max())}


def detection_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Image-level detection: is the image defective (1) or good (0)?

    * auroc:  threshold-free; 0.5 = chance, 1.0 = perfect ranking
    * ap:     average precision (area under the precision-recall curve)
    * f1_max: best F1 over all thresholds. An ORACLE number (the threshold is
              picked on the test labels): an upper bound, not a deployable result.
    """
    precision, recall, _ = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return {"auroc": float(roc_auc_score(y, scores)),
            "ap": float(average_precision_score(y, scores)),
            "f1_max": float(f1.max())}


def evaluate_test(student_for, partitions_dir: Path | str, device,
                  test: dict[str, np.ndarray] | None = None) -> dict[str, dict[str, float]]:
    """Image-level metrics per category on the test set, plus their mean.

    student_for(category) returns the student to use for that category: the one
    global model for federated / centralised training, a different model per
    category for local-only training."""
    meta = load_meta(partitions_dir)
    test = test or load_test_data(partitions_dir)
    results = {}
    for index, category in enumerate(meta["categories"]):
        mask = test["category"] == index
        scores = image_scores(student_for(category), test["x"][mask], device)
        results[category] = detection_metrics(test["y"][mask], scores)
    results["mean"] = {k: float(np.mean([results[c][k] for c in meta["categories"]]))
                       for k in ("auroc", "ap", "f1_max")}
    return results
