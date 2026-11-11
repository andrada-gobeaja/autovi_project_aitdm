"""Unsupervised anomaly detection for AutoVI (Renault Group) with Reverse
Distillation, used by the Flower app.

Reverse Distillation (Deng & Li, "Anomaly Detection via Reverse Distillation from
One-Class Embedding", CVPR 2022, "RD4AD"), compact re-implementation:

  teacher  E  frozen CNN pretrained on ImageNet (ResNet18 or WideResNet50-2): its
              layer1 / layer2 / layer3 features of the image
  OCBE        trainable "one-class bottleneck": fuses E's three feature maps and
              compresses them to a small map (1/32 of the image size)
  student  D  trainable decoder: from the bottleneck it rebuilds E's three feature
              maps in REVERSE order (layer3, then layer2, then layer1)

Trained on good images only, D learns to rebuild the teacher's features of good
parts. On a defect it fails, so:

  anomaly map   = sum over the 3 scales of (1 - cosine similarity between E's and
                  D's feature vectors), upsampled to the image size and smoothed,
  anomaly score = maximum of the anomaly map, the same definition as the official
                  AutoVI evaluation code (github.com/phcarval/autovi_evaluation_code).

Federated learning: only the trainable part (OCBE + D, the "student") is sent to
the server and averaged. Every client and the server build the SAME frozen teacher
from the ImageNet weights themselves; it never travels.

Differences to the original code, for a model that is light enough to exchange
every round: 2 blocks per decoder stage and 1 block in the one-class embedding.

The app switches to this module by itself when the partitions' meta.json says
task = "unsupervised_anomaly_detection" (written by data_splitting/prepare_autovi.py).
MedMNIST / HAR runs keep using the classifier in task.py, unchanged.
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from aitdm_fl.task import ArrayDataset, load_npz_arrays

ANOMALY_TASK = "unsupervised_anomaly_detection"
SMOOTHING_SIGMA = 4.0      # Gaussian blur of the anomaly map (pixels), as in RD4AD
THRESHOLD_QUANTILE = 0.95  # each client: 95 % of its good validation images score below
# Channels of the teacher's layer1, layer2, layer3
BACKBONES = {"resnet18": (64, 128, 256), "wide_resnet50_2": (256, 512, 1024)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def is_anomaly_task(partitions_dir: str) -> bool:
    meta = Path(partitions_dir) / "meta.json"
    return meta.exists() and json.loads(meta.read_text()).get("task") == ANOMALY_TASK


def read_meta(partitions_dir: str) -> dict:
    return json.loads((Path(partitions_dir) / "meta.json").read_text())


# ---------------------------------------------------------------------------
# Teacher: frozen, pretrained, never sent
# ---------------------------------------------------------------------------
class Teacher(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"rd-backbone must be one of {list(BACKBONES)}, got '{backbone}'")
        net = getattr(torchvision.models, backbone)(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # always in eval mode (frozen BatchNorm)
        return super().train(False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem((x - self.mean) / self.std)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        return [f1, f2, self.layer3(f2)]


@functools.lru_cache(maxsize=4)
def get_teacher(backbone: str, device: str) -> Teacher:
    """One teacher per process and device (the simulation re-uses its client
    processes, so the weights are loaded once, not every round)."""
    return Teacher(backbone).to(device)


# ---------------------------------------------------------------------------
# Student: one-class bottleneck + reverse decoder (the part that is federated)
# ---------------------------------------------------------------------------
def conv_bn_relu(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Block(nn.Module):
    """Bottleneck residual block that can halve ("down") or double ("up") the
    resolution."""

    def __init__(self, cin: int, cout: int, mid: int, resample: str | None = None):
        super().__init__()
        if resample == "up":
            middle = nn.ConvTranspose2d(mid, mid, 2, stride=2, bias=False)
            shortcut = [nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(cin, cout, 1, bias=False)]
        elif resample == "down":
            middle = nn.Conv2d(mid, mid, 3, stride=2, padding=1, bias=False)
            shortcut = [nn.Conv2d(cin, cout, 1, stride=2, bias=False)]
        else:
            middle = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
            shortcut = [nn.Conv2d(cin, cout, 1, bias=False)] if cin != cout else []
        self.body = nn.Sequential(
            nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            middle, nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.shortcut = (nn.Sequential(*shortcut, nn.BatchNorm2d(cout)) if shortcut
                         else nn.Identity())

    def forward(self, x):
        return F.relu(self.body(x) + self.shortcut(x))


class RDStudent(nn.Module):
    """OCBE (multi-scale fusion + one-class embedding) and the reverse decoder."""

    def __init__(self, backbone: str):
        super().__init__()
        c1, c2, c3 = BACKBONES[backbone]
        c4 = 2 * c3
        self.backbone = backbone
        # Multi-scale feature fusion: bring layer1 and layer2 down to layer3's size
        self.down1 = nn.Sequential(conv_bn_relu(c1, c2, 2), conv_bn_relu(c2, c3, 2))
        self.down2 = conv_bn_relu(c2, c3, 2)
        # One-class embedding: compress the fused features (1/16 -> 1/32 of the image)
        self.embed = Block(3 * c3, c4, c4 // 4, "down")
        # Reverse decoder: bottleneck -> layer3 -> layer2 -> layer1 sizes and channels
        self.dec3 = nn.Sequential(Block(c4, c3, c3 // 2, "up"), Block(c3, c3, c3 // 2))
        self.dec2 = nn.Sequential(Block(c3, c2, c2 // 2, "up"), Block(c2, c2, c2 // 2))
        self.dec1 = nn.Sequential(Block(c2, c1, c1 // 2, "up"), Block(c1, c1, c1 // 2))

    def forward(self, teacher_feats: list[torch.Tensor]) -> list[torch.Tensor]:
        f1, f2, f3 = teacher_feats
        z = self.embed(torch.cat([self.down1(f1), self.down2(f2), f3], dim=1))
        d3 = self.dec3(z)
        d2 = self.dec2(d3)
        return [self.dec1(d2), d2, d3]  # same order as the teacher: layer1, 2, 3


def build_rd_student(backbone: str, sample_shape: tuple[int, ...]) -> RDStudent:
    """sample_shape = (C, H, W) of one image."""
    channels, height, width = sample_shape
    if channels != 3:
        raise ValueError(f"Reverse Distillation needs RGB images, got {channels} channels")
    if height % 32 or width % 32:
        raise ValueError(f"Image size {height}x{width} must be a multiple of 32: "
                         "use --size 256 (or 128) in prepare_autovi.py")
    return RDStudent(backbone)


def student_from_partitions(partitions_dir: str, backbone: str) -> RDStudent:
    """For the server and analysis scripts (clients use their own data's shape)."""
    height, width, channels = read_meta(partitions_dir)["sample_shape"]
    return build_rd_student(backbone, (channels, height, width))


# ---------------------------------------------------------------------------
# Data (no augmentation: a flipped or rotated part may itself be a defect)
# ---------------------------------------------------------------------------
def load_ad_client_data(data_path: Path, batch_size: int,
                        seed: int | None = None) -> tuple[DataLoader, DataLoader]:
    x_tr, y_tr = load_npz_arrays(data_path, "train")
    x_va, y_va = load_npz_arrays(data_path, "val")
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_loader = DataLoader(ArrayDataset(x_tr, y_tr), batch_size=batch_size,
                              shuffle=True, generator=generator)
    return train_loader, DataLoader(ArrayDataset(x_va, y_va), batch_size=batch_size)


def load_ad_test_set(partitions_dir: str) -> dict:
    """The official AutoVI test set and what is needed to write its anomaly maps."""
    path = Path(partitions_dir) / "test.npz"
    x, y = load_npz_arrays(path, "test")
    data = np.load(path)
    meta = read_meta(partitions_dir)
    return {"x": x, "y": y, "obj": data["obj_test"], "path": data["path_test"],
            "objects": meta["objects"], "original_size": meta["object_original_size"]}


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def distillation_loss(teacher_feats, student_feats) -> torch.Tensor:
    """Mean over positions of (1 - cosine similarity), summed over the 3 scales."""
    return sum((1 - F.cosine_similarity(t, s, dim=1)).mean()
               for t, s in zip(teacher_feats, student_feats))


def train_rd(student: RDStudent, loader, epochs: int, lr: float, device,
             proximal_mu: float = 0.0) -> float:
    """Train the student on good images. Labels are ignored. Adam with
    betas (0.5, 0.999) as in RD4AD. proximal_mu > 0 adds the FedProx penalty,
    exactly as task.train() does."""
    teacher = get_teacher(student.backbone, str(device))
    student.to(device)
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, betas=(0.5, 0.999))
    global_params = [p.detach().clone() for p in student.parameters()]
    total_loss, total_batches = 0.0, 0
    for _ in range(epochs):
        for inputs, _labels in loader:
            inputs = inputs.to(device)
            with torch.no_grad():
                t_feats = teacher(inputs)
            optimizer.zero_grad()
            loss = distillation_loss(t_feats, student(t_feats))
            if proximal_mu > 0:
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(student.parameters(), global_params))
                loss = loss + (proximal_mu / 2) * prox
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
    return total_loss / max(total_batches, 1)


def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
    radius = int(math.ceil(3 * sigma))
    t = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    return k / k.sum()


@torch.no_grad()
def anomaly_maps(student: RDStudent, x: np.ndarray | torch.Tensor, device,
                 batch_size: int = 32) -> tuple[np.ndarray, float]:
    """Per-pixel anomaly maps (N, H, W) at the input size, and the mean loss."""
    teacher = get_teacher(student.backbone, str(device))
    student.to(device)
    student.eval()
    x = torch.as_tensor(x)
    size = tuple(x.shape[-2:])
    kernel = _gaussian_kernel(SMOOTHING_SIGMA, device)
    pad = len(kernel) // 2
    maps, loss_sum = [], 0.0
    for start in range(0, len(x), batch_size):
        inputs = x[start:start + batch_size].to(device)
        t_feats = teacher(inputs)
        s_feats = student(t_feats)
        amap = torch.zeros(len(inputs), 1, *size, device=device)
        for t, s in zip(t_feats, s_feats):
            d = 1 - F.cosine_similarity(t, s, dim=1)                  # (B, h, w)
            loss_sum += d.mean(dim=(1, 2)).sum().item()
            amap += F.interpolate(d[:, None], size=size, mode="bilinear", align_corners=False)
        amap = F.conv2d(F.pad(amap, (pad, pad, 0, 0), mode="reflect"), kernel.view(1, 1, 1, -1))
        amap = F.conv2d(F.pad(amap, (0, 0, pad, pad), mode="reflect"), kernel.view(1, 1, -1, 1))
        maps.append(amap[:, 0].cpu().numpy())
    return np.concatenate(maps), loss_sum / max(len(x), 1)


def image_scores(maps: np.ndarray) -> np.ndarray:
    """Image-level score = maximum of the anomaly map (as in the official evaluation)."""
    return maps.reshape(len(maps), -1).max(axis=1)


def evaluate_val(student, loader, device) -> dict[str, float]:
    """On a client's GOOD validation images: the distillation loss and the score
    below which THRESHOLD_QUANTILE of them fall (a decision threshold that the
    client can share without sharing any image)."""
    maps, loss = anomaly_maps(student, loader.dataset.x, device)
    scores = image_scores(maps)
    threshold = float(np.quantile(scores, THRESHOLD_QUANTILE)) if len(scores) else 0.0
    return {"loss": loss, "threshold": threshold}


def detection_metrics(scores: np.ndarray, y: np.ndarray, obj: np.ndarray,
                      objects: list[str], threshold: float | None = None) -> dict[str, float]:
    """Image-level metrics on the test set (y: 0 good, 1 defect).

    auroc            mean over objects of the per-object AUROC (main number: like the
                     AutoVI paper, every object is evaluated on its own)
    auroc_<object>   per object
    auroc_pooled     all objects together (also needs the scores of different
                     objects to be comparable)
    f1_max           best F1 over all thresholds: an upper bound, it peeks at test labels
    f1               F1 at the threshold agreed by the clients (if given)
    """
    results = {}
    per_object = []
    for i, name in enumerate(objects):
        mask = obj == i
        if len(np.unique(y[mask])) == 2:
            results[f"auroc_{name}"] = float(roc_auc_score(y[mask], scores[mask]))
            per_object.append(results[f"auroc_{name}"])
    results["auroc"] = float(np.mean(per_object)) if per_object else 0.0
    results["auroc_pooled"] = float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else 0.0
    precision, recall, _ = precision_recall_curve(y, scores)
    f1s = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    results["f1_max"] = float(f1s.max())
    if threshold is not None:
        results["threshold"] = float(threshold)
        results["f1"] = float(f1_score(y, scores >= threshold, zero_division=0))
    return results


def write_anomaly_maps(maps: np.ndarray, test: dict, out_dir: Path) -> None:
    """Write one 16-bit, single-channel PNG per test image, at the original image
    size, in the layout of the official evaluation code:

        <out_dir>/<object>/test/<defect>/<image_id>.png

    Each object's maps are divided by that object's maximum, so 16 bits are used
    well. This does not change any metric (same order of all values of an object)."""
    for i, name in enumerate(test["objects"]):
        idx = np.flatnonzero(test["obj"] == i)
        if len(idx) == 0:
            continue
        width, height = test["original_size"][name]
        scale = float(maps[idx].max()) or 1.0
        for j in idx:
            m = torch.from_numpy(maps[j] / scale)[None, None]
            m = F.interpolate(m, size=(height, width), mode="bilinear", align_corners=False)
            arr = (m[0, 0].clamp(0, 1).numpy() * 65535).astype(np.uint16)
            target = out_dir / f"{test['path'][j]}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(target)
