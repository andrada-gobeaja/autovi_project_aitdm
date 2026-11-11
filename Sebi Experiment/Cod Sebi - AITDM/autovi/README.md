# AutoVI – federated unsupervised anomaly detection

Stage 1 of the AITDM project on the [AutoVI](https://auto-vi.utc.fr/) dataset (Renault Group):
**one federated client per category**, FedAvg, compared with a centralised and a local-only
baseline. Built on the starter kit in this repository: the Flower scaffolding (strategies,
differential privacy, attackers, run scripts) is reused, the model, the data preparation and
the evaluation are new, because AutoVI is not a classification task.

| | Starter kit (`fl_app/`) | AutoVI (`fl_app_autovi/`, `autovi/`) |
|---|---|---|
| Task | supervised classification | unsupervised anomaly detection: training data is defect-free only |
| Model | MONAI CNN / MLP, cross-entropy | STFPM: frozen ImageNet ResNet-18 teacher + trained student |
| What is federated | the whole classifier | the student only (2.8 M parameters, 11 MB) |
| Non-IID | label skew (Dirichlet) | each client photographs a different object, with different amounts of data |
| Metrics | accuracy, balanced accuracy, F1 | image-level AUROC, pixel-level AU-sPRO (official script) |

## How it works

* **Clients.** Client *i* owns the good training images of one category (`client_<i>.npz`).
  It never sees defects and never shares images.
* **Model.** A frozen ResNet-18 pre-trained on ImageNet (the *teacher*, identical and public for
  everyone) and a randomly initialised copy (the *student*). The student learns to reproduce the
  teacher's feature maps on good images. On a defect it fails to, and the disagreement per pixel
  is the anomaly map; its maximum is the image's anomaly score.
* **One round of FedAvg.** The server sends the global student; every client trains it for
  `local-epochs` on its own images; the server averages the returned students, weighted by the
  number of training images.
* **Evaluation.** During training every client reports the loss of the global student on its
  held-out good images (it has no defects to test on). After the last round the server computes
  the image-level AUROC on every category's official test set. `autovi/evaluate.py` then adds the
  pixel-level AU-sPRO with the [official evaluation code](https://github.com/phcarval/autovi_evaluation_code).
* **Baselines** (`autovi/baselines.py`), same student, optimiser, number of epochs and evaluation:
  *centralised* = one student on the pooled images; *local* = one student per client, no
  collaboration (the usual one-model-per-category protocol, e.g. of PatchCore).

## Files

```text
fl_app_autovi/
├── pyproject.toml          run configuration (rounds, local epochs, lr, strategy, DP, ...)
└── autovi_fl/
    ├── task.py             student / teacher, loss, anomaly maps, training, metrics (no Flower)
    ├── client_app.py       ClientApp: local training and validation
    └── server_app.py       ServerApp: strategy, DP, test evaluation after the last round
autovi/
├── prepare_data.py         AutoVI folders -> client_<i>.npz, test.npz, meta.json
├── evaluate.py             one evaluation for every kind of run -> results.json, example heat maps
├── baselines.py            centralised and local-only training
├── summarize.py            mean ± std tables over the seeds
├── run_experiments.sh      everything above, for all seeds
└── wsl_setup.sh, wsl_sync.sh, wsl_run.sh    Windows + WSL2 helpers
```

## Protocol (agree on it in the team: all numbers depend on it)

| Setting | Value |
|---|---|
| Input | RGB, resized to 384 × 384 (no crop), ImageNet mean / std |
| Validation | 15 % of each client's good training images, random, split seed 0 |
| Student | ResNet-18 layers 1–3, SGD lr 0.4, momentum 0.9, weight decay 1e-4, batch 32 (STFPM paper) |
| Federated | FedAvg, 3 clients, all clients every round, 50 rounds × 2 local epochs |
| Baselines | 100 epochs |
| Seeds | 1, 2, 3 (reported as mean ± std) |
| Test set | the official one, used once per run, never for tuning |

## Running it

Windows (WSL2 Ubuntu, NVIDIA GPU). Expected folders next to this repository:
`Dataset(s)/<category>/...` and `autovi_evaluation_code/`.

```bash
# once: uv + Python 3.12 + PyTorch (CUDA) + Flower, in ~/aitdm inside WSL
wsl -d Ubuntu -- bash "/mnt/c/<...>/Project/AITDM_FLOWER_PROJECT_TUTORIAL/autovi/wsl_setup.sh"

# the whole experiment; results are copied to <Project>/results/autovi
wsl -d Ubuntu -- bash "/mnt/c/<...>/Project/AITDM_FLOWER_PROJECT_TUTORIAL/autovi/wsl_run.sh"
```

Linux / inside WSL / fep, from the repository root with the venv active:

```bash
pip install -e fl_app_autovi
python autovi/prepare_data.py --dataset-root /path/to/autovi \
    --categories tank_screw pipe_staple underbody_pipes --size 384

# one federated run
CLIENT_GPUS=1 APP_DIR=fl_app_autovi bash scripts/run_simulation.sh data/partitions/autovi_384 "seed=1"
python autovi/evaluate.py --run outputs/sim-autovi_384-<timestamp> --partitions data/partitions/autovi_384

# or everything (FedAvg + both baselines, seeds 1 2 3, table)
bash autovi/run_experiments.sh
```

Measured on an RTX 4060 Laptop (8 GB) under WSL2, 384 px: a federated round with 2 local epochs
takes about 10 s, a centralised epoch about 3 s, and the official AU-sPRO evaluation several
minutes per category (CPU). `NO_SPRO=1` skips the latter for quick tests.

**GPU memory.** `run_experiments.sh` trains one simulated client at a time (`CLIENT_GPUS=1`).
Three clients at once need more than 8 GB; under WSL that does not raise an error, it silently
spills into system memory and the run appears to hang. When you call `scripts/run_simulation.sh`
yourself, put `CLIENT_GPUS=1` in front as well.

`evaluate.py` looks for the dataset in `../autovi/<category>/` and the official code in
`../autovi_evaluation_code/` (set `AUTOVI_DATASET_ROOT` / `AUTOVI_EVAL_CODE` otherwise).

**The other three categories** (second teammate): the same commands with
`CATEGORIES="engine_wiring pipe_clip underbody_screw" PARTITIONS=$PWD/data/partitions/autovi_b_384 OUT_NAME=b`.

**Stage 2** works as in the tutorials, e.g. `STRATEGY=fedprox`, or
`EXTRA_CONFIG="dp-noise-multiplier=1.0"`, `EXTRA_CONFIG="malicious-clients='2'"`
([tutorial 4](../docs/04_aggregation_strategies.md)).

## A second non-IID split: Dirichlet over the category

`prepare_data.py --strategy dirichlet --alpha 0.3` deals the good training images to the clients
with the starter kit's `dirichlet_split()`, using the category as the label: clients then hold
different mixtures of the objects instead of one object each (one category per client is the
limit α → 0). The pooled training / validation images and the test set are identical to the
default split, so the centralised baseline is shared and only the federated runs are needed:

```bash
python autovi/prepare_data.py --dataset-root ../autovi --strategy dirichlet --alpha 0.3 \
    --out data/partitions/autovi_384_dirichlet0.3
PARTITIONS=$PWD/data/partitions/autovi_384_dirichlet0.3 OUT_NAME=dirichlet0.3 BASELINES=0 \
    bash autovi/run_experiments.sh
python autovi/summarize.py          # all experiments in outputs/autovi/
```

## Side experiment: Dirichlet label skew (supervised)

The main experiment has no label skew, because the training data has one label only. To use
the starter kit's Dirichlet split anyway, `autovi/labelled_from_test.py` builds a labelled
dataset from the **test** folders (classes `<category>/good` and `<category>/defective`), and
`autovi/run_supervised_dirichlet.sh` runs the unchanged starter-kit pipeline on it
(`split_data.py`, the `fl_app/` classifier, `experiments/`):

```bash
bash autovi/run_supervised_dirichlet.sh        # FedAvg Dirichlet(0.3), FedAvg IID, centralised; seed 1
```

It is a different task, to be reported separately: the classifier is trained on defects
(supervised), and the official test set is consumed (a random 20 % of it is the new test set),
so the numbers are not comparable with the unsupervised results above or with published
AutoVI results. Results: `outputs/experiments/autovi-sup-*`.

## Reading the results

`outputs/autovi/<experiment>/seed<s>/`:

* `results.json`: per category and mean: `auroc`, `ap`, `f1_max` (image level) and `auc_spro`
  for the false-positive limits 0.01 … 1 (pixel level, official script).
* `metrics.json`: the training history (per round: clients' training loss and validation loss).
* `examples_<category>.png`: anomaly maps of a few defective and good test images.
* `final_model.pt` (or `final_model_<category>.pt` for local-only): the student's weights.

Notes for the report:

* `f1_max` uses the threshold that is best on the test labels: an upper bound, not a
  deployable number.
* For `pipe_staple` the ground-truth mask is the whole image (a logical defect), so its
  AU-sPRO mostly says whether defective images get high scores everywhere.
* GPU training with mixed precision is not bit-reproducible: the same seed gives very close,
  not identical, numbers.
