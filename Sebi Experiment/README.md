# Sebi Experiment – federated STFPM on tank_screw, pipe_staple, underbody_pipes

Federated unsupervised anomaly detection on AutoVI with Flower: one student–teacher model
(STFPM, ResNet-18), FedAvg with 3 clients, compared with centralised and local-only training.

## Contents

```text
Cod Sebi - AITDM/
├── fl_app_autovi/          the Flower app (model, client, server, run configuration)
├── autovi/                 data preparation, evaluation, baselines, summary, run scripts
│   └── README.md           how it works, the protocol, how to run it
└── starter_kit_changes/    the one change made to the course starter kit (scripts/_common.sh)
figures/
├── fedavg/                 FedAvg, one category per client
├── dirichlet0.3-fedavg/    FedAvg, Dirichlet(0.3) split over the categories
├── centralised/            one model on the pooled images of the three categories
├── local/                  one model per category, no collaboration
├── centralised-all6/       one centralised model for all six categories
└── splits/                 which client holds what, for the Dirichlet splits
```

Every `examples_<category>.png` shows test images (top) and the model's anomaly maps (bottom)
for seed 1 of that experiment.

## Running the code

The code builds on the course starter kit
([AITDM_FLOWER_PROJECT_TUTORIAL](https://github.com/Chiru-CatalinMihail/AITDM_FLOWER_PROJECT_TUTORIAL)):
copy `fl_app_autovi/` and `autovi/` into the root of a clone of it, apply
`starter_kit_changes/_common.sh.patch` (`git apply`), then follow `autovi/README.md`.
The dataset and the official evaluation code
([autovi_evaluation_code](https://github.com/phcarval/autovi_evaluation_code)) are not included.

## Data

The figures contain images of the AutoVI dataset, © 2023-2024 Renault Group, licensed
CC BY-NC-SA 4.0: Carvalho, P., Lafou, M., Durupt, A., Leblanc, A., & Grandvalet, Y. (2024).
*The Automotive Visual Inspection Dataset (AutoVI)*. https://doi.org/10.5281/zenodo.10459003
