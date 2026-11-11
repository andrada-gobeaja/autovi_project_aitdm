"""Flower ClientApp: what every inspection station (= one AutoVI category) runs
on its own images. Adapted from fl_app/aitdm_fl/client_app.py; the differences
are the model (STFPM student instead of a classifier) and the metrics."""

from collections import OrderedDict
from pathlib import Path

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from autovi_fl.task import (
    build_student,
    derive_seed,
    get_device,
    get_weights,
    load_client_data,
    set_seed,
    set_weights,
    validation_metrics,
)
from autovi_fl.task import train as train_fn

app = ClientApp()


def get_data_path(context: Context) -> Path:
    """Which partition does this client own?

    * Deployment: --node-config "data-path='/abs/path/client_0.npz'", or
      --node-config "partition-id=0 num-partitions=3"
    * Simulation: Flower sets partition-id automatically (0 ... num-supernodes-1).
    """
    if "data-path" in context.node_config:
        return Path(str(context.node_config["data-path"]))
    partition_id = int(context.node_config["partition-id"])
    return Path(str(context.run_config["partitions-dir"])) / f"client_{partition_id}.npz"


def is_malicious(context: Context) -> bool:
    """Is this client an attacker? (robustness experiments, see docs/04)"""
    if context.node_config.get("malicious", False):
        return True
    ids = str(context.run_config["malicious-clients"]).replace(" ", "")
    pid = context.node_config.get("partition-id")
    return pid is not None and ids != "" and str(pid) in ids.split(",")


def poison(global_state: dict, local_state: dict, attack: str) -> dict:
    """Model poisoning: replace the honest update with a harmful one
    ("sign-flip", "noise", or "none" = send the honest weights)."""
    if attack == "none":
        return local_state
    poisoned = {}
    for name, w_local in local_state.items():
        w_global = global_state[name]
        if not torch.is_floating_point(w_local):
            poisoned[name] = w_local
        elif attack == "sign-flip":
            poisoned[name] = w_global - 5.0 * (w_local - w_global)
        elif attack == "noise":
            poisoned[name] = torch.randn_like(w_local)
        else:
            raise ValueError(f"Unknown attack '{attack}'. Use 'sign-flip', 'noise' or 'none'.")
    return poisoned


def local_student(msg: Message) -> torch.nn.Module:
    """The global student received from the server."""
    student = build_student()
    set_weights(student, msg.content["arrays"].to_torch_state_dict())
    return student


def client_seed(msg: Message, context: Context) -> int:
    """Reproducible seed for this client in this round (run config 'seed')."""
    return derive_seed(int(context.run_config["seed"]), get_data_path(context).name,
                       int(msg.content["config"]["server-round"]))


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Receive the global student, train it on the local good images, send it back."""
    config = msg.content["config"]
    cfg = context.run_config
    seed = client_seed(msg, context)
    set_seed(seed)

    train_loader, _ = load_client_data(get_data_path(context), int(cfg["batch-size"]), seed=seed)
    student = local_student(msg)
    global_state = get_weights(student)

    train_loss = train_fn(
        student,
        train_loader,
        epochs=int(cfg["local-epochs"]),
        lr=float(config["lr"]),
        device=get_device(),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight-decay"]),
        proximal_mu=float(config.get("proximal-mu", 0.0)),  # set by FedProx
        amp=bool(cfg["amp"]),
    )

    # "num-examples" is used by the server to weight the aggregation (FedAvg)
    num_examples = len(train_loader.dataset)
    local_state = get_weights(student)
    if is_malicious(context):
        local_state = poison(global_state, local_state, str(cfg["attack"]))
        # Nothing stops a client from lying about its size: FedAvg trusts it
        num_examples *= int(cfg["attack-inflate-examples"])

    metrics = MetricRecord({
        "train_loss": train_loss,
        "num-examples": num_examples,
        "unit-weight": 1,  # equal weights, used when differential privacy is on
    })
    content = RecordDict({"arrays": ArrayRecord(OrderedDict(local_state)), "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the global student on this client's held-out GOOD images.

    A client has no labelled defects, so it cannot compute a detection rate. What
    it can measure is how well the global student imitates the teacher on its own
    normal data: a high loss means the global model does not fit this client."""
    cfg = context.run_config
    _, val_loader = load_client_data(get_data_path(context), int(cfg["batch-size"]))
    results = validation_metrics(local_student(msg), val_loader, get_device())

    metrics = MetricRecord({
        "eval_loss": results["loss"],
        "eval_score_mean": results["score_mean"],
        "eval_score_max": results["score_max"],
        "num-examples": len(val_loader.dataset),
        "unit-weight": 1,
    })
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
