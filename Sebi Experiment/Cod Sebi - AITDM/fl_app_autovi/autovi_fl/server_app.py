"""Flower ServerApp: orchestrates the rounds and aggregates the clients' students.
Adapted from fl_app/aitdm_fl/server_app.py: the strategies and differential
privacy are unchanged, the model and the evaluation are those of AutoVI."""

import json
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import (
    Grid,
    NeighboringRelation,
    PrivacyConfig,
    RdpAccountant,
    SamplingMethod,
    ServerApp,
)
from flwr.serverapp.strategy import (
    DifferentialPrivacyServerSideFixedClipping,
    FedAdam,
    FedAvg,
    FedAvgM,
    FedMedian,
    FedProx,
    FedTrimmedAvg,
    FedYogi,
    Krum,
    QFedAvg,
)

from autovi_fl.task import (
    build_student,
    evaluate_test,
    get_device,
    get_weights,
    load_meta,
    load_test_data,
    set_seed,
    set_weights,
)

app = ServerApp()

STRATEGIES = ["fedavg", "fedprox", "fedavgm", "fedadam", "fedyogi",
              "fedmedian", "fedtrimmedavg", "krum", "qfedavg"]


def build_strategy(context: Context):
    """Create the aggregation strategy selected with run config 'strategy'.
    See docs/04_aggregation_strategies.md for what each one does."""
    cfg = context.run_config
    common = dict(
        fraction_train=float(cfg["fraction-train"]),
        fraction_evaluate=float(cfg["fraction-evaluate"]),
        # Without these the server starts as soon as 2 clients are connected!
        min_train_nodes=int(cfg["min-nodes"]),
        min_evaluate_nodes=int(cfg["min-nodes"]),
        min_available_nodes=int(cfg["min-nodes"]),
    )
    if float(cfg["dp-noise-multiplier"]) > 0:
        # DP-FedAvg averages client updates with EQUAL weights
        common["weighted_by_key"] = "unit-weight"
    name = str(cfg["strategy"]).lower()

    if name == "fedavg":
        strategy = FedAvg(**common)
    elif name == "fedprox":
        strategy = FedProx(proximal_mu=float(cfg["proximal-mu"]), **common)
    elif name == "fedavgm":
        strategy = FedAvgM(server_momentum=float(cfg["server-momentum"]), **common)
    elif name == "fedadam":
        strategy = FedAdam(eta=float(cfg["server-lr"]), **common)
    elif name == "fedyogi":
        strategy = FedYogi(eta=float(cfg["server-lr"]), **common)
    elif name == "fedmedian":
        strategy = FedMedian(**common)
    elif name == "fedtrimmedavg":
        strategy = FedTrimmedAvg(beta=float(cfg["trim-beta"]), **common)
    elif name == "krum":
        strategy = Krum(num_malicious_nodes=int(cfg["num-malicious"]), **common)
    elif name == "qfedavg":
        strategy = QFedAvg(client_learning_rate=float(cfg["qfedavg-client-lr"]),
                           q=float(cfg["qfedavg-q"]), **common)
    else:
        raise ValueError(f"Unknown strategy '{name}'. Use one of: {', '.join(STRATEGIES)}")

    noise = float(cfg["dp-noise-multiplier"])
    if noise > 0:
        strategy = DifferentialPrivacyServerSideFixedClipping(
            strategy,
            noise_multiplier=noise,
            clipping_norm=float(cfg["dp-clipping-norm"]),
            num_sampled_clients=int(cfg["min-nodes"]),
            accountant=build_accountant(strategy, cfg),
        )
    return strategy


def build_accountant(strategy, cfg):
    """Track the privacy budget (epsilon at delta = dp-delta) over the rounds.
    Flower's accountant only supports strategies that aggregate like FedAvg."""
    if not (isinstance(strategy, FedAvg)
            and type(strategy).aggregate_train is FedAvg.aggregate_train):
        print(f"[DP] No epsilon accounting for '{cfg['strategy']}' (FedAvg-style "
              "aggregation only). Noise is still added.")
        return None
    return RdpAccountant(PrivacyConfig(
        target_delta=float(cfg["dp-delta"]),
        population_size=int(cfg["min-nodes"]),
        neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
        sampling_method=SamplingMethod.NO_AMPLIFICATION,
    ))


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    partitions_dir = str(cfg["partitions-dir"])
    if not partitions_dir:
        raise ValueError(
            "run config 'partitions-dir' is empty. Pass it with "
            "--run-config \"partitions-dir='/abs/path/to/data/partitions/<name>'\""
        )

    # Same seed -> same initial student, same client shuffling, same DP noise
    set_seed(int(cfg["seed"]))
    global_student = build_student()

    meta = load_meta(partitions_dir)
    device = get_device()
    num_rounds = int(cfg["num-server-rounds"])
    test_every_round = bool(cfg["test-every-round"])
    test_data = {}  # loaded on first use (470 MB at 384 px)

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
        """Image-level detection of the aggregated student on every category's
        test set (good + defective images; the only place labels are used).

        By default only once, after the last round: the test set is for the final
        numbers. Follow the training with the clients' validation loss instead."""
        if server_round != num_rounds and not (test_every_round and server_round > 0):
            return None
        if not test_data:
            test_data.update(load_test_data(partitions_dir))
        student = build_student()
        set_weights(student, arrays.to_torch_state_dict())
        results = evaluate_test(lambda category: student, partitions_dir, device, test=test_data)
        print(f"[round {server_round}] image-level AUROC: "
              + ", ".join(f"{c}={results[c]['auroc']:.4f}" for c in [*meta["categories"], "mean"]))
        return MetricRecord({f"{metric}_{category}": value
                             for category, values in results.items()
                             for metric, value in values.items()})

    strategy = build_strategy(context)
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(get_weights(global_student)),
        train_config=ConfigRecord({"lr": float(cfg["learning-rate"])}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    save_dir = str(cfg["save-dir"])
    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(result.arrays.to_torch_state_dict(), out / "final_model.pt")
        history = {
            "run_config": {**dict(cfg), "mode": "federated"},
            "train_metrics_clientapp": _to_dict(result.train_metrics_clientapp),
            "evaluate_metrics_clientapp": _to_dict(result.evaluate_metrics_clientapp),
            "evaluate_metrics_serverapp": _to_dict(result.evaluate_metrics_serverapp),
        }
        spent = strategy.privacy_spent() if hasattr(strategy, "privacy_spent") else None
        if spent is not None:
            history["privacy"] = {"epsilon": spent.epsilon, "delta": spent.delta}
        (out / "metrics.json").write_text(json.dumps(history, indent=2))
        print(f"\nSaved final model and metrics to {out}")


def _to_dict(metrics_per_round) -> dict:
    """{round: MetricRecord} -> {round: {name: value}} for JSON."""
    return {int(rnd): dict(rec) for rnd, rec in metrics_per_round.items()}
