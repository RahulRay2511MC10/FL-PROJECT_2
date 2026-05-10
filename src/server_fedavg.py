import flwr as fl
from flwr.server.strategy import FedAvg

import csv
import os
import time

os.makedirs(
    "results/fedavg",
    exist_ok=True
)

accuracy_file = "results/fedavg/accuracy.csv"
loss_file = "results/fedavg/loss.csv"
time_file = "results/fedavg/round_time.csv"
from flwr.common import Metrics
from typing import List, Tuple

def weighted_average(
    metrics: List[Tuple[int, Metrics]]
) -> Metrics:

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    if total_examples == 0:
        return {"accuracy": 0.0}

    weighted_acc = sum(
        num_examples * m.get("accuracy", 0.0)
        for num_examples, m in metrics
    )

    return {
        "accuracy":
        weighted_acc / total_examples
    }
for file_path, header in [
    (accuracy_file, ["round", "accuracy"]),
    (loss_file, ["round", "loss"]),
    (time_file, ["round", "time"]),
]:
    if not os.path.exists(file_path):
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


class FedAvgStrategy(FedAvg):

    

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.round_start_time = None

    def configure_fit(
        self,
        server_round,
        parameters,
        client_manager,
    ):

        self.round_start_time = time.time()

        return super().configure_fit(
            server_round,
            parameters,
            client_manager,
        )

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):

        aggregated = super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )

        if aggregated is not None:

            loss, metrics = aggregated

            accuracy = metrics.get(
                "accuracy",
                0.0
            )

            round_time = (
                time.time()
                - self.round_start_time
            )

            with open(
                accuracy_file,
                "a",
                newline=""
            ) as f:
                csv.writer(f).writerow([
                    server_round,
                    accuracy
                ])

            with open(
                loss_file,
                "a",
                newline=""
            ) as f:
                csv.writer(f).writerow([
                    server_round,
                    loss
                ])

            with open(
                time_file,
                "a",
                newline=""
            ) as f:
                csv.writer(f).writerow([
                    server_round,
                    round_time
                ])

        return aggregated


strategy = FedAvg(
    fraction_fit=1.0,
    min_fit_clients=10,
    min_available_clients=10,
    min_evaluate_clients=10,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=
    weighted_average,
)
fl.server.start_server(
    server_address="0.0.0.0:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(
        num_rounds=10
    ),
)