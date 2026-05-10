import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics

import csv
import os
import time
import argparse
from typing import List, Tuple

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--alpha",   type=str, default="0.1",
                    help="Dirichlet alpha or 'iid'")
parser.add_argument("--rounds",  type=int, default=10)
parser.add_argument("--out_dir", type=str, default="results/tifl/0.1",
                    help="Directory to write CSV metrics into")
args = parser.parse_args()

ALPHA   = args.alpha
ROUNDS  = args.rounds
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

# ── Output files ──────────────────────────────────────────────────────────────
accuracy_file  = os.path.join(OUT_DIR, "accuracy.csv")
loss_file      = os.path.join(OUT_DIR, "loss.csv")
time_file      = os.path.join(OUT_DIR, "round_time.csv")
straggler_file = os.path.join(OUT_DIR, "straggler_ratio.csv")

for fpath, header in [
    (accuracy_file,  ["round", "accuracy"]),
    (loss_file,      ["round", "loss"]),
    (time_file,      ["round", "time"]),
    (straggler_file, ["round", "straggler_ratio"]),
]:
    with open(fpath, "w", newline="") as f:   # always start fresh
        csv.writer(f).writerow(header)


# ── Metric aggregation ────────────────────────────────────────────────────────
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {"accuracy": 0.0}
    return {"accuracy": sum(n * m.get("accuracy", 0.0)
                            for n, m in metrics) / total}


# ── Strategy ──────────────────────────────────────────────────────────────────
class AdaptiveTiFLStrategy(FedAvg):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_start_time = None
        self.straggler_ratio  = 0.0

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_start_time = time.time()

        available = list(client_manager.all().values())

        # Rotate tiers every 3 rounds: FAST → MEDIUM → SLOW
        tier_idx = (server_round - 1) % 3
        if tier_idx == 0:
            selected, tier_name = available[:3],  "FAST"
        elif tier_idx == 1:
            selected, tier_name = available[3:6], "MEDIUM"
        else:
            selected, tier_name = available[6:],  "SLOW"

        # Straggler ratio = fraction of selected clients in slow tier
        slow_set = set(id(c) for c in available[6:])
        self.straggler_ratio = (
            sum(1 for c in selected if id(c) in slow_set) / len(selected)
            if selected else 0.0
        )

        print(f"\n[Round {server_round}] Tier: {tier_name} "
              f"| alpha={ALPHA} | straggler_ratio={self.straggler_ratio:.2f}")

        fit_ins = fl.common.FitIns(parameters, {})
        return [(c, fit_ins) for c in selected]

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if aggregated is not None:
            loss, metrics = aggregated
            accuracy   = metrics.get("accuracy", 0.0)
            round_time = time.time() - self.round_start_time

            print(f"  Accuracy  : {accuracy:.4f}")
            print(f"  Loss      : {loss:.4f}")
            print(f"  Round Time: {round_time:.2f}s")

            with open(accuracy_file,  "a", newline="") as f:
                csv.writer(f).writerow([server_round, accuracy])
            with open(loss_file,      "a", newline="") as f:
                csv.writer(f).writerow([server_round, loss])
            with open(time_file,      "a", newline="") as f:
                csv.writer(f).writerow([server_round, round_time])
            with open(straggler_file, "a", newline="") as f:
                csv.writer(f).writerow([server_round, self.straggler_ratio])

        return aggregated


strategy = AdaptiveTiFLStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,            # one tier = 3 clients
    min_evaluate_clients=3,
    min_available_clients=10,     # wait for all 10 before starting
    evaluate_metrics_aggregation_fn=weighted_average,
)

print(f"\nServer starting | alpha={ALPHA} | rounds={ROUNDS} | out={OUT_DIR}")
fl.server.start_server(
    server_address="0.0.0.0:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=ROUNDS),
)
