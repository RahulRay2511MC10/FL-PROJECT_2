import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs(
    "results/comparison_plots",
    exist_ok=True
)

# -----------------------------
# Accuracy Comparison
# -----------------------------

fedavg_acc = pd.read_csv(
    "results/fedavg/accuracy.csv"
)

tifl_acc = pd.read_csv(
    "results/metrics/accuracy.csv"
)

plt.figure(figsize=(6,4))

plt.plot(
    fedavg_acc["round"],
    fedavg_acc["accuracy"],
    marker="o",
    label="FedAvg"
)

plt.plot(
    tifl_acc["round"],
    tifl_acc["accuracy"],
    marker="o",
    label="Adaptive TiFL"
)

plt.xlabel("Rounds")
plt.ylabel("Accuracy")

plt.title("FedAvg vs Adaptive TiFL")

plt.legend()
plt.grid(True)

plt.savefig(
    "results/comparison_plots/accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# Time Comparison
# -----------------------------

fedavg_time = pd.read_csv(
    "results/fedavg/round_time.csv"
)

tifl_time = pd.read_csv(
    "results/metrics/round_time.csv"
)

plt.figure(figsize=(6,4))

plt.plot(
    fedavg_time["round"],
    fedavg_time["time"],
    marker="o",
    label="FedAvg"
)

plt.plot(
    tifl_time["round"],
    tifl_time["time"],
    marker="o",
    label="Adaptive TiFL"
)

plt.xlabel("Rounds")
plt.ylabel("Time (seconds)")

plt.title("Round Time Comparison")

plt.legend()
plt.grid(True)

plt.savefig(
    "results/comparison_plots/time_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nComparison plots generated!")