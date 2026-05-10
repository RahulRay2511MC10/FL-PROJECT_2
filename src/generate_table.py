import pandas as pd

fedavg_acc = pd.read_csv(
    "results/fedavg/accuracy.csv"
)

fedavg_time = pd.read_csv(
    "results/fedavg/round_time.csv"
)

tifl_acc = pd.read_csv(
    "results/metrics/accuracy.csv"
)

tifl_time = pd.read_csv(
    "results/metrics/round_time.csv"
)

summary = pd.DataFrame({
    "Method": [
        "FedAvg",
        "Adaptive TiFL"
    ],

    "Final Accuracy": [
        fedavg_acc["accuracy"].iloc[-1],
        tifl_acc["accuracy"].iloc[-1]
    ],

    "Average Round Time": [
        fedavg_time["time"].mean(),
        tifl_time["time"].mean()
    ]
})

print("\nExperimental Results:\n")

print(summary)

summary.to_csv(
    "results/final_results_table.csv",
    index=False
)

print("\nSaved to:")
print("results/final_results_table.csv")