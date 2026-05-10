import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs(
    "results/plots",
    exist_ok=True
)

# -----------------------------
# Accuracy Plot
# -----------------------------

accuracy_df = pd.read_csv(
    "results/metrics/accuracy.csv"
)

plt.figure(figsize=(6, 4))

plt.plot(
    accuracy_df["round"],
    accuracy_df["accuracy"],
    marker="o"
)

plt.xlabel("Communication Rounds")
plt.ylabel("Accuracy")
plt.title("Global Accuracy vs Rounds")

plt.grid(True)

plt.savefig(
    "results/plots/accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# Loss Plot
# -----------------------------

loss_df = pd.read_csv(
    "results/metrics/loss.csv"
)

plt.figure(figsize=(6, 4))

plt.plot(
    loss_df["round"],
    loss_df["loss"],
    marker="o"
)

plt.xlabel("Communication Rounds")
plt.ylabel("Loss")
plt.title("Global Loss vs Rounds")

plt.grid(True)

plt.savefig(
    "results/plots/loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------
# Round Time Plot
# -----------------------------

time_df = pd.read_csv(
    "results/metrics/round_time.csv"
)

plt.figure(figsize=(6, 4))

plt.plot(
    time_df["round"],
    time_df["time"],
    marker="o"
)

plt.xlabel("Communication Rounds")
plt.ylabel("Time (seconds)")
plt.title("Round Time vs Rounds")

plt.grid(True)

plt.savefig(
    "results/plots/round_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nPlots saved successfully!")