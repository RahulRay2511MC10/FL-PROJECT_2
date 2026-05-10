# Adaptive TiFL: Tier-Based Federated Learning for System Heterogeneity Mitigation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Flower](https://img.shields.io/badge/framework-Flower-green.svg)](https://flower.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Group Information

| Field | Details |
|---|---|
| **Category** | Category 7 — System Heterogeneity |
| **Dataset** | MNIST |
| **Framework** | Flower (flwr) |
| **Paper 1** | TiFL: A Tier-Based Federated Learning System (Chai et al., HPDC 2020) |
| **Paper 2** | Hermes: An Efficient Federated Learning Framework for Heterogeneous Mobile Clients (Li et al., MobiCom 2021) |
| **YouTube** | [Project Demo Video](https://youtube.com/your-link-here) |

---

## Overview

This project implements **Adaptive TiFL** — a tier-based federated learning strategy that addresses system heterogeneity by grouping clients into performance tiers and rotating participation across rounds. This eliminates the straggler bottleneck present in standard FedAvg, where the slowest client determines the wall-clock time of every training round.

**Core idea:** Instead of waiting for all clients every round, only one tier (Fast / Medium / Slow) trains per round. Each tier gets a dedicated round, so all clients contribute while fast clients are never blocked by slow ones.

### Key Results

| α | Partitioning | Final Accuracy | Avg Round Time | Convergence Round |
|---|---|---|---|---|
| 0.01 | Extreme non-IID | ~82% | ~28s | Round 5 |
| 0.1  | Strong non-IID   | 94.28% | 28.67s | Round 3 |
| 0.5  | Moderate non-IID | ~96%   | ~28s   | Round 3 |
| 1.0  | Mild non-IID     | ~97%   | ~28s   | Round 2 |
| IID  | IID baseline     | ~98%   | ~28s   | Round 2 |
| FedAvg (α=0.1) | Baseline | ~92% | 58.84s | Round 7 |

**Speedup over FedAvg: 2.05× reduction in average round time**

---

## Repository Structure

```
fl-project/
├── README.md                   ← this file
├── requirements.txt            ← pinned dependencies
│
├── configs/                    ← YAML config for every experiment
│   ├── alpha_0.01.yaml
│   ├── alpha_0.1.yaml
│   ├── alpha_0.5.yaml
│   ├── alpha_1.0.yaml
│   ├── iid.yaml
│   └── fedavg_baseline.yaml
│
├── src/                        ← all source code
│   ├── server.py               ← Adaptive TiFL server strategy
│   ├── server_fedavg.py        ← FedAvg baseline server
│   ├── client.py               ← Flower client (--cid, --alpha flags)
│   ├── data.py                 ← Dirichlet + IID partitioning
│   ├── model.py                ← CNN architecture
│   ├── utils.py                ← utility functions
│   ├── run_all_alphas.py       ← master runner for all 5 experiments
│   ├── plot_results.py         ← single-run plots
│   ├── compare_results.py      ← FedAvg vs TiFL comparison plots
│   └── generate_table.py       ← summary results table
│
├── results/                    ← auto-generated after running experiments
│   ├── tifl/
│   │   ├── 0.01/
│   │   │   ├── accuracy.csv
│   │   │   ├── loss.csv
│   │   │   ├── round_time.csv
│   │   │   └── straggler_ratio.csv
│   │   ├── 0.1/
│   │   ├── 0.5/
│   │   ├── 1.0/
│   │   └── iid/
│   ├── fedavg/
│   │   ├── accuracy.csv
│   │   ├── loss.csv
│   │   └── round_time.csv
│   └── plots/
│       ├── fig1_acc_global.png
│       ├── fig2_loss_global.png
│       ├── fig3_round_time.png
│       ├── fig4_iid_vs_noniid.png
│       └── fig5_comparison.png
│
└── report/
    └── TiFL_IEEE_Report.pdf    ← final submitted report
```

---

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- macOS or Linux (Windows supported but not tested)
- ~2 GB free disk space (MNIST download + logs)

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/fl-project.git
cd fl-project
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Experiments

### Option A — Run All 5 α Values Automatically ✅ Recommended

Runs all five data partitioning configurations sequentially with no manual steps.

```bash
python3 src/run_all_alphas.py
```

Experiments run in order: `0.01 → 0.1 → 0.5 → 1.0 → iid`  
Each experiment: ~5–6 minutes | **Total: ~30 minutes**

Expected terminal output:
```
============================================================
  Experiment  : alpha = 0.01
  Output dir  : results/tifl/0.01
============================================================
  Server PID  : 12345
  Launched 10 clients. Waiting for completion...
  ✓ Completed : alpha = 0.01
```

---

### Option B — Run a Single Experiment Manually

Open two terminal windows from the project root.

**Terminal 1 — Start the Adaptive TiFL server:**
```bash
python3 src/server.py --alpha 0.1 --rounds 10 --out_dir results/tifl/0.1
```

Wait until you see:
```
Server starting | alpha=0.1 | rounds=10
INFO flwr: Started server process
INFO flwr: Waiting for connections...
```

**Terminal 2 — Launch all 10 clients:**
```bash
# macOS — opens each client in a new Terminal tab
bash src/start_clients.sh

# Or manually in background:
for i in {0..9}; do
    python3 src/client.py --cid $i --alpha 0.1 &
    sleep 1
done
```

---

### Option C — Run FedAvg Baseline

**Terminal 1:**
```bash
python3 src/server_fedavg.py
```

**Terminal 2:**
```bash
for i in {0..9}; do
    python3 src/client.py --cid $i --alpha 0.1 &
    sleep 1
done
```

Results saved to `results/fedavg/`.

---

## Verifying Results

```bash
# Check all alpha folders were created
ls results/tifl/
# Expected: 0.01  0.1  0.5  1.0  iid

# Check CSVs for one alpha
ls results/tifl/0.1/
# Expected: accuracy.csv  loss.csv  round_time.csv  straggler_ratio.csv

# Quick sanity check — final accuracy should be ~94%
tail -1 results/tifl/0.1/accuracy.csv
# Expected: 10,0.9428
```

---

## Generating Plots

After all experiments complete:

```bash
# Plots for a single alpha run
python3 src/plot_results.py
# → results/plots/accuracy.png, loss.png, round_time.png

# FedAvg vs TiFL comparison
python3 src/compare_results.py
# → results/comparison_plots/accuracy_comparison.png, time_comparison.png

# Summary results table
python3 src/generate_table.py
# → results/final_results_table.csv (printed to terminal)
```

---

## Configuration Details

### Client Tier Assignment

| Tier   | Client IDs  | Simulated Delay | Represents        |
|--------|-------------|-----------------|-------------------|
| Fast   | 0, 1, 2     | 0 seconds       | GPU workstation   |
| Medium | 3, 4, 5     | 3 seconds       | Modern laptop CPU |
| Slow   | 6, 7, 8, 9  | 6 seconds       | Edge / IoT device |

### Training Hyperparameters

| Parameter           | Value  |
|---------------------|--------|
| Number of clients   | 10     |
| Communication rounds| 10     |
| Local epochs        | 5      |
| Batch size          | 32     |
| Optimizer           | SGD    |
| Learning rate       | 0.01   |
| Momentum            | 0.9    |
| Clients per round   | 3 (one tier) |

### Data Partitioning (α)

| α    | Type                    | Description                            |
|------|-------------------------|----------------------------------------|
| 0.01 | Extreme non-IID         | Each client gets ≈ 1–2 digit classes   |
| 0.1  | Strong non-IID          | High label skew across clients         |
| 0.5  | Moderate non-IID        | Moderate label imbalance               |
| 1.0  | Mild non-IID            | Near-balanced with slight skew         |
| IID  | IID                     | Equal random split across all clients  |

### Model Architecture (CNN on MNIST)

```
Input (1×28×28)
  → Conv2D(1→32, kernel=3) → ReLU → MaxPool2D(2×2)
  → Conv2D(32→64, kernel=3) → ReLU → MaxPool2D(2×2)
  → Flatten(1600)
  → Linear(1600→128) → ReLU → Dropout(0.5)
  → Linear(128→10)
  → Output (10 classes)

Total parameters: 206,922 (~0.83 MB per model transfer)
```

---

## Troubleshooting

**Port 8080 already in use:**
```bash
# macOS
lsof -ti:8080 | xargs kill -9

# Linux
fuser -k 8080/tcp
```

**`ModuleNotFoundError: No module named 'flwr'`:**
```bash
pip install -r requirements.txt
```

**Accuracy stays at 0.0:**  
Ensure `server.py` has both fixes applied:
```python
evaluate_metrics_aggregation_fn=weighted_average  # Fix 1
min_fit_clients=3                                  # Fix 2 (not 10)
```

**Experiment hangs and never finishes:**  
Check the server log for errors:
```bash
cat results/tifl/0.1/server.log
```
Most likely cause: fewer than 10 clients connected. Restart and ensure all 10 client processes launch successfully.

---

## References

1. B. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," *AISTATS*, 2017.
2. Z. Chai et al., "TiFL: A Tier-Based Federated Learning System," *HPDC*, 2020.
3. C. Li et al., "Hermes: An Efficient Federated Learning Framework for Heterogeneous Mobile Clients," *MobiCom*, 2021.
4. D. J. Beutel et al., "Flower: A Friendly Federated Learning Research Framework," *arXiv:2007.14390*, 2020.
5. Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition," *Proc. IEEE*, 1998.
