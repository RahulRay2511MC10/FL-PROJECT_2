"""
run_all_alphas.py
=================
Runs 5 federated learning experiments sequentially:
    alpha ∈ {0.01, 0.1, 0.5, 1.0, iid}

Usage (from your project root, e.g. fl-project/):
    python3 src/run_all_alphas.py

Results saved to:
    results/tifl/0.01/   accuracy.csv, loss.csv, round_time.csv, straggler_ratio.csv
    results/tifl/0.1/    ...
    results/tifl/0.5/    ...
    results/tifl/1.0/    ...
    results/tifl/iid/    ...
"""

import subprocess
import time
import os
import sys

# ── Config ────────────────────────────────────────────────────────────────────
ALPHAS      = ["0.01", "0.1", "0.5", "1.0", "iid"]
NUM_CLIENTS = 10
NUM_ROUNDS  = 10

# Paths relative to project root (works whether you call from root or src/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "src")

SERVER_SCRIPT = os.path.join(SRC, "server.py")
CLIENT_SCRIPT = os.path.join(SRC, "client.py")

WAIT_FOR_SERVER = 5    # seconds to let server start before launching clients
WAIT_AFTER_RUN  = 3    # seconds between experiments


# ── Helpers ───────────────────────────────────────────────────────────────────
def kill_port(port=8080):
    """Free port 8080 if something is already using it."""
    if sys.platform == "darwin":   # macOS
        subprocess.run(f"lsof -ti:{port} | xargs kill -9",
                       shell=True, stderr=subprocess.DEVNULL)
    else:                          # Linux
        subprocess.run(f"fuser -k {port}/tcp",
                       shell=True, stderr=subprocess.DEVNULL)
    time.sleep(1)


def run_experiment(alpha: str) -> bool:
    out_dir = os.path.join(ROOT, "results", "tifl", alpha)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Experiment  : alpha = {alpha}")
    print(f"  Output dir  : {out_dir}")
    print(f"{'='*60}")

    kill_port(8080)

    # ── Start server ──────────────────────────────────────────────
    server_log = open(os.path.join(out_dir, "server.log"), "w")
    server_proc = subprocess.Popen(
        [sys.executable, SERVER_SCRIPT,
         "--alpha",   alpha,
         "--rounds",  str(NUM_ROUNDS),
         "--out_dir", out_dir],
        stdout=server_log,
        stderr=subprocess.STDOUT,
        cwd=ROOT,                  # run from project root so imports work
    )
    print(f"  Server PID  : {server_proc.pid}")
    time.sleep(WAIT_FOR_SERVER)

    # ── Start clients ─────────────────────────────────────────────
    client_procs = []
    for cid in range(NUM_CLIENTS):
        log = open(os.path.join(out_dir, f"client_{cid}.log"), "w")
        proc = subprocess.Popen(
            [sys.executable, CLIENT_SCRIPT,
             "--cid",   str(cid),
             "--alpha", alpha],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
        client_procs.append(proc)

    print(f"  Launched {NUM_CLIENTS} clients. Waiting for completion...")

    # ── Wait for server to finish ─────────────────────────────────
    try:
        server_proc.wait(timeout=600)   # 10-minute max per experiment
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] alpha={alpha} — killing all processes.")
        server_proc.kill()
        for p in client_procs:
            p.kill()
        return False
    finally:
        server_log.close()

    # Clean up lingering clients
    for p in client_procs:
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()

    time.sleep(WAIT_AFTER_RUN)
    print(f"  ✓ Completed : alpha = {alpha}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Multi-Alpha FL Runner")
    print(f"  Alphas  : {ALPHAS}")
    print(f"  Clients : {NUM_CLIENTS}   Rounds : {NUM_ROUNDS}")
    print(f"  Root    : {ROOT}")
    print("=" * 60)

    summary = []
    for alpha in ALPHAS:
        ok = run_experiment(alpha)
        summary.append((alpha, "✓ OK" if ok else "✗ FAILED"))

    print("\n" + "=" * 60)
    print("  ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    for alpha, status in summary:
        print(f"  alpha = {alpha:>6}  →  {status}")
    print(f"\nResults in: {os.path.join(ROOT, 'results', 'tifl', '<alpha>')}/")
    print("Files: accuracy.csv  loss.csv  round_time.csv  straggler_ratio.csv")


if __name__ == "__main__":
    main()
