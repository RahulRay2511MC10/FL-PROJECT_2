import flwr as fl
import torch
import argparse
import numpy as np
import time
from torch.utils.data import DataLoader

from model import CNN
from data import load_datasets

parser = argparse.ArgumentParser()
parser.add_argument("--cid",   type=int,   required=True)
parser.add_argument("--alpha", type=str,   default="0.1",
                    help="Dirichlet alpha value, or 'iid' for IID partitioning")
args = parser.parse_args()

cid   = args.cid
ALPHA = args.alpha          # now a string: "0.01", "0.1", "0.5", "1.0", "iid"

NUM_CLIENTS = 10
CLIENT_DELAYS = {
    0: 0,  1: 0,  2: 0,   # Fast tier
    3: 3,  4: 3,  5: 3,   # Medium tier
    6: 6,  7: 6,  8: 6,  9: 6,  # Slow tier
}
delay  = CLIENT_DELAYS[cid]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Convert alpha for data loader ─────────────────────────────────────────────
# load_datasets accepts float alpha or the string "iid"
alpha_val = ALPHA  # keep as string; data.py will handle "iid" vs float

client_datasets, test_dataset = load_datasets(
    num_clients=NUM_CLIENTS,
    alpha=alpha_val,
)

train_dataset = client_datasets[cid]

# Print label distribution
labels = [label for _, label in train_dataset]
unique, counts = np.unique(labels, return_counts=True)
print(f"\nClient {cid} | alpha={ALPHA} | label distribution:")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c}")

model      = CNN().to(DEVICE)
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
testloader  = DataLoader(test_dataset,  batch_size=32)
criterion  = torch.nn.CrossEntropyLoss()
optimizer  = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train():
    model.train()
    print(f"\nClient {cid} sleeping {delay}s (tier simulation)...")
    time.sleep(delay)
    for epoch in range(5):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def test():
    model.eval()
    correct = total = loss_total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss_total += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss_total, correct / total


class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = {k: torch.tensor(v)
                      for k, v in zip(model.state_dict().keys(), parameters)}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train()
        return self.get_parameters(config), len(train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test()
        return float(loss), len(test_dataset), {"accuracy": float(accuracy)}


fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
