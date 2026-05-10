from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
])


def dirichlet_partition(dataset, num_clients, alpha):
    """Partition dataset using Dirichlet(alpha) distribution — non-IID."""
    labels = np.array(dataset.targets)
    num_classes = 10
    client_indices = [[] for _ in range(num_clients)]

    np.random.seed(42)          # reproducible across all alpha runs
    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
        proportions = proportions / proportions.sum()
        split_points = (
            np.cumsum(proportions) * len(class_indices)
        ).astype(int)[:-1]
        split_indices = np.split(class_indices, split_points)
        for i in range(num_clients):
            client_indices[i].extend(split_indices[i])

    return [Subset(dataset, idxs) for idxs in client_indices]


def iid_partition(dataset, num_clients):
    """Partition dataset equally and randomly — IID baseline."""
    np.random.seed(42)
    indices   = np.random.permutation(len(dataset))
    shard_size = len(dataset) // num_clients
    return [
        Subset(dataset, indices[i * shard_size:(i + 1) * shard_size])
        for i in range(num_clients)
    ]


def load_datasets(num_clients=10, alpha=0.1):
    """
    alpha: float (e.g. 0.01, 0.1, 0.5, 1.0) → Dirichlet non-IID
           string "iid"                        → equal random IID split
    """
    train_dataset = datasets.MNIST(
        root="./data", train=True,  download=True, transform=transform)
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform)

    # Handle both float and string inputs
    if str(alpha).lower() == "iid":
        client_datasets = iid_partition(train_dataset, num_clients)
    else:
        client_datasets = dirichlet_partition(
            train_dataset, num_clients, float(alpha))

    return client_datasets, test_dataset
