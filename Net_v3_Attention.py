import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ScalarTokenAttention(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, hidden_dim: int = 32, num_heads: int = 4):
        super().__init__()
        if z_dim < 2 or hidden_dim % num_heads:
            raise ValueError(
                "Need at least two auxiliary tokens and divisible head width")
        self.z_dim = z_dim
        self.num_heads = num_heads
        self.query = nn.Linear(x_dim, hidden_dim)
        self.scalar_embedding = nn.Linear(1, hidden_dim)
        self.feature_identity = nn.Embedding(z_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True)
        self.output = nn.Linear(hidden_dim, x_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor, return_weights: bool = False):
        if x.ndim != 2 or z.ndim != 2 or x.shape[0] != z.shape[0] or z.shape[1] != self.z_dim:
            raise ValueError(
                f"Expected x=(batch, features), z=(batch, {self.z_dim})")
        q = self.query(x).unsqueeze(1)  # (batch, 1, hidden_dim)
        ids = torch.arange(self.z_dim, device=z.device)
        tokens = self.scalar_embedding(
            z.unsqueeze(-1)) + self.feature_identity(ids)
        # Per head, softmax runs over 3 thermal tokens or 2 uncertainty tokens.
        context, weights = self.attention(
            q, tokens, tokens,
            need_weights=return_weights,
            average_attn_weights=False,
        )
        result = self.output(context.squeeze(1))
        return (result, weights) if return_weights else result


class BPPointNet(nn.Module):
    def __init__(self, input_dim: int = 20, output_dim: int = 1,
                 z1_dim: int = 3, z2_dim: int = 2, n1: int = 24, n2: int = 6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, n1)
        self.fc2 = nn.Linear(n1, n2)
        self.fc3 = nn.Linear(n2, output_dim)
        self.relu = nn.ReLU()
        self.thermal1 = ScalarTokenAttention(n1, z1_dim)
        self.thermal2 = ScalarTokenAttention(n2, z1_dim)
        self.uncertainty1 = ScalarTokenAttention(n1, z2_dim)
        self.uncertainty2 = ScalarTokenAttention(n2, z2_dim)

    def forward(self, x: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor,
                return_weights: bool = False):
        h1 = self.relu(self.fc1(x))
        if return_weights:
            t1, wt1 = self.thermal1(h1, z1, True)
            u1, wu1 = self.uncertainty1(h1, z2, True)
        else:
            t1 = self.thermal1(h1, z1)
            u1 = self.uncertainty1(h1, z2)
        h1 = h1 + t1 + u1

        h2 = self.relu(self.fc2(h1))
        if return_weights:
            t2, wt2 = self.thermal2(h2, z1, True)
            u2, wu2 = self.uncertainty2(h2, z2, True)
        else:
            t2 = self.thermal2(h2, z1)
            u2 = self.uncertainty2(h2, z2)
        prediction = self.fc3(h2 + t2 + u2)
        if return_weights:
            return prediction, {"thermal1": wt1, "uncertainty1": wu1,
                                "thermal2": wt2, "uncertainty2": wu2}
        return prediction


def _read_training_data(folder: Path):
    x = np.loadtxt(folder / "merged_strains_.txt").T
    z1 = np.loadtxt(folder / "merged_Thermal_Coefs.txt").T
    z2 = np.loadtxt(folder / "merged_uq_data.txt").T
    y = np.loadtxt(folder / "merged_F_data.txt").reshape(-1, 1)
    if x.shape[1:] != (20,) or z1.shape[1:] != (3,) or z2.shape[1:] != (2,):
        raise ValueError(
            f"Unexpected feature shapes: {x.shape}, {z1.shape}, {z2.shape}")
    if len({len(x), len(z1), len(z2), len(y)}) != 1:
        raise ValueError(
            "Training inputs and targets have different sample counts")
    if not all(np.isfinite(a).all() for a in (x, z1, z2, y)):
        raise ValueError("Training data contains non-finite values")
    return [torch.tensor(a, dtype=torch.float32) for a in (x, z1, z2, y)]


def _statistics(training_tensors):
    result = {}
    for name, tensor in zip(("x", "z1", "z2", "y"), training_tensors):
        result[name + "_mean"] = tensor.mean(dim=0)
        result[name + "_std"] = tensor.std(dim=0,
                                           unbiased=False).clamp_min(1e-8)
    return result


def _normalize(tensors, stats):
    return [(tensor - stats[name + "_mean"]) / stats[name + "_std"]
            for name, tensor in zip(("x", "z1", "z2", "y"), tensors)]


def _relative_l2(prediction, target):
    return float(torch.linalg.vector_norm(prediction - target) /
                 torch.linalg.vector_norm(target).clamp_min(1e-8))


def train(data_dir: Path, checkpoint_path: Path, metrics_path: Path,
          plot_path: Path, epochs: int = 4500, patience: int = 1500,
          seed: int = 42, device_name: str = "cpu", show_plot: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name)
    arrays = _read_training_data(data_dir)
    if len(arrays[0]) != 4000:
        raise ValueError(
            "Expected 4000 samples; define a new split for a different dataset")

    # Keep the original last 400 samples as the held-out test set.
    # The final 400 of the former training segment form a validation set.
    splits = {"train": slice(0, 3200), "val": slice(3200, 3600),
              "test": slice(3600, 4000)}
    stats = _statistics([array[splits["train"]] for array in arrays])
    normalized = _normalize(arrays, stats)
    sets = {name: [tensor[sl].to(device) for tensor in normalized]
            for name, sl in splits.items()}
    model = BPPointNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    wait = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        x, z1, z2, y = sets["train"]
        loss = criterion(model(x, z1, z2), y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            x, z1, z2, y = sets["val"]
            val_loss = float(criterion(model(x, z1, z2), y))
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 100 == 0:
            print(
                f"Epoch {epoch}: train MSE={float(loss):.6g}, val MSE={val_loss:.6g}")
        if wait >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        x, z1, z2, _ = sets["test"]
        prediction = model(x, z1, z2).cpu() * stats["y_std"] + stats["y_mean"]
    target = arrays[3][splits["test"]]
    metrics = {"seed": seed, "train_count": 3200, "validation_count": 400,
               "test_count": 400, "best_epoch": best_epoch,
               "relative_l2_test": _relative_l2(prediction, target),
               "rmse_test": float(torch.sqrt(torch.mean((prediction - target) ** 2)))}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state,
                "normalization": {k: v.cpu() for k, v in stats.items()},
                "architecture": {"input_dim": 20, "output_dim": 1,
                                 "z1_dim": 3, "z2_dim": 2, "n1": 24, "n2": 6},
                "metrics": metrics}, checkpoint_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Plot the held-out sequence using the original load units.
    import matplotlib.pyplot as plt
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    index = np.arange(len(target))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(index, target[:, 0].numpy(), label="Real", linewidth=1.8)
    ax.plot(index, prediction[:, 0].numpy(), label="Predicted", linewidth=1.5)
    ax.set_xlabel("Held-out sample index")
    ax.set_ylabel("Load (original data units)")
    ax.set_title(
        f"Test load identification | relative L2 error: {metrics['relative_l2_test']:.2%}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    print(f"Saved test comparison plot to {plot_path}")
    if show_plot:
        plt.show()
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parent / "Net_Attention_v3.pth")
    parser.add_argument("--metrics", type=Path,
                        default=Path(__file__).resolve().parent / "Net_Attention_v3_metrics.json")
    parser.add_argument("--plot", type=Path,
                        default=Path(__file__).resolve().parent / "Net_Attention_v3_test_comparison.png")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--epochs", type=int, default=4500)
    parser.add_argument("--patience", type=int, default=4500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args.data_dir, args.checkpoint, args.metrics, args.plot,
          args.epochs, args.patience, args.seed, args.device, args.show_plot)
