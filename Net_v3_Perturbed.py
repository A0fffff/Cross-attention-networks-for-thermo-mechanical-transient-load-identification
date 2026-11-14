import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from Net_v3_Attention import _read_training_data
from Net_v3_Impact import ImpactAttentionNet


SCALES = (0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)


def wave_denoise_matlab(signal, target_window):
    """Match wave_denoise.m: cubic SG weights and extended endpoint values."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError("Filtering requires a finite 1-D signal of at least 4 points")
    if target_window < 3:
        raise ValueError("SG target window must be at least 3")
    max_frame = len(x) - 1
    if max_frame % 2 == 0:
        max_frame = max(3, max_frame - 1)
    applied = min(max_frame, target_window)
    if applied % 2 == 0:
        applied += 1
    m = applied // 2
    j = np.arange(-m, m + 1, dtype=np.float64)
    denominator = (2*m - 1) * (2*m + 1) * (2*m + 3)
    weights = 3 * (3*m*m + 3*m - 1 - 5*j*j) / denominator
    padded = np.pad(x, (m, m), mode="edge")
    return np.convolve(padded, weights, mode="valid"), applied


def read_case(path, name, width):
    a = np.loadtxt(path / name, ndmin=2)
    if a.shape[1] != width:
        if a.shape[0] != width:
            raise ValueError(
                f"{name}: expected {width} features, got {a.shape}")
        a = a.T
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values")
    return torch.tensor(a, dtype=torch.float32)


def physical(model, x, thermal, structural, stats):
    prediction = model((x - stats["x_mean"]) / stats["x_std"],
                       (thermal - stats["z1_mean"]) / stats["z1_std"],
                       (structural - stats["z2_mean"]) / stats["z2_std"])
    return prediction * stats["y_std"] + stats["y_mean"]


def curve_loss(prediction, target):
    peak = target.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    residual = (prediction - target) / peak
    point = residual.square().mean()
    smooth = F.avg_pool1d(residual.transpose(1, 2), 21, stride=1,
                          padding=10, count_include_pad=False).square().mean()
    target_delta = target[:, 1:] - target[:, :-1]
    pred_delta = prediction[:, 1:] - prediction[:, :-1]
    delta_scale = target_delta.square().mean(dim=1, keepdim=True).sqrt()
    delta_scale = torch.maximum(delta_scale, 0.01 * peak)
    slope = ((pred_delta - target_delta) / delta_scale).square().mean()
    return point + 0.5 * smooth + 0.1 * slope


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    source = torch.load(args.init_checkpoint,
                        map_location="cpu", weights_only=True)
    stats = {k: v.to(device) for k, v in source["normalization"].items()}
    arrays = _read_training_data(args.data_dir)
    if len(arrays[0]) != 4000:
        raise ValueError("Expected ten aligned histories of 400 samples")
    x, thermal, structural, target = [a.reshape(10, 400, -1).to(device)
                                      for a in arrays]
    if not 1 <= args.val_case <= 10:
        raise ValueError("--val-case must be in 1..10")
    train_ids = [j for j in range(10) if j != args.val_case - 1]
    val_id = args.val_case - 1
    model = ImpactAttentionNet(source["architecture"]).to(device)
    model.point_net.load_state_dict(source["model_state_dict"])
    optimizer = torch.optim.AdamW([
        {"params": model.point_net.parameters(), "lr": args.point_lr},
        {"params": model.temporal.parameters(), "lr": args.temporal_lr}],
        weight_decay=1e-5)
    scales = torch.tensor(SCALES, device=device)
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        chosen = scales[torch.randint(len(scales), (len(train_ids), 1, 1),
                                      device=device)]
        full = physical(model, x[train_ids], thermal[train_ids],
                        structural[train_ids], stats)
        small = physical(model, x[train_ids] * chosen, thermal[train_ids],
                         structural[train_ids], stats)
        loss = (curve_loss(full, target[train_ids]) +
                curve_loss(small, target[train_ids] * chosen)) / 2
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                terms = []
                for scale in (1.0, 0.005, 0.01, 0.025):
                    value = physical(model, x[val_id:val_id+1] * scale,
                                     thermal[val_id:val_id+1],
                                     structural[val_id:val_id+1], stats)
                    terms.append(float(curve_loss(
                        value, target[val_id:val_id+1] * scale)))
                score = float(np.mean(terms))
            if score < best - 1e-8:
                best, best_epoch, stale = score, epoch, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                stale += 10
            if epoch == 1 or epoch % 100 == 0:
                print(f"epoch={epoch} train={float(loss):.6g} val={score:.6g}",
                      flush=True)
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        v = physical(model, x[val_id:val_id+1],
                     thermal[val_id:val_id+1], structural[val_id:val_id+1],
                     stats)
    relative_l2 = float(torch.linalg.vector_norm(v - target[val_id:val_id+1]) /
                        torch.linalg.vector_norm(target[val_id:val_id+1]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "Perturbed_Attention_v3.pth"
    torch.save({"model_state_dict": best_state,
                "base_architecture": source["architecture"],
                "normalization": source["normalization"],
                "training": {"seed": args.seed, "train_cases": [j+1 for j in train_ids],
                             "validation_case": args.val_case,
                             "best_epoch": best_epoch, "validation_loss": best,
                             "validation_relative_l2": relative_l2,
                             "scales": SCALES,
                             "input_checkpoint": str(args.init_checkpoint),
                             "input_checkpoint_sha256": hashlib.sha256(
                                 args.init_checkpoint.read_bytes()).hexdigest()}}, checkpoint)
    (args.output_dir / "Perturbed_Attention_v3_metrics.json").write_text(
        json.dumps(torch.load(checkpoint, map_location="cpu", weights_only=True)[
            "training"], indent=2), encoding="utf-8")
    print(f"Saved {checkpoint}; validation relative L2={relative_l2:.3%}")


def predict(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ImpactAttentionNet(saved["base_architecture"])
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    stats = saved["normalization"]
    case = args.data_dir / "testdata" / str(args.case)
    x = read_case(case, "strains.txt", 20)
    thermal = read_case(case, "Thermal_Coefs.txt", 3)
    structural = read_case(case, "uq_data.txt", 2)
    if len({len(x), len(thermal), len(structural)}) != 1:
        raise ValueError("Test inputs have different sample counts")
    with torch.no_grad():
        prediction = physical(model, x[None], thermal[None],
                              structural[None], stats)[0, :, 0].numpy()
    filtered, applied_window = wave_denoise_matlab(prediction, args.sg_window)
    result = args.output_dir / f"Perturbed_Attention_v3_case{args.case}.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real_path = case / "F_data.txt"
    real = read_case(case, "F_data.txt", 1)[
        :, 0].numpy() if real_path.exists() else None
    if real is not None and len(real) != len(prediction):
        raise ValueError("Reference load length does not match inputs")
    with result.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "predicted_load_raw",
                         "predicted_load_filtered"] +
                        (["real_load"] if real is not None else []))
        for i, value in enumerate(prediction):
            writer.writerow([i, float(value), float(filtered[i])] +
                            ([float(real[i])] if real is not None else []))
    if real is not None:
        denominator = np.linalg.norm(real)
        print(f"Case {args.case} relative L2: raw "
              f"{np.linalg.norm(prediction-real)/denominator:.3%}; "
              f"filtered {np.linalg.norm(filtered-real)/denominator:.3%}")
    fig, ax = plt.subplots(figsize=(10, 4))
    if real is not None:
        ax.plot(real, label="Real", linewidth=1.5)
    ax.plot(prediction, label="Identification (raw)", linewidth=1.0,
            alpha=0.6)
    ax.plot(filtered, label=f"Identification (SG {applied_window})",
            linewidth=1.5)
    ax.set(xlabel="Sample index", ylabel="Load", title=f"Case {args.case}")
    ax.legend()
    fig.tight_layout()
    plot_path = result.with_suffix(".png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved {result}")
    print(f"Saved {plot_path}")
    print(f"Applied MATLAB-equivalent SG window: {applied_window}")


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("train", "predict"), default="train")
    parser.add_argument("--data-dir", type=Path, default=project)
    parser.add_argument("--init-checkpoint", type=Path,
                        default=project / "Net_Attention_v3_amplitude_candidate.pth")
    parser.add_argument("--checkpoint", type=Path,
                        default=project / "perturbed_v3" / "Perturbed_Attention_v3.pth")
    parser.add_argument("--output-dir", type=Path,
                        default=project / "perturbed_v3")
    parser.add_argument("--case", type=int, default=3)
    parser.add_argument("--sg-window", type=int, default=41)
    parser.add_argument("--val-case", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--point-lr", type=float, default=1e-5)
    parser.add_argument("--temporal-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
