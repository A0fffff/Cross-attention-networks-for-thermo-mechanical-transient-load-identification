import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks
from torch import nn

from Net_v3_Attention import BPPointNet


FILES = {
    "mechanical_strain": ("merged_strains_.txt", 20),
    "thermal_coefficients": ("merged_Thermal_Coefs.txt", 3),
    "structural_features": ("merged_uq_data.txt", 2),
    "load": ("merged_F_data.txt", 1),
}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_data(folder: Path, case_length: int, aux_prefix_is_aligned: bool):
    if case_length <= 0:
        raise ValueError("case_length must be positive")
    arrays = {}
    for name, (filename, width) in FILES.items():
        path = folder / filename
        values = np.loadtxt(path)
        if width == 1:
            values = values.reshape(-1, 1)
        else:
            if values.ndim != 2 or values.shape[0] != width:
                raise ValueError(
                    f"{path}: expected {width} feature rows, got {values.shape}")
            values = values.T
        arrays[name] = values

    n = len(arrays["load"])
    if aux_prefix_is_aligned:
        for name in ("thermal_coefficients", "structural_features"):
            if len(arrays[name]) >= n:
                arrays[name] = arrays[name][:n]

    problems = []
    for name, values in arrays.items():
        if len(values) != n:
            problems.append(f"{name}: {len(values)} samples, load: {n}")
        bad = np.flatnonzero(~np.isfinite(values).all(axis=1))
        if len(bad):
            counts = [int(np.sum((bad >= j * case_length) &
                                 (bad < (j + 1) * case_length)))
                      for j in range((len(values) + case_length - 1) // case_length)]
            problems.append(f"{name}: {len(bad)} non-finite sample rows; "
                            f"first zero-based indices {bad[:10].tolist()}; "
                            f"bad rows by case {counts}")
    if n < case_length or n % case_length:
        problems.append(f"load: {n} samples cannot form complete "
                        f"histories of {case_length} samples")
    if len(arrays["mechanical_strain"]) == n:
        zero_strain = np.all(arrays["mechanical_strain"] == 0, axis=1)
        nonzero_load = np.abs(arrays["load"][:, 0]) > 1e-6
        # The convolution starts from rest, so the first sample of each
        # independent history can be exactly zero even with a nonzero load.
        first_sample = np.arange(n) % case_length == 0
        suspect = np.flatnonzero(zero_strain & nonzero_load & ~first_sample)
        if len(suspect):
            problems.append(f"mechanical_strain: {len(suspect)} rows have all "
                            f"20 strains exactly zero away from case starts "
                            f"while load is nonzero; "
                            f"first zero-based indices {suspect[:10].tolist()}")
    if len(arrays["structural_features"]) == n:
        zero_structural = np.flatnonzero(
            np.any(arrays["structural_features"] == 0, axis=1))
        if len(zero_structural):
            problems.append(f"structural_features: {len(zero_structural)} rows "
                            f"contain zero; first zero-based indices "
                            f"{zero_structural[:10].tolist()}")
    if problems:
        raise ValueError("Data validation failed:\n  " + "\n  ".join(problems))

    case_count = n // case_length
    tensors = tuple(torch.tensor(arrays[name].reshape(case_count, case_length, -1),
                                 dtype=torch.float32)
                    for name in FILES)
    return tensors, {"case_count": case_count, "case_length": case_length,
                     "aux_prefix_is_aligned": aux_prefix_is_aligned,
                     "input_files": {name: filename for name, (filename, _) in FILES.items()},
                     "input_sha256": {name: sha256(folder / filename)
                                      for name, (filename, _) in FILES.items()}}


class ImpactAttentionNet(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.point_net = BPPointNet(**architecture)
        self.temporal = nn.Sequential(
            nn.Conv1d(26, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=1),
        )
        nn.init.zeros_(self.temporal[-1].weight)
        nn.init.zeros_(self.temporal[-1].bias)

    def forward(self, x, thermal, structural):
        if x.ndim != 3 or thermal.shape[:2] != x.shape[:2] or structural.shape[:2] != x.shape[:2]:
            raise ValueError(
                "Expected aligned (cases, time, features) tensors")
        cases, steps, _ = x.shape
        point = self.point_net(x.reshape(-1, 20), thermal.reshape(-1, 3),
                               structural.reshape(-1, 2)).reshape(cases, steps, 1)
        context = torch.cat((x, thermal, structural, point),
                            dim=-1).transpose(1, 2)
        return point + self.temporal(context).transpose(1, 2)


def peak_locations(y: np.ndarray, prominence_fraction: float = 0.15):
    magnitude = np.abs(y)
    amplitude = float(np.max(magnitude))
    if amplitude <= 0:
        return []
    indices, _ = find_peaks(magnitude, height=prominence_fraction * amplitude,
                            prominence=0.08 * amplitude, distance=5)
    if not len(indices):
        indices = np.array([int(np.argmax(magnitude))])
    return [int(i) for i in indices]


def impact_loss(prediction, target, peaks, peak_weight=8.0,
                event_weight=2.0, quiet_weight=0.1):
    pred = prediction.squeeze(-1)
    real = target.squeeze(-1)
    scale = real.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    residual = (pred - real) / scale
    weights = 1.0 + peak_weight * (real.abs() / scale).square()
    point_loss = (weights * residual.square()).sum() / weights.sum()
    event_errors = [residual[case, index] for case, indices in enumerate(peaks)
                    for index in indices]
    event_loss = (torch.stack(event_errors).square().mean() if event_errors
                  else point_loss.new_zeros(()))
    quiet = real.abs() < 0.08 * scale
    quiet_loss = ((pred[quiet] / scale.expand_as(real)[quiet]).square().mean()
                  if bool(quiet.any()) else point_loss.new_zeros(()))
    return point_loss + event_weight * event_loss + quiet_weight * quiet_loss


def evaluate(prediction, target, case_ids, window=12):
    pred = prediction.detach().cpu().numpy().squeeze(-1)
    real = target.detach().cpu().numpy().squeeze(-1)
    peak_records = []
    for row, case_id in enumerate(case_ids):
        for target_index in peak_locations(real[row]):
            lo = max(0, target_index - window)
            hi = min(len(real[row]), target_index + window + 1)
            sign = 1.0 if real[row, target_index] >= 0 else -1.0
            predicted_index = lo + int(np.argmax(sign * pred[row, lo:hi]))
            true_amplitude = float(real[row, target_index])
            predicted_amplitude = float(pred[row, predicted_index])
            peak_records.append({
                "case": int(case_id), "true_index": target_index,
                "predicted_index": predicted_index,
                "lag_samples": predicted_index - target_index,
                "true_amplitude": true_amplitude,
                "predicted_amplitude": predicted_amplitude,
                "signed_amplitude_error_fraction":
                    (predicted_amplitude - true_amplitude) / abs(true_amplitude),
            })
    quiet_rms = []
    for row in range(len(case_ids)):
        scale = max(float(np.max(np.abs(real[row]))), 1.0)
        quiet = np.abs(real[row]) < 0.08 * scale
        quiet_rms.append(
            float(np.sqrt(np.mean(pred[row, quiet] ** 2)) / scale)
            if np.any(quiet) else None)
    return {
        "relative_l2_raw": float(np.linalg.norm(pred - real) /
                                 max(np.linalg.norm(real), 1e-8)),
        "mean_absolute_peak_amplitude_error_fraction":
            float(np.mean([abs(p["signed_amplitude_error_fraction"])
                           for p in peak_records])) if peak_records else None,
        "mean_absolute_peak_lag_samples":
            float(np.mean([abs(p["lag_samples"]) for p in peak_records]))
            if peak_records else None,
        "mean_quiet_rms_fraction": (float(np.mean([v for v in quiet_rms
                                                  if v is not None]))
                                    if any(v is not None for v in quiet_rms)
                                    else None),
        "peaks": peak_records,
    }


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=script_dir)
    parser.add_argument("--init-checkpoint", type=Path,
                        default=script_dir / "Net_Attention_v3_amplitude_candidate.pth")
    parser.add_argument("--output-dir", type=Path,
                        default=script_dir / "impact_v3")
    parser.add_argument("--case-length", type=int, default=400)
    parser.add_argument("--val-case", type=int, default=9,
                        help="one-based validation case; 0 trains on every case")
    parser.add_argument("--aux-prefix-is-aligned", action="store_true",
                        help="explicitly use the first N auxiliary samples when files are longer")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    arrays, data_info = read_data(args.data_dir, args.case_length,
                                  args.aux_prefix_is_aligned)
    total = data_info["case_count"]
    if not 0 <= args.val_case <= total:
        raise ValueError(
            f"Choose validation case in 0..{total}; 0 disables validation")
    train_cases = [i for i in range(1, total + 1)
                   if i != args.val_case]
    if not train_cases:
        raise ValueError("No training cases remain")
    data_info.update({"training_cases": train_cases,
                      "validation_case": args.val_case or None})
    print(json.dumps(data_info, indent=2), flush=True)
    if args.check_data:
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    source = torch.load(args.init_checkpoint,
                        map_location="cpu", weights_only=True)
    stats = {key: value.to(device)
             for key, value in source["normalization"].items()}
    model = ImpactAttentionNet(source["architecture"]).to(device)
    model.point_net.load_state_dict(source["model_state_dict"])

    x, thermal, structural, y = [array.to(device) for array in arrays]
    x = (x - stats["x_mean"]) / stats["x_std"]
    thermal = (thermal - stats["z1_mean"]) / stats["z1_std"]
    structural = (structural - stats["z2_mean"]) / stats["z2_std"]
    train_idx = torch.tensor([i - 1 for i in train_cases], device=device)
    val_idx = args.val_case - 1 if args.val_case else None
    train_peaks = [peak_locations(y[i - 1, :, 0].cpu().numpy())
                   for i in train_cases]
    val_peaks = ([peak_locations(y[val_idx, :, 0].cpu().numpy())]
                 if val_idx is not None else None)
    if not all(train_peaks) or (val_peaks is not None and not val_peaks[0]):
        raise ValueError(
            "Each training and validation case must contain an identifiable impact peak")

    def physical_prediction(indices):
        return (model(x[indices], thermal[indices], structural[indices]) *
                stats["y_std"] + stats["y_mean"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=1e-5)
    best = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_pred = physical_prediction(train_idx)
        loss = impact_loss(train_pred, y[train_idx], train_peaks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if val_idx is None:
            best = float(loss)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            if epoch == 1 or epoch % 100 == 0:
                print(f"epoch={epoch} train_loss={best:.6g}", flush=True)
        elif epoch == 1 or epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = physical_prediction(slice(val_idx, val_idx + 1))
                score = float(impact_loss(
                    val_pred, y[val_idx:val_idx + 1], val_peaks))
            if score < best - 1e-8:
                best = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
                wait = 0
            else:
                wait += 10
            if epoch == 1 or epoch % 100 == 0:
                print(f"epoch={epoch} train_loss={float(loss):.6g} "
                      f"val_loss={score:.6g}", flush=True)
            if wait >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    if val_idx is not None:
        with torch.no_grad():
            validation_prediction = physical_prediction(
                slice(val_idx, val_idx + 1))
        validation_metrics = evaluate(validation_prediction,
                                      y[val_idx:val_idx + 1], [args.val_case])
    else:
        validation_prediction = None
        validation_metrics = None
    result = {
        "seed": args.seed, "best_epoch": best_epoch,
        "selection_loss": best, "data": data_info,
        "initial_checkpoint": str(args.init_checkpoint),
        "initial_checkpoint_sha256": sha256(args.init_checkpoint),
        "architecture": "v3 attention with centered 9-sample temporal residual",
        "loss": {"peak_weight": 8.0, "event_weight": 2.0,
                 "quiet_weight": 0.1},
        "validation": validation_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state,
                "normalization": source["normalization"],
                "base_architecture": source["architecture"],
                "metrics": result}, args.output_dir / "Impact_Attention_v3.pth")
    (args.output_dir / "Impact_Attention_v3_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    if val_idx is not None:
        with (args.output_dir / "Impact_Attention_v3_validation_predictions.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ("case", "sample_index", "real_load", "predicted_load_raw"))
            for i, (real, predicted) in enumerate(zip(
                    y[val_idx, :, 0].cpu().numpy(),
                    validation_prediction[0, :, 0].cpu().numpy())):
                writer.writerow(
                    (args.val_case, i, float(real), float(predicted)))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y[val_idx, :, 0].cpu().numpy(), label="Real", linewidth=1.8)
        ax.plot(validation_prediction[0, :, 0].cpu().numpy(),
                label="Impact v3", linewidth=1.5)
        ax.set(xlabel="Sample index", ylabel="Load (source units)",
               title=f"Validation impact case {args.val_case}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir /
                    "Impact_Attention_v3_validation_comparison.png", dpi=200)
        plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
