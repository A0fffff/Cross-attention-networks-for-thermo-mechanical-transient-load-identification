import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from Net_v3_Attention import BPPointNet as MHCAPointNet

# ---------------------------------------------------------
# 1. Model Definitions
# ---------------------------------------------------------


class CrossAttention(nn.Module):
    def __init__(self, x_dim=20, z_dim=3, hidden_dim=32, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(x_dim, hidden_dim)
        self.k_proj = nn.Linear(z_dim, hidden_dim)
        self.v_proj = nn.Linear(z_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, x_dim)

    def forward(self, x, z):
        Q = self.q_proj(x)
        K = self.k_proj(z)
        V = self.v_proj(z)

        Q = Q.unsqueeze(1)
        K = K.unsqueeze(2)

        scores = torch.bmm(Q, K) / (self.hidden_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        V = V.unsqueeze(1)
        H = attn_weights * V
        H = H.squeeze(1)

        out = self.out_proj(H)
        return out


class BPPointNet(nn.Module):
    def __init__(self, input_dim=20, output_dim=1, z1_dim=3, z2_dim=2, n1=24, n2=6):
        super(BPPointNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, n1)
        self.fc2 = nn.Linear(n1, n2)
        self.fc3 = nn.Linear(n2, output_dim)
        self.relu = nn.ReLU()

        self.cross_attn1_z1 = CrossAttention(
            x_dim=n1, z_dim=z1_dim, hidden_dim=32)
        self.cross_attn2_z1 = CrossAttention(
            x_dim=n2, z_dim=z1_dim, hidden_dim=32)

        self.cross_attn1_z2 = CrossAttention(
            x_dim=n1, z_dim=z2_dim, hidden_dim=32)
        self.cross_attn2_z2 = CrossAttention(
            x_dim=n2, z_dim=z2_dim, hidden_dim=32)

    def forward(self, x, z1, z2):
        h1 = self.relu(self.fc1(x))
        h1 = h1 + self.cross_attn1_z1(h1, z1) + self.cross_attn1_z2(h1, z2)

        h2 = self.relu(self.fc2(h1))
        h2 = h2 + self.cross_attn2_z1(h2, z1) + self.cross_attn2_z2(h2, z2)

        out = self.fc3(h2)
        return out

# ---------------------------------------------------------
# 2. Optimization Search Function
# ---------------------------------------------------------


def optimize_bounds_for_uq(model_path, strains_file, thermal_file, uq_baseline_file):
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return None, None, None

    print("Loading static datasets for inference...")
    strains_np = np.loadtxt(strains_file).T
    thermal_coefs_np = np.loadtxt(thermal_file)

    x_test = torch.tensor(strains_np, dtype=torch.float32)
    z1_test = torch.tensor(thermal_coefs_np, dtype=torch.float32)

    if x_test.ndim == 1:
        x_test = x_test.view(-1, 1)
    if z1_test.ndim == 1:
        z1_test = z1_test.view(-1, 1)

    x_test1 = (x_test - x_test.mean()) / (x_test.std() + 1e-8)
    z1_test1 = (z1_test - z1_test.mean()) / (z1_test.std() + 1e-8)

    num_steps = x_test1.shape[0]

    print("Loading baseline UQ data to compute correct fixed normalization constants...")
    try:
        uq_baseline_np = np.loadtxt(uq_baseline_file).T
        uq_baseline_tensor = torch.tensor(uq_baseline_np, dtype=torch.float32)
        z2_fixed_mean = uq_baseline_tensor.mean()
        z2_fixed_std = uq_baseline_tensor.std()
        print(
            f"Fixed Z2 Mean: {z2_fixed_mean:.4f}, Fixed Z2 Std: {z2_fixed_std:.4f}")
    except Exception as e:
        print(f"Error loading UQ baseline file: {e}")
        return None, None, None

    print("Initializing model and loading weights...")
    model = BPPointNet(input_dim=20, output_dim=1, z1_dim=3, z2_dim=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    checkpoint_path = os.path.join(os.path.dirname(model_path),
                                   "Net_Attention_v3_amplitude_candidate.pth")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    nominal_model = MHCAPointNet(**saved["architecture"])
    nominal_model.load_state_dict(saved["model_state_dict"])
    nominal_model.eval()
    stats = saved["normalization"]
    with torch.no_grad():
        nominal_prediction = nominal_model(
            (x_test - stats["x_mean"]) / stats["x_std"],
            (z1_test - stats["z1_mean"]) / stats["z1_std"],
            (uq_baseline_tensor - stats["z2_mean"]) / stats["z2_std"])
        nominal_prediction = (nominal_prediction * stats["y_std"]
                              + stats["y_mean"])
    y_test_mean = nominal_prediction.mean()
    y_test_std = nominal_prediction.std()
    has_target = True

    # Identify the case with its time-varying structural parameters using
    # the same v2 model and scaling as the interval objective.
    nominal_z2 = (uq_baseline_tensor - z2_fixed_mean) / (z2_fixed_std + 1e-8)
    with torch.no_grad():
        nominal_values = (model(x_test1, z1_test1, nominal_z2)
                          * (y_test_std + 1e-8) + y_test_mean)
    nominal_values = nominal_values[:, 0].numpy()
    first_point_offset = float(nominal_values[0])

    bounds = [(0.95, 1.05), (0.95, 1.05)]

    max_values = np.zeros(num_steps)
    min_values = np.zeros(num_steps)

    def objective(k_vars, target_index, is_maximize):
        k1, k2 = k_vars

        col1 = k1 * np.ones(num_steps)
        col2 = k2 * np.ones(num_steps)
        uq_data_np = np.column_stack((col1, col2))

        z2_test = torch.tensor(uq_data_np, dtype=torch.float32)
        z2_test1 = (z2_test - z2_fixed_mean) / (z2_fixed_std + 1e-8)

        with torch.no_grad():
            pred = model(x_test1, z1_test1, z2_test1)

        if has_target:
            pred = pred * (y_test_std + 1e-8) + y_test_mean

        data = pred.numpy()
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)

        target_value = data[target_index, 0]

        return -target_value if is_maximize else target_value

    print(
        f"Starting Differential Evolution global optimization for {num_steps} time steps...")
    print("This might take a few minutes, please be patient...")

    for i in range(num_steps):
        if i % 20 == 0:
            print(f"Optimizing time step {i}/{num_steps}...")

        # Using differential_evolution for global minimum search
        # popsize=10 is enough for a 2D problem, keeps it relatively fast
        res_min = differential_evolution(
            objective,
            bounds=bounds,
            args=(i, False),
            popsize=10,
            tol=1e-3,
            seed=42,
            polish=True
        )
        min_values[i] = res_min.fun

        # Using differential_evolution for global maximum search
        res_max = differential_evolution(
            objective,
            bounds=bounds,
            args=(i, True),
            popsize=10,
            tol=1e-3,
            seed=42,
            polish=True
        )
        max_values[i] = -res_max.fun

    # Apply one common offset after optimization; this preserves interval width.
    min_values -= first_point_offset
    max_values -= first_point_offset
    nominal_values -= first_point_offset
    print(
        f"Global optimization completed. First-point offset: {first_point_offset:.6f}")
    return min_values, max_values, nominal_values


# ---------------------------------------------------------
# 3. Execution and Saving to Excel
# ---------------------------------------------------------
if __name__ == "__main__":
    current_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(current_path)

    MODEL_WEIGHTS = os.path.join(dir_path, "Net_Model.pth")

    casenum = 2
    folder = dir_path + "/testdata/" + str(casenum) + "/"
    # NEW_STRAINS_FILE = folder + "Thermal_Bias.txt"
    NEW_STRAINS_FILE = folder + "strains.txt"
    NEW_THERMAL_FILE = folder + "Thermal_Coefs.txt"
    NEW_UQ_BASELINE_FILE = folder + "uq_data.txt"
    NEW_TARGET_FILE = folder + "F_data.txt"

    min_vals, max_vals, nominal_vals = optimize_bounds_for_uq(
        model_path=MODEL_WEIGHTS,
        strains_file=NEW_STRAINS_FILE,
        thermal_file=NEW_THERMAL_FILE,
        uq_baseline_file=NEW_UQ_BASELINE_FILE
    )

    if min_vals is not None and max_vals is not None:
        output_excel_path = os.path.join(
            dir_path, "Optimized_UQ_Bounds_firstpoint_aligned.xlsx")
        output_csv_path = os.path.join(
            dir_path, "Optimized_UQ_Bounds_firstpoint_aligned.csv")
        print("Saving optimized bounds to Excel file...")

        df = pd.DataFrame({
            "Max_Value": max_vals,
            "Min_Value": min_vals,
            "Nominal_Prediction": nominal_vals
        })
        if os.path.exists(NEW_TARGET_FILE):
            real_values = np.loadtxt(NEW_TARGET_FILE).reshape(-1)
            if len(real_values) != len(min_vals):
                raise ValueError("Reference load length does not match bounds")
            df["real value"] = real_values

        try:
            df.to_excel(output_excel_path, index=False)
            df.to_csv(output_csv_path, index=False)
            print(
                f"Saved first-point-aligned bounds to {output_excel_path} and {output_csv_path}")
        except Exception as e:
            print(f"Failed to save Excel file: {e}")
