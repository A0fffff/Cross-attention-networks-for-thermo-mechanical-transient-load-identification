import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ================= Configuration =================
MODEL_PATH = 'poly_model_scaled.pth'
n = 6
TESTSET_DIR = 'D:/Curriculum/y2/paper5/FEM2/pics/testset' + str(n)
OUTPUT_TXT_PATH = './testset' + str(n) + '_predictions.txt'

# Path to the ground truth labels for comparison
GROUND_TRUTH_PATH = os.path.join(
    TESTSET_DIR, 'detailed_data', 'Thermal_Coefs.txt')

# NORM_MEAN = np.array([199.99992, 200.00008, 199.99979], dtype=np.float32)
# NORM_STD  = np.array([70.71067,  70.71066,  73.607025], dtype=np.float32)
NORM_MEAN = np.array([63.67700608, 73.16238986, 72.11799827], dtype=np.float32)
NORM_STD = np.array([50.3549114,  51.6393158,  46.61936971], dtype=np.float32)

IMG_SIZE = 128
# ==================================================

# --- 1. Model Definition ---


class PolyRegresser(nn.Module):
    def __init__(self):
        super(PolyRegresser, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3,
                      padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3,
                      padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3,
                      padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3,
                      padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

# --- 2. Main Prediction Logic ---


def predict_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PolyRegresser().to(device)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"[Fatal Error] Could not find model: {MODEL_PATH}")
        return

    config_path = os.path.join(
        TESTSET_DIR, 'detailed_data', 'Contour_Config.txt')
    try:
        with open(config_path, 'r') as f:
            lines = f.readlines()
            min_val, max_val = map(float, lines[1].strip().split())
        print(f"Loaded config: Min={min_val}, Max={max_val}")
    except Exception as e:
        print(f"[Fatal Error] Failed to read config: {e}")
        return

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    all_predictions = []

    print("Starting batch prediction...")
    with torch.no_grad():
        for i in range(1, 401):
            img_name = f"{i:04d}.png"
            img_path = os.path.join(TESTSET_DIR, img_name)

            try:
                image = Image.open(img_path).convert('RGB')
            except FileNotFoundError:
                print(f"[Warning] Image missing: {img_path}")
                all_predictions.append([0.0, 0.0, 0.0])
                continue

            input_tensor = transform(image)
            input_tensor = input_tensor * (max_val - min_val) + min_val
            input_tensor = input_tensor.unsqueeze(0).to(device)

            output = model(input_tensor)
            pred_norm = output.cpu().numpy()[0]
            pred_real = pred_norm * NORM_STD + NORM_MEAN
            all_predictions.append(pred_real)

            if i % 100 == 0:
                print(f"Processed {i}/400 images...")

    predictions_array = np.array(all_predictions)

    try:
        with open(OUTPUT_TXT_PATH, 'w') as f:
            for row in predictions_array:
                formatted_row = []
                for val in row:
                    if val == 0.0:
                        formatted_row.append("  0.00000000E+00")
                    else:
                        # Standard Fortran E16.8 format calculation: 0.XXXXXXXXE+YY
                        exponent = int(np.floor(np.log10(abs(val)))) + 1
                        mantissa = val / (10**exponent)
                        # >12.8f ensures 8 decimal places and pads with spaces to length 12
                        # +03d ensures exponent has sign and is padded to length 3 (e.g., +02)
                        formatted_str = f"{mantissa:>12.8f}E{exponent:+03d}"
                        formatted_row.append(formatted_str)
                # Join the 3 strings together without extra spaces,
                # as each is already exactly 16 characters wide
                f.write("".join(formatted_row) + "\n")
        print(f"Prediction complete! Results saved to: {OUTPUT_TXT_PATH}")
    except Exception as e:
        print(f"[Error] Failed to save predictions to txt: {e}")

    # --- 3. Plot Comparison ---
    print("Loading ground truth for visualization...")
    try:
        ground_truth = np.loadtxt(GROUND_TRUTH_PATH, dtype=np.float32)

        if len(ground_truth) != len(predictions_array):
            print("[Warning] Shape mismatch between predictions and ground truth.")

        # Create a figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        coef_names = ['Coefficient 1', 'Coefficient 2', 'Coefficient 3']

        for idx in range(3):
            # Plot Ground Truth
            axes[idx].plot(ground_truth[:, idx],
                           label='Ground Truth', color='blue', alpha=0.7)
            # Plot Prediction
            axes[idx].plot(predictions_array[:, idx], label='Prediction',
                           color='red', linestyle='--', alpha=0.8)

            axes[idx].set_title(f'Comparison of Thermal {coef_names[idx]}')
            axes[idx].set_xlabel('Sample Index')
            axes[idx].set_ylabel('Value')
            axes[idx].legend()
            axes[idx].grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plot_save_path = 'comparison_plot.png'
        plt.savefig(plot_save_path, dpi=300)
        print(f"Comparison plot successfully saved to: {plot_save_path}")

    except Exception as e:
        print(f"[Error] Failed to generate comparison plot: {e}")


if __name__ == '__main__':
    predict_and_evaluate()
