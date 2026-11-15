import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# ================= Configuration =================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 1e-4

DATASET_ROOT = 'D:/Curriculum/y2/paper5/FEM2/pics'

# NORM_MEAN = np.array([199.99992, 200.00008, 199.99979], dtype=np.float32)
# NORM_STD  = np.array([70.71067,  70.71066,  73.607025], dtype=np.float32)
NORM_MEAN = np.array([63.67700608, 73.16238986, 72.11799827], dtype=np.float32)
NORM_STD = np.array([50.3549114,  51.6393158,  46.61936971], dtype=np.float32)

MODEL_SAVE_PATH = 'poly_model_scaled.pth'
# ==================================================

# --- 1. Dataset Definition ---


class ScaledThermalDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # 1. Load Image
        try:
            image = Image.open(item['img_path']).convert('RGB')
        except FileNotFoundError:
            print(f"[Error] Image not found: {item['img_path']}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        # Transform returns tensor in range [0.0, 1.0]
        if self.transform:
            image = self.transform(image)

        # 2. Scale image directly using Min and Max
        min_val = item['min_max'][0]
        max_val = item['min_max'][1]

        # Linear transformation: scale to [min_val, max_val]
        image = image * (max_val - min_val) + min_val

        # 3. Load and Normalize Label
        raw_label = np.array(item['label'], dtype=np.float32)
        norm_label = (raw_label - NORM_MEAN) / NORM_STD
        label_tensor = torch.tensor(norm_label, dtype=torch.float32)

        return image, label_tensor


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

# --- 3. Data Loading Logic ---


def prepare_dataset_list(root_dir):
    data_list = []

    trainset_folders = [f for f in os.listdir(root_dir) if f.startswith(
        'trainset') and os.path.isdir(os.path.join(root_dir, f))]
    print(f"Found {len(trainset_folders)} training sets.")

    for folder_name in trainset_folders:
        folder_path = os.path.join(root_dir, folder_name)
        detailed_dir = os.path.join(folder_path, 'detailed_data')

        config_path = os.path.join(detailed_dir, 'Contour_Config.txt')
        label_path = os.path.join(detailed_dir, 'Thermal_Coefs.txt')

        if not os.path.exists(config_path) or not os.path.exists(label_path):
            print(
                f"[Warning] Missing config or label files in {folder_name}. Skipping...")
            continue

        try:
            with open(config_path, 'r') as f:
                lines = f.readlines()
                min_val, max_val = map(float, lines[1].strip().split())
        except Exception as e:
            print(f"[Error] Failed to read {config_path}: {e}")
            continue

        try:
            labels = np.loadtxt(label_path, dtype=np.float32)
        except Exception as e:
            print(f"[Error] Failed to read {label_path}: {e}")
            continue

        for i in range(1, 401):
            img_filename = f"{i:04d}.png"
            img_path = os.path.join(folder_path, img_filename)

            data_list.append({
                'img_path': img_path,
                'min_max': [min_val, max_val],
                'label': labels[i - 1]
            })

    return data_list

# --- 4. Training Loop ---


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Training on device: {device}")

    print("Scanning dataset directories...")
    data_list = prepare_dataset_list(DATASET_ROOT)
    print(f"Successfully prepared {len(data_list)} data samples.")

    if len(data_list) == 0:
        print("[Fatal Error] No valid data found.")
        return

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = ScaledThermalDataset(data_list, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)

    model = PolyRegresser().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training loop...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # Notice: Now we only receive images and targets, min/max is already baked in!
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Average Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training completed. Model saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    train_model()
