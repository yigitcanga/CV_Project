import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, jaccard_score, precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision.models import resnet18
import torch.nn.functional as F

# -----------------------
# Config
# -----------------------
class Config:
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 50
    LEARNING_RATE = 5e-4
    CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model_resnet_unet.pth")

# -----------------------
# Utils
# -----------------------
def load_mask(path):
    mask = Image.open(path).convert("L")
    mask = mask.resize(Config.IMAGE_SIZE)
    mask = np.array(mask) > 0
    return mask.astype(np.uint8)

def generate_change_mask(mask_before, mask_after):
    return (mask_before != mask_after).astype(np.uint8)

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize(Config.IMAGE_SIZE)
    image = transforms.ToTensor()(image)
    return image

# -----------------------
# Dataset
# -----------------------
class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.before_images = sorted(os.listdir(os.path.join(root_dir, "images_before")))
        self.after_images = sorted(os.listdir(os.path.join(root_dir, "images_after")))
        self.mask_before = sorted(os.listdir(os.path.join(root_dir, "masks_before")))
        self.mask_after = sorted(os.listdir(os.path.join(root_dir, "masks_after")))

    def __len__(self):
        return len(self.before_images)

    def __getitem__(self, idx):
        path_before_img = os.path.join(self.root_dir, "images_before", self.before_images[idx])
        path_after_img = os.path.join(self.root_dir, "images_after", self.after_images[idx])
        path_before_mask = os.path.join(self.root_dir, "masks_before", self.mask_before[idx])
        path_after_mask = os.path.join(self.root_dir, "masks_after", self.mask_after[idx])

        img_before = load_image(path_before_img)
        img_after = load_image(path_after_img)
        mask_before = load_mask(path_before_mask)
        mask_after = load_mask(path_after_mask)

        input_tensor = torch.cat([img_before, img_after], dim=0)  # [6, H, W]
        change_mask = generate_change_mask(mask_before, mask_after)
        change_mask = torch.tensor(change_mask, dtype=torch.float32).unsqueeze(0)

        return input_tensor, change_mask


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # Sigmoid uygula çünkü BCEWithLogitsLoss yerine geçiyor
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1 - dice

def combo_loss(pred, target, bce_weight=0.5):
    bce = nn.BCEWithLogitsLoss()(pred, target)
    d_loss = dice_loss(pred, target)
    return bce_weight * bce + (1 - bce_weight) * d_loss


# -----------------------
# Model
# -----------------------
class ResNetChangeDetection(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = resnet18(pretrained=pretrained)

        # Giriş kanalını 6 yapmak için ilk katmanı değiştir
        self.encoder_conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.encoder_conv1.weight.data[:, :3] = resnet.conv1.weight.data
        self.encoder_conv1.weight.data[:, 3:] = resnet.conv1.weight.data

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512

        # Decoder (U-Net tarzı skip connection ile)
        self.up1 = self.upsample(512, 256)
        self.up2 = self.upsample(512, 128)  # 256 + 256
        self.up3 = self.upsample(256, 64)   # 128 + 128
        self.up4 = self.upsample(128, 32)   # 64 + 64

        self.final_conv = nn.Conv2d(96, 1, kernel_size=1)

    def upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x1 = self.relu(self.bn1(self.encoder_conv1(x)))  # [B, 64, 224, 224]
        x2 = self.encoder1(self.maxpool(x1))             # [B, 64, 112, 112]
        x3 = self.encoder2(x2)                           # [B, 128, 56, 56]
        x4 = self.encoder3(x3)                           # [B, 256, 28, 28]
        x5 = self.encoder4(x4)                           # [B, 512, 14, 14]

        # Decoder with skip connections
        d1 = self.up1(x5)                                # [B, 256, 28, 28]
        d1 = torch.cat([d1, x4], dim=1)                  # [B, 512, 28, 28]

        d2 = self.up2(d1)                                # [B, 128, 56, 56]
        d2 = torch.cat([d2, x3], dim=1)                  # [B, 256, 56, 56]

        d3 = self.up3(d2)                                # [B, 64, 112, 112]
        d3 = torch.cat([d3, x2], dim=1)                  # [B, 128, 112, 112]

        d4 = self.up4(d3)                                # [B, 32, 224, 224]
        d4 = torch.cat([d4, x1], dim=1)                  # [B, 64, 224, 224]

        out = self.final_conv(d4)                        # [B, 1, 224, 224]

        return out


# -----------------------
# Training Function
# -----------------------
def train():
    full_dataset = ChangeDetectionDataset("dataset/train")
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    model = ResNetChangeDetection().to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_loss = float("inf")
    patience = 4
    patience_counter = 0
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            inputs = inputs.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = combo_loss(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"\nEpoch {epoch} Train Loss: {avg_train_loss:.4f}")

        # --------- Validation ---------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                inputs = inputs.to(Config.DEVICE)
                targets = targets.to(Config.DEVICE)

                outputs = model(inputs)
                loss = combo_loss(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch {epoch} Validation Loss: {avg_val_loss:.4f}")

        # --------- Early Stopping ---------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print("Best model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


# -----------------------
# Evaluation Function
# -----------------------
def evaluate(model, dataloader):

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(Config.DEVICE)
            y = y.to(Config.DEVICE)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(y.cpu().numpy().flatten())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    print("Precision:", precision_score(targets, preds))
    print("F1 Score:", f1_score(targets, preds))
    print("IoU Score:", jaccard_score(targets, preds))

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    train()

    test_dataset = ChangeDetectionDataset("dataset/test")
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    model = ResNetChangeDetection().to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE))

    evaluate(model, test_loader)
