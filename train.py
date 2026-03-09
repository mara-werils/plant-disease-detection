"""
PlantGuard AI — MobileNetV2 Fine-Tuning on PlantVillage
Run on RunPod GPU Pod (RTX 3090/4090 recommended)

Usage:
  pip install torch torchvision transformers datasets pillow huggingface_hub
  python train.py
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from huggingface_hub import HfApi, login
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── CONFIG ───────────────────────────────────────────────
DATASET_NAME = "PlantVillage"
NUM_CLASSES = 38
BATCH_SIZE = 64
EPOCHS_PHASE1 = 8      # Frozen backbone
EPOCHS_PHASE2 = 12     # Fine-tuning
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-5
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"🖥️  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─── STEP 1: Download PlantVillage Dataset ───────────────
print("\n📥 Downloading PlantVillage dataset...")

DATASET_DIR = None

# Method: Use opendatasets (Kaggle)
try:
    import opendatasets
except ImportError:
    os.system("pip install opendatasets -q")
    import opendatasets

kaggle_url = "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"

# Check if already downloaded
candidates = [
    "./plantvillage-dataset/color",
    "./plantvillage-dataset",
    "./PlantVillage",
    "./color",
]
for c in candidates:
    if os.path.exists(c):
        subdirs = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))]
        if len(subdirs) >= 30:
            DATASET_DIR = c
            print(f"   Already downloaded: {DATASET_DIR} ({len(subdirs)} classes)")
            break

if DATASET_DIR is None:
    print("   Downloading from Kaggle (~1 GB)...")
    print("   (If prompted for Kaggle credentials, enter your kaggle.com username + API key)")
    print("   (Get API key at: https://www.kaggle.com/settings → Create New Token)")
    opendatasets.download(kaggle_url, data_dir="./")

    # Find the downloaded directory
    for root, dirs, files in os.walk("./plantvillage-dataset"):
        if len(dirs) >= 30:
            DATASET_DIR = root
            break

    if DATASET_DIR is None:
        # Fallback: search everywhere
        for root, dirs, files in os.walk("./"):
            if len(dirs) >= 30 and any("Apple" in d for d in dirs):
                DATASET_DIR = root
                break

    print(f"   Dataset dir: {DATASET_DIR}")

if DATASET_DIR is None:
    print("❌ Could not find dataset! Please download manually.")
    exit(1)

# ─── STEP 2: Data Augmentation Pipeline ──────────────────
print("\n🔄 Setting up Data Augmentation pipeline...")

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("   Augmentations: RandomResizedCrop, HFlip, VFlip, Rotation(±30°),")
print("   ColorJitter, Affine, GaussianBlur, RandomErasing")

# ─── STEP 3: Load Dataset ────────────────────────────────
print(f"\n📂 Loading dataset from: {DATASET_DIR}")

full_dataset = ImageFolder(DATASET_DIR)
class_names = full_dataset.classes
num_classes = len(class_names)
total_images = len(full_dataset)

print(f"   Total images: {total_images}")
print(f"   Classes: {num_classes}")

# Split: 70% train, 15% val, 15% test
train_size = int(0.70 * total_images)
val_size = int(0.15 * total_images)
test_size = total_images - train_size - val_size

generator = torch.Generator().manual_seed(42)
train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)

# Apply transforms
class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

train_data = TransformedSubset(train_set, train_transform)
val_data = TransformedSubset(val_set, val_transform)
test_data = TransformedSubset(test_set, val_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"   Train: {train_size} | Val: {val_size} | Test: {test_size}")

# ─── STEP 4: Build Model ─────────────────────────────────
print("\n🧠 Building MobileNetV2 model...")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Replace classifier head
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.last_channel, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)

model = model.to(DEVICE)
print(f"   Architecture: MobileNetV2 (ImageNet pretrained)")
print(f"   Classifier: 1280 → 256 → {num_classes} (with Dropout 0.4/0.3)")

# ─── STEP 5: Training Loop ───────────────────────────────
criterion = nn.CrossEntropyLoss()
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
        if batch_idx % 50 == 0:
            print(f"      Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}", end="\r")
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total

# ─── PHASE 1: Frozen Backbone ────────────────────────────
print("\n" + "=" * 60)
print("  PHASE 1: Frozen Backbone Training")
print("=" * 60)

# Freeze all backbone layers
for param in model.features.parameters():
    param.requires_grad = False

optimizer = optim.Adam(model.classifier.parameters(), lr=LR_PHASE1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

best_val_acc = 0
for epoch in range(EPOCHS_PHASE1):
    t0 = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)
    elapsed = time.time() - t0

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"  Epoch {epoch+1}/{EPOCHS_PHASE1} ({elapsed:.0f}s) | "
          f"Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}% | "
          f"Loss: {train_loss:.4f}/{val_loss:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_phase1.pth")

print(f"  Best Phase 1 Val Acc: {best_val_acc*100:.1f}%")

# ─── PHASE 2: Fine-Tuning ────────────────────────────────
print("\n" + "=" * 60)
print("  PHASE 2: Fine-Tuning (Last 30 Layers Unfrozen)")
print("=" * 60)

# Unfreeze last 30 layers
for i, param in enumerate(model.features.parameters()):
    total_params = sum(1 for _ in model.features.parameters())
    if i >= total_params - 30:
        param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_PHASE2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE2)

# Load best Phase 1 weights
model.load_state_dict(torch.load(f"{SAVE_DIR}/best_phase1.pth", weights_only=True))

best_val_acc = 0
patience_counter = 0
for epoch in range(EPOCHS_PHASE2):
    t0 = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    scheduler.step()
    elapsed = time.time() - t0

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"  Epoch {epoch+1}/{EPOCHS_PHASE2} ({elapsed:.0f}s) | "
          f"Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}% | "
          f"Loss: {train_loss:.4f}/{val_loss:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_final.pth")
    else:
        patience_counter += 1
        if patience_counter >= 4:
            print("  Early stopping triggered!")
            break

print(f"\n  Best Final Val Acc: {best_val_acc*100:.1f}%")

# ─── STEP 6: Test Evaluation ─────────────────────────────
print("\n" + "=" * 60)
print("  FINAL TEST EVALUATION")
print("=" * 60)

model.load_state_dict(torch.load(f"{SAVE_DIR}/best_final.pth", weights_only=True))
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"  Test Accuracy: {test_acc*100:.2f}%")
print(f"  Test Loss: {test_loss:.4f}")

# ─── STEP 7: Save Training Plots ─────────────────────────
print("\n📊 Saving training plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
ax1.plot(history["val_loss"], label="Val Loss", linewidth=2)
ax1.axvline(x=EPOCHS_PHASE1-0.5, color='gray', linestyle='--', alpha=0.5, label="Phase 1→2")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training & Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot([a*100 for a in history["train_acc"]], label="Train Acc", linewidth=2)
ax2.plot([a*100 for a in history["val_acc"]], label="Val Acc", linewidth=2)
ax2.axvline(x=EPOCHS_PHASE1-0.5, color='gray', linestyle='--', alpha=0.5, label="Phase 1→2")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training & Validation Accuracy")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/training_curves.png", dpi=150)
print(f"   Saved: {SAVE_DIR}/training_curves.png")

# ─── STEP 8: Export for HuggingFace ──────────────────────
print("\n📦 Exporting model for HuggingFace...")

export_dir = f"{SAVE_DIR}/hf_export"
os.makedirs(export_dir, exist_ok=True)

# Save class mapping
id2label = {i: name for i, name in enumerate(class_names)}
label2id = {name: i for i, name in enumerate(class_names)}

config = {
    "architectures": ["MobileNetV2ForImageClassification"],
    "model_type": "mobilenet_v2",
    "num_labels": num_classes,
    "id2label": id2label,
    "label2id": label2id,
    "image_size": IMAGE_SIZE,
    "test_accuracy": round(test_acc * 100, 2),
}

with open(f"{export_dir}/config.json", "w") as f:
    json.dump(config, f, indent=2)

torch.save(model.state_dict(), f"{export_dir}/pytorch_model.bin")

print(f"   Model saved to: {export_dir}/")
print(f"   Config saved with {num_classes} class labels")

# ─── SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TRAINING COMPLETE!")
print("=" * 60)
print(f"""
  Model: MobileNetV2 (ImageNet → PlantVillage)
  Dataset: {total_images} images, {num_classes} classes
  Data Aug: RandomCrop, Flip, Rotation, ColorJitter, Blur, Erasing
  Phase 1: {EPOCHS_PHASE1} epochs (frozen backbone, LR={LR_PHASE1})
  Phase 2: {EPOCHS_PHASE2} epochs (fine-tuning, LR={LR_PHASE2})
  Test Accuracy: {test_acc*100:.2f}%

  Files saved:
    {SAVE_DIR}/best_final.pth          — Best model weights
    {SAVE_DIR}/training_curves.png     — Loss/Accuracy plots
    {SAVE_DIR}/hf_export/              — HuggingFace-ready export
""")
