import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =============================
# CONFIG
# =============================
DATA_DIR = "../dataset"
BATCH_SIZE = 16
IMG_SIZE = 224

# =============================
# TRANSFORMS
# =============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# =============================
# DATASETS
# =============================
train_dataset = datasets.ImageFolder(
    root=DATA_DIR + "/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root=DATA_DIR + "/val",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
print("Classes:", class_names)

# =============================
# SHOW SAMPLE IMAGES
# =============================
images, labels = next(iter(train_loader))

plt.figure(figsize=(8, 8))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    img = images[i].permute(1, 2, 0)
    plt.imshow(img)
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()
