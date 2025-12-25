import os
import shutil
import random
import xml.etree.ElementTree as ET

# =============================
# PATH CONFIGURATION
# =============================
IMAGE_DIR = "../images"
ANNOTATION_DIR = "../annotations"
OUTPUT_DIR = "../dataset"

TRAIN_RATIO = 0.8
CLASSES = {
    "with_mask": "with_mask",
    "without_mask": "without_mask",
    "mask_weared_incorrect": "incorrect_mask"
}

# =============================
# CREATE FOLDER STRUCTURE
# =============================
for split in ["train", "val"]:
    for cls in CLASSES.values():
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# =============================
# READ ANNOTATIONS
# =============================
data = []

for xml_file in os.listdir(ANNOTATION_DIR):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOTATION_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    image_path = os.path.join(IMAGE_DIR, filename)

    if not os.path.exists(image_path):
        continue

    # Ambil label pertama (cukup untuk klasifikasi)
    obj = root.find("object")
    if obj is None:
        continue

    label = obj.find("name").text
    if label not in CLASSES:
        continue

    data.append((image_path, CLASSES[label]))

# =============================
# SHUFFLE & SPLIT DATA
# =============================
random.shuffle(data)
split_index = int(len(data) * TRAIN_RATIO)

train_data = data[:split_index]
val_data = data[split_index:]

# =============================
# COPY FILES
# =============================
def copy_files(data_list, split_name):
    for img_path, label in data_list:
        dst = os.path.join(OUTPUT_DIR, split_name, label)
        shutil.copy(img_path, dst)

copy_files(train_data, "train")
copy_files(val_data, "val")

print("✅ Dataset preparation completed!")
print(f"Total images: {len(data)}")
print(f"Train: {len(train_data)}")
print(f"Validation: {len(val_data)}")
