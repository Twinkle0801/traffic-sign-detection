import os
from PIL import Image

base = "data/traffic_sign_dataset"
splits = ["train", "valid", "test"]
corrupt = []

for split in splits:
    img_dir = os.path.join(base, split, "images")
    for fname in os.listdir(img_dir):
        path = os.path.join(img_dir, fname)
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            corrupt.append((path, str(e)))

print(f"Checked all images. Corrupt files found: {len(corrupt)}")
for path, err in corrupt:
    print(" -", path, "|", err)