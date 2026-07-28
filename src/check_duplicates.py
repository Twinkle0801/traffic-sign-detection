import os
import hashlib

base = "data/traffic_sign_dataset"
splits = ["train", "valid", "test"]
hashes = {}
duplicates = []

for split in splits:
    img_dir = os.path.join(base, split, "images")
    for fname in os.listdir(img_dir):
        path = os.path.join(img_dir, fname)
        with open(path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h in hashes:
            duplicates.append((path, hashes[h]))
        else:
            hashes[h] = path

print(f"Duplicate images found: {len(duplicates)}")
for a, b in duplicates:
    print(" -", a, "is a duplicate of", b)