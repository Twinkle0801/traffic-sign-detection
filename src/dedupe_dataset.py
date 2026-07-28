import os
import hashlib

base = "data/traffic_sign_dataset"
# Order matters: we check train first so train images are always the ones KEPT
splits = ["train", "valid", "test"]

hashes = {}       # hash -> (split, filepath_without_ext)
to_delete = []     # list of (split, filepath_without_ext)

for split in splits:
    img_dir = os.path.join(base, split, "images")
    for fname in os.listdir(img_dir):
        path = os.path.join(img_dir, fname)
        with open(path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        stem = os.path.splitext(fname)[0]
        if h in hashes:
            # already seen this exact image earlier (train wins, since it's checked first)
            to_delete.append((split, stem, fname))
        else:
            hashes[h] = (split, stem)

print(f"Found {len(to_delete)} duplicate images to remove.\n")

removed_images = 0
removed_labels = 0

for split, stem, fname in to_delete:
    img_path = os.path.join(base, split, "images", fname)
    lbl_path = os.path.join(base, split, "labels", stem + ".txt")

    if os.path.exists(img_path):
        os.remove(img_path)
        removed_images += 1
    if os.path.exists(lbl_path):
        os.remove(lbl_path)
        removed_labels += 1

print(f"Removed {removed_images} duplicate images and {removed_labels} matching label files.")