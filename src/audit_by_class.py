import os
import random
import cv2
import yaml
from collections import defaultdict

base = "data/traffic_sign_dataset"
split = "train"
img_dir = os.path.join(base, split, "images")
lbl_dir = os.path.join(base, split, "labels")

with open(os.path.join(base, "data.yaml")) as f:
    names = yaml.safe_load(f)["names"]

# group image filenames by which classes they contain
by_class = defaultdict(list)
for fname in os.listdir(lbl_dir):
    stem = os.path.splitext(fname)[0]
    with open(os.path.join(lbl_dir, fname)) as f:
        classes_in_file = {int(line.split()[0]) for line in f if line.strip()}
    for cls in classes_in_file:
        by_class[cls].append(stem)

os.makedirs("data/preview_by_class", exist_ok=True)

SAMPLES_PER_CLASS = 3
for cls, stems in by_class.items():
    name = names[cls] if isinstance(names, list) else names.get(cls, str(cls))
    safe_name = name.replace(" ", "_")
    chosen = random.sample(stems, min(SAMPLES_PER_CLASS, len(stems)))
    for stem in chosen:
        img_path = os.path.join(img_dir, stem + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, stem + ".png")
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        with open(lbl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                c, xc, yc, bw, bh = map(float, line.split())
                c = int(c)
                x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
                x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
                lbl_name = names[c] if isinstance(names, list) else names.get(c, str(c))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, lbl_name, (x1, max(y1-5,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        out_path = f"data/preview_by_class/{safe_name}_{stem}.jpg"
        cv2.imwrite(out_path, img)

print(f"Saved samples for {len(by_class)} classes to data/preview_by_class/")