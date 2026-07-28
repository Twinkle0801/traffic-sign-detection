import os
import random
import cv2
import yaml

base = "data/traffic_sign_dataset"
split = "train"
img_dir = os.path.join(base, split, "images")
lbl_dir = os.path.join(base, split, "labels")

with open(os.path.join(base, "data.yaml")) as f:
    names = yaml.safe_load(f)["names"]

sample_files = random.sample(os.listdir(img_dir), 6)
os.makedirs("data/preview", exist_ok=True)

for fname in sample_files:
    img_path = os.path.join(img_dir, fname)
    lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                if not line.strip():
                    continue
                cls, xc, yc, bw, bh = map(float, line.split())
                cls = int(cls)
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                label = names[cls] if isinstance(names, list) else names.get(cls, str(cls))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    out_path = os.path.join("data/preview", fname)
    cv2.imwrite(out_path, img)
    print("Saved:", out_path)