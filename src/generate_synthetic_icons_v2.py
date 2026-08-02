"""
Generates synthetic vehicle-icon speed sign examples -- v2 with automated
filtering to skip sign variants that don't look like a plain red-bordered
circular number sign (e.g. RAPPEL panels, camera-zone signs, mis-cropped boxes).
"""
import os
import random
import cv2
import numpy as np
import yaml

random.seed(42)

BASE = "data/traffic_sign_dataset"
SPLIT = "train"
IMG_DIR = os.path.join(BASE, SPLIT, "images")
LBL_DIR = os.path.join(BASE, SPLIT, "labels")

OUT_DIR = "data/synthetic_vehicle_icons_v2"
OUT_IMG_DIR = os.path.join(OUT_DIR, "images")
OUT_LBL_DIR = os.path.join(OUT_DIR, "labels")
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

with open(os.path.join(BASE, "data.yaml")) as f:
    names = yaml.safe_load(f)["names"]

SPEED_LIMIT_CLASSES = set(range(2, 14))
N_SAMPLES = 150
MIN_RED_RATIO = 0.08  # minimum fraction of border pixels that must look "red"


def has_red_border(img, x1, y1, x2, y2):
    """Check the outer ring of the box for red pixels (real speed sign border)."""
    crop = img[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return False
    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return False

    # Sample a thin ring around the outer 20% of the crop
    border = max(2, int(min(h, w) * 0.12))
    mask = np.zeros((h, w), dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Red hue wraps around 0/180 in OpenCV's HSV
    red_mask = ((h_ch < 10) | (h_ch > 170)) & (s_ch > 80) & (v_ch > 50)
    border_pixels = mask.sum()
    red_border_pixels = (mask & red_mask).sum()

    if border_pixels == 0:
        return False
    return (red_border_pixels / border_pixels) >= MIN_RED_RATIO


def draw_car_icon(img, x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    body_top = y1 + int(h * 0.10)
    body_bottom = y1 + int(h * 0.50)
    cv2.rectangle(img, (x1 + int(w * 0.18), body_top), (x1 + int(w * 0.82), body_bottom), (30, 30, 30), -1)
    wheel_r = max(2, int(h * 0.05))
    cv2.circle(img, (x1 + int(w * 0.32), body_bottom), wheel_r, (20, 20, 20), -1)
    cv2.circle(img, (x1 + int(w * 0.68), body_bottom), wheel_r, (20, 20, 20), -1)


def draw_truck_icon(img, x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    top = y1 + int(h * 0.08)
    bottom = y1 + int(h * 0.50)
    cv2.rectangle(img, (x1 + int(w * 0.12), top), (x1 + int(w * 0.55), bottom), (25, 25, 25), -1)
    cv2.rectangle(img, (x1 + int(w * 0.55), top + int(h * 0.05)), (x1 + int(w * 0.88), bottom), (25, 25, 25), -1)
    wheel_r = max(2, int(h * 0.045))
    cv2.circle(img, (x1 + int(w * 0.28), bottom), wheel_r, (15, 15, 15), -1)
    cv2.circle(img, (x1 + int(w * 0.70), bottom), wheel_r, (15, 15, 15), -1)


ICON_FUNCS = [draw_car_icon, draw_truck_icon]  # dropped the odd-looking motorcycle icon

candidates = []
for fname in os.listdir(LBL_DIR):
    lbl_path = os.path.join(LBL_DIR, fname)
    stem = os.path.splitext(fname)[0]
    img_path = None
    for ext in [".jpg", ".png", ".jpeg"]:
        p = os.path.join(IMG_DIR, stem + ext)
        if os.path.exists(p):
            img_path = p
            break
    if img_path is None:
        continue
    with open(lbl_path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            cls = int(parts[0])
            if cls in SPEED_LIMIT_CLASSES:
                candidates.append((img_path, stem, line.strip()))

print(f"Found {len(candidates)} speed-limit sign instances to check.")
random.shuffle(candidates)

count = 0
checked = 0
for img_path, stem, label_line in candidates:
    if count >= N_SAMPLES:
        break
    checked += 1

    img = cv2.imread(img_path)
    if img is None:
        continue
    h_img, w_img = img.shape[:2]

    cls, xc, yc, bw, bh = label_line.split()
    cls, xc, yc, bw, bh = int(cls), float(xc), float(yc), float(bw), float(bh)

    x1 = int((xc - bw / 2) * w_img)
    y1 = int((yc - bh / 2) * h_img)
    x2 = int((xc + bw / 2) * w_img)
    y2 = int((yc + bh / 2) * h_img)

    if (x2 - x1) < 25 or (y2 - y1) < 25:
        continue

    if not has_red_border(img, x1, y1, x2, y2):
        continue  # skip variants without a clean red border (RAPPEL, camera signs, bad crops, etc.)

    icon_fn = random.choice(ICON_FUNCS)
    icon_fn(img, x1, y1, x2, y2)

    out_stem = f"synth_{count:04d}_{stem}"
    cv2.imwrite(os.path.join(OUT_IMG_DIR, out_stem + ".jpg"), img)
    with open(os.path.join(OUT_LBL_DIR, out_stem + ".txt"), "w") as f:
        f.write(label_line + "\n")
    count += 1

print(f"\nChecked {checked} candidates, kept {count} with a valid red border.")
print(f"Generated {count} synthetic vehicle-icon examples in {OUT_DIR}/")