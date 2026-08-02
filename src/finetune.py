from ultralytics import YOLO
import shutil, os

# Fine-tuning requires a proper YOLO dataset structure (train/valid folders + data.yaml),
# not just a flat images/labels folder. Since we only have 4 images, we'll use the SAME
# 4 images for both train and valid -- not ideal for evaluation, but fine for a small
# targeted fine-tune whose goal is just exposing the model to this new visual style.

FT_BASE = "data/finetune_vehicle_signs"
FT_TRAIN_IMG = f"{FT_BASE}/ft_dataset/train/images"
FT_TRAIN_LBL = f"{FT_BASE}/ft_dataset/train/labels"
FT_VAL_IMG = f"{FT_BASE}/ft_dataset/valid/images"
FT_VAL_LBL = f"{FT_BASE}/ft_dataset/valid/labels"

for d in [FT_TRAIN_IMG, FT_TRAIN_LBL, FT_VAL_IMG, FT_VAL_LBL]:
    os.makedirs(d, exist_ok=True)

src_img_dir = f"{FT_BASE}/images"
src_lbl_dir = f"{FT_BASE}/labels"

for fname in os.listdir(src_img_dir):
    stem = os.path.splitext(fname)[0]
    lbl_name = stem + ".txt"
    if not os.path.exists(os.path.join(src_lbl_dir, lbl_name)):
        continue
    shutil.copy(os.path.join(src_img_dir, fname), os.path.join(FT_TRAIN_IMG, fname))
    shutil.copy(os.path.join(src_lbl_dir, lbl_name), os.path.join(FT_TRAIN_LBL, lbl_name))
    shutil.copy(os.path.join(src_img_dir, fname), os.path.join(FT_VAL_IMG, fname))
    shutil.copy(os.path.join(src_lbl_dir, lbl_name), os.path.join(FT_VAL_LBL, lbl_name))

# Write a data.yaml for this fine-tune dataset using the SAME 15 classes as the main dataset
import yaml
with open("data/traffic_sign_dataset/data.yaml") as f:
    main_cfg = yaml.safe_load(f)

ft_yaml = {
    "train": os.path.abspath(FT_TRAIN_IMG),
    "val": os.path.abspath(FT_VAL_IMG),
    "nc": main_cfg["nc"],
    "names": main_cfg["names"],
}
ft_yaml_path = f"{FT_BASE}/ft_data.yaml"
with open(ft_yaml_path, "w") as f:
    yaml.dump(ft_yaml, f)

print(f"Fine-tune dataset prepared: {len(os.listdir(FT_TRAIN_IMG))} images")
print(f"Wrote {ft_yaml_path}")

# Continue training from the existing best.pt (not from scratch, not from yolov8n.pt)
model = YOLO("models/best.pt")

model.train(
    data=ft_yaml_path,
    epochs=10,
    imgsz=416,
    batch=4,
    device="cpu",
    project="runs/detect",
    name="finetune_vehicle_signs",
    lr0=0.0005,       # smaller learning rate -- gentle nudge, not a full retrain
    patience=0,        # disable early stopping for this tiny run
)

print("\nFine-tuning complete. New weights at runs/detect/finetune_vehicle_signs/weights/best.pt")