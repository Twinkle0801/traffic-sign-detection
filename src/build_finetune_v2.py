import os, random, shutil, yaml

random.seed(42)

FT_DIR = "data/finetune_v2"
TRAIN_IMG = f"{FT_DIR}/train/images"
TRAIN_LBL = f"{FT_DIR}/train/labels"
VAL_IMG = f"{FT_DIR}/valid/images"
VAL_LBL = f"{FT_DIR}/valid/labels"
for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
    os.makedirs(d, exist_ok=True)

def copy_pair(img_src, lbl_src, img_dst_dir, lbl_dst_dir, prefix=""):
    stem = os.path.splitext(os.path.basename(img_src))[0]
    ext = os.path.splitext(img_src)[1]
    shutil.copy(img_src, os.path.join(img_dst_dir, prefix + stem + ext))
    shutil.copy(lbl_src, os.path.join(lbl_dst_dir, prefix + stem + ".txt"))

# 1. Real vehicle-icon images (4 images) -- split 3 train / 1 valid
real_img_dir = "data/finetune_vehicle_signs/images"
real_lbl_dir = "data/finetune_vehicle_signs/labels"
real_files = [f for f in os.listdir(real_img_dir)]
random.shuffle(real_files)
real_train, real_val = real_files[:-1], real_files[-1:]

for fname in real_train:
    stem = os.path.splitext(fname)[0]
    lbl = os.path.join(real_lbl_dir, stem + ".txt")
    if os.path.exists(lbl):
        copy_pair(os.path.join(real_img_dir, fname), lbl, TRAIN_IMG, TRAIN_LBL, prefix="real_")
for fname in real_val:
    stem = os.path.splitext(fname)[0]
    lbl = os.path.join(real_lbl_dir, stem + ".txt")
    if os.path.exists(lbl):
        copy_pair(os.path.join(real_img_dir, fname), lbl, VAL_IMG, VAL_LBL, prefix="real_")

# 2. Synthetic images (v2) -- 90% train / 10% valid
synth_img_dir = "data/synthetic_vehicle_icons_v2/images"
synth_lbl_dir = "data/synthetic_vehicle_icons_v2/labels"
synth_files = os.listdir(synth_img_dir)
random.shuffle(synth_files)
split_point = int(len(synth_files) * 0.9)
for fname in synth_files[:split_point]:
    stem = os.path.splitext(fname)[0]
    copy_pair(os.path.join(synth_img_dir, fname), os.path.join(synth_lbl_dir, stem + ".txt"), TRAIN_IMG, TRAIN_LBL)
for fname in synth_files[split_point:]:
    stem = os.path.splitext(fname)[0]
    copy_pair(os.path.join(synth_img_dir, fname), os.path.join(synth_lbl_dir, stem + ".txt"), VAL_IMG, VAL_LBL)

# 3. Sample of original unmodified images (replay buffer to prevent forgetting)
orig_img_dir = "data/traffic_sign_dataset/train/images"
orig_lbl_dir = "data/traffic_sign_dataset/train/labels"
orig_files = os.listdir(orig_img_dir)
random.shuffle(orig_files)
N_ORIGINAL = 300
sampled_orig = orig_files[:N_ORIGINAL]
for fname in sampled_orig:
    stem = os.path.splitext(fname)[0]
    lbl = os.path.join(orig_lbl_dir, stem + ".txt")
    if os.path.exists(lbl):
        copy_pair(os.path.join(orig_img_dir, fname), lbl, TRAIN_IMG, TRAIN_LBL, prefix="orig_")

# Write data.yaml
with open("data/traffic_sign_dataset/data.yaml") as f:
    main_cfg = yaml.safe_load(f)

ft_yaml = {
    "train": os.path.abspath(TRAIN_IMG),
    "val": os.path.abspath(VAL_IMG),
    "nc": main_cfg["nc"],
    "names": main_cfg["names"],
}
with open(f"{FT_DIR}/data.yaml", "w") as f:
    yaml.dump(ft_yaml, f)

print(f"Train images: {len(os.listdir(TRAIN_IMG))}")
print(f"Valid images: {len(os.listdir(VAL_IMG))}")
print(f"Wrote {FT_DIR}/data.yaml")