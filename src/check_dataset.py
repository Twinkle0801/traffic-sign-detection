import os

base = "data/traffic_sign_dataset"
for split in ["train", "valid", "test"]:
    img_dir = os.path.join(base, split, "images")
    lbl_dir = os.path.join(base, split, "labels")
    if not os.path.exists(img_dir):
        print(f"{split}: MISSING folder")
        continue
    n_img = len(os.listdir(img_dir))
    n_lbl = len(os.listdir(lbl_dir))
    print(f"{split}: {n_img} images, {n_lbl} labels")