import os
import yaml

base = "data/traffic_sign_dataset"
splits = ["train", "valid", "test"]

with open(os.path.join(base, "data.yaml")) as f:
    data_cfg = yaml.safe_load(f)
num_classes = data_cfg["nc"]

errors = []

for split in splits:
    lbl_dir = os.path.join(base, split, "labels")
    for fname in os.listdir(lbl_dir):
        path = os.path.join(lbl_dir, fname)
        with open(path) as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{split}/{fname} line {line_num}: expected 5 values, got {len(parts)}")
                    continue
                try:
                    cls = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                except ValueError:
                    errors.append(f"{split}/{fname} line {line_num}: non-numeric value")
                    continue
                if cls < 0 or cls >= num_classes:
                    errors.append(f"{split}/{fname} line {line_num}: class id {cls} out of range (0-{num_classes-1})")
                if not all(0.0 <= c <= 1.0 for c in coords):
                    errors.append(f"{split}/{fname} line {line_num}: coordinate out of [0,1] range -> {coords}")

print(f"Checked all label files. Errors found: {len(errors)}\n")
for e in errors[:50]:
    print(" -", e)
if len(errors) > 50:
    print(f"... and {len(errors) - 50} more")