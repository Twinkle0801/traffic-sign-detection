import os
import yaml
from collections import defaultdict

base = "data/traffic_sign_dataset"
splits = ["train", "valid", "test"]

with open(os.path.join(base, "data.yaml")) as f:
    names = yaml.safe_load(f)["names"]

coverage = defaultdict(lambda: {s: 0 for s in splits})

for split in splits:
    lbl_dir = os.path.join(base, split, "labels")
    for fname in os.listdir(lbl_dir):
        with open(os.path.join(lbl_dir, fname)) as f:
            for line in f:
                if line.strip():
                    cls = int(line.split()[0])
                    coverage[cls][split] += 1

print(f"{'Class':<18}{'train':<10}{'valid':<10}{'test':<10}")
missing_flag = False
for cls in sorted(coverage.keys()):
    name = names[cls] if isinstance(names, list) else names.get(cls, str(cls))
    row = coverage[cls]
    print(f"{name:<18}{row['train']:<10}{row['valid']:<10}{row['test']:<10}")
    if row["valid"] == 0 or row["test"] == 0:
        missing_flag = True

if missing_flag:
    print("\n⚠ At least one class is missing from valid or test.")
else:
    print("\nAll classes present in every split.")