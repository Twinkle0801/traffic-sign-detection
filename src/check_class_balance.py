import os
import yaml
from collections import Counter

base = "data/traffic_sign_dataset"
with open(os.path.join(base, "data.yaml")) as f:
    data_cfg = yaml.safe_load(f)

names = data_cfg["names"]
counter = Counter()

for split in ["train", "valid", "test"]:
    lbl_dir = os.path.join(base, split, "labels")
    for fname in os.listdir(lbl_dir):
        with open(os.path.join(lbl_dir, fname)) as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    counter[class_id] += 1

print(f"{'Class':<20}{'Count'}")
for class_id, count in sorted(counter.items()):
    name = names[class_id] if isinstance(names, list) else names.get(class_id, "UNKNOWN")
    print(f"{name:<20}{count}")