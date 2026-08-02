import os

base = "data/finetune_vehicle_signs/labels"
valid_classes = set(range(15))  # 0-14, matching your existing 15 classes

for fname in os.listdir(base):
    if fname == "classes.txt":
        continue
    path = os.path.join(base, fname)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            cls = int(line.split()[0])
            if cls not in valid_classes:
                print(f"INVALID class {cls} in {fname}")

print("Check complete.")