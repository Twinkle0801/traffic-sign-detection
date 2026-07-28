import os

base = "data/traffic_sign_dataset"
splits = ["train", "valid", "test"]
seen = set()

for split in splits:
    lbl_dir = os.path.join(base, split, "labels")
    for fname in os.listdir(lbl_dir):
        with open(os.path.join(lbl_dir, fname)) as f:
            for line in f:
                if line.strip():
                    seen.add(int(line.split()[0]))

print("Unique class IDs found:", sorted(seen))
print("Total distinct classes:", len(seen))
print("Max class ID:", max(seen))
print("Required nc value:", max(seen) + 1)