from ultralytics import YOLO
import yaml

model = YOLO("models/best.pt")

with open("data/traffic_sign_dataset/data.yaml") as f:
    names = yaml.safe_load(f)["names"]

metrics = model.val(data="data/traffic_sign_dataset/data.yaml", split="val")

print("\nOverall:")
print(f"  mAP50:    {metrics.box.map50:.3f}")
print(f"  mAP50-95: {metrics.box.map:.3f}")
print(f"  Precision:{metrics.box.mp:.3f}")
print(f"  Recall:   {metrics.box.mr:.3f}")

print("\nPer-class AP50 (only classes with validation instances are scored):")
class_maps = metrics.box.ap50
class_indices = metrics.box.ap_class_index  # the actual class IDs these scores belong to

for idx, ap in zip(class_indices, class_maps):
    idx = int(idx)
    name = names[idx] if isinstance(names, list) else names.get(idx, str(idx))
    print(f"  {name:<18} {ap:.3f}")

all_class_ids = set(names.keys()) if isinstance(names, dict) else set(range(len(names)))
scored_ids = set(int(i) for i in class_indices)
missing = all_class_ids - scored_ids
if missing:
    print("\nClasses with no validation instances (not scored):")
    for idx in sorted(missing):
        name = names[idx] if isinstance(names, list) else names.get(idx, str(idx))
        print(f"  {name}")