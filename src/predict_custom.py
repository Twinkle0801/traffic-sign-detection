from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model.predict(
    source="data/custom_test",
    conf=0.25,
    save=True,
    project="runs/detect",
    name="custom_inference",
)

for r in results:
    print(f"\n{r.path}")
    if len(r.boxes) == 0:
        print("  No detections")
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = r.names[cls_id]
        print(f"  {name}: {conf:.2f} confidence")