from ultralytics import YOLO

model = YOLO("models/best.pt")

model.train(
    data="data/finetune_v2/data.yaml",
    epochs=20,
    imgsz=416,
    batch=8,
    device="cpu",
    project="runs/detect",
    name="finetune_v2",
    lr0=0.001,
    patience=5,
)

print("\nFine-tune v2 complete.")