from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # pretrained nano model — smallest, fastest YOLOv8 variant

    model.train(
        data="data/traffic_sign_dataset/data.yaml",
        epochs=30,
        imgsz=416,      # dataset images are already 416x416, no need to upscale
        batch=16,
        device="cpu",
        project="runs/detect",
        name="train",
    )

if __name__ == "__main__":
    main()