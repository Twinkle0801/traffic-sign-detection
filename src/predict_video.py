from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model.predict(
    source="data/custom_test/test_video.mp4",
    conf=0.25,
    save=True,
    project="runs/detect",
    name="video_inference",
    stream=True,
)

frame_count = 0
for r in results:
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed frame {frame_count}, detections: {len(r.boxes)}")

print(f"\nDone. Total frames processed: {frame_count}")