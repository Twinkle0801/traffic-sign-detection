
from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model.predict(
    source=0,
    conf=0.25,
    show=True,
    stream=True,
)

for r in results:
    pass  # the loop itself is what drives frame-by-frame processing and keeps the window open

