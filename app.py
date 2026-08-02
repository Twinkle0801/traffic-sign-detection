import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time

st.set_page_config(page_title="Traffic Sign Detection", layout="centered")
st.title("Traffic Sign Detection")
st.write("Upload a road image to detect traffic signs using a YOLOv8 model trained on 15 sign classes.")

@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Signs"):
        start = time.time()
        results = model.predict(image, conf=0.25)
        elapsed = (time.time() - start) * 1000

        r = results[0]
        annotated = r.plot()  # numpy array (BGR) with boxes drawn

        with col2:
            st.image(annotated, caption="Detection Result", use_container_width=True, channels="BGR")

        st.subheader("Detected Signs")
        if len(r.boxes) == 0:
            st.write("No signs detected.")
        else:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = r.names[cls_id]
                st.write(f"**{name}** — {conf:.1%} confidence")

        st.caption(f"Inference time: {elapsed:.1f} ms")

        from io import BytesIO
        import cv2

        _, buffer = cv2.imencode(".jpg", annotated)
        st.download_button(
            label="Download Result",
            data=BytesIO(buffer.tobytes()),
            file_name="detection_result.jpg",
            mime="image/jpeg",
        )