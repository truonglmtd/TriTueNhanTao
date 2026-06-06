import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO(
    "runs/detect/runs/volleyball_yolov85/weights/best.pt"
)

st.title("Hệ thống phát hiện người chơi, trọng tài và bóng bằng YOLOv8")

uploaded_file = st.file_uploader(
    "Tải ảnh lên",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    results = model(image)

    result_img = results[0].plot()

    st.image(result_img)