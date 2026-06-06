from ultralytics import YOLO

# Đổi đường dẫn ảnh/video cần test ở biến source
model = YOLO('runs/volleyball_yolov8/weights/best.pt')

model.predict(
    source='sample_data',
    imgsz=640,
    conf=0.25,
    save=True,
    project='outputs',
    name='predict_result'
)
