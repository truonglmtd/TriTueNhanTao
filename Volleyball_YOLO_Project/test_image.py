from ultralytics import YOLO
from pathlib import Path
import sys

def find_model():
    candidates = [
        "runs/detect/runs/volleyball_yolov8/weights/best.pt",
        "runs/detect/runs/volleyball_yolov8/weights/last.pt",
        "yolo26n.pt",
        "yolov8s.pt",
        "yolov8n.pt",
    ]
    for c in candidates:
        if Path(c).exists():
            return str(Path(c))
    return None

model_path = find_model()
if model_path is None:
    print("Không tìm thấy model. Vui lòng đặt 'yolo26n.pt' hoặc cung cấp model hợp lệ.")
    sys.exit(1)

model = YOLO(model_path)

results = model.predict(
    source="dataset/images/val",
    save=True,
    conf=0.25
)

print("Done")