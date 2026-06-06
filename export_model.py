from ultralytics import YOLO

model = YOLO('runs/volleyball_yolov8/weights/best.pt')
model.export(format='onnx')
