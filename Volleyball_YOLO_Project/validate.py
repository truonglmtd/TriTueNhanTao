from ultralytics import YOLO

model = YOLO('runs/volleyball_yolov8/weights/best.pt')
metrics = model.val(data='data.yaml', imgsz=640)
print(metrics)
