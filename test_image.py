from ultralytics import YOLO

model = YOLO("runs/detect/runs/volleyball_yolov85/weights/best.pt")

results = model.predict(
    source="dataset/images/val",
    save=True,
    conf=0.25
)

print("Done")