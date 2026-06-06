from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')

    model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        project='runs',
        name='volleyball_yolov8',
        device=0,
        workers=0
    )

if __name__ == "__main__":
    main()