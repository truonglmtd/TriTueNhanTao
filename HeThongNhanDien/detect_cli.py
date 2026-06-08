import argparse
import sys
from pathlib import Path

def find_default_model():
    candidates = [
        "runs/detect/runs/volleyball_yolov8/weights/best.pt",
        "runs/detect/runs/volleyball_yolov8/weights/last.pt",
        "yolo26n.pt",
        "yolov8s.pt",
        "yolov8n.pt",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p)
    return None

def main():
    parser = argparse.ArgumentParser(description="Simple YOLO detect CLI for this project")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--source", type=str, default="runs/detect/predict", help="Image or folder to run prediction on")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--iou", type=float, default=0.25, help="IoU threshold")
    parser.add_argument("--max_det", type=int, default=200, help="Maximum number of detections")
    parser.add_argument("--save", action="store_true", help="Save prediction outputs")
    parser.add_argument("--dry-run", action="store_true", help="Only load model and exit")

    args = parser.parse_args()

    model_path = args.model or find_default_model()
    if model_path is None:
        print("Không tìm thấy file model. Vui lòng cung cấp --model hoặc đặt 'yolo26n.pt' vào thư mục dự án.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Không thể import ultralytics:", e)
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    print("Model loaded.")

    if args.dry_run:
        print("Dry-run complete. Model is usable.")
        return

    print(f"Running prediction on: {args.source}")
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save=args.save,
    )

    print("Prediction finished.")
    try:
        for r in results:
            path = getattr(r, "path", None)
            if path:
                print("Result path:", path)
    except Exception:
        pass

if __name__ == "__main__":
    main()
