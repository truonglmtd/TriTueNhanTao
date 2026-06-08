import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def find_default_weights():
    candidates = [
        "runs/detect/runs/volleyball_yolov8/weights/best.pt",
        "runs/detect/runs/volleyball_yolov8/weights/last.pt",
        "yolov8s.pt",
        "yolov8n.pt",
        "yolo26n.pt",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p)
    return "yolov8s.pt"


def main():
    parser = argparse.ArgumentParser(description="Train the volleyball YOLO model")
    parser.add_argument("--weights", default=None, help="Initial weights file or backbone model")
    parser.add_argument("--data", default="data.yaml", help="Dataset YAML config")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--project", default="runs", help="Project output directory")
    parser.add_argument("--name", default="volleyball_yolov8", help="Training run name")
    parser.add_argument("--device", default=None, help="Device to use, e.g. 0 or cpu")
    parser.add_argument("--workers", type=int, default=0, help="Number of data loader workers")
    args = parser.parse_args()

    if args.device is None:
        args.device = "0" if torch.cuda.is_available() else "cpu"
    else:
        if args.device != "cpu":
            if args.device.isdigit() and not torch.cuda.is_available():
                print("CUDA không khả dụng. Chuyển sang device=cpu.")
                args.device = "cpu"
            elif args.device.lower() in {"cuda", "gpu"} and not torch.cuda.is_available():
                print("CUDA không khả dụng. Chuyển sang device=cpu.")
                args.device = "cpu"
            elif args.device.lower() == "cuda" and torch.cuda.is_available():
                args.device = "0"

    model_path = args.weights or find_default_weights()
    print(f"Using weights: {model_path}")
    model = YOLO(model_path)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()