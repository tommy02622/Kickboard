import argparse
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for kickboard safety detection")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model or checkpoint")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="", help="Device, e.g., 0, 0,1, or cpu")
    parser.add_argument("--project", default="runs/detect", help="Project directory")
    parser.add_argument("--name", default="train", help="Run name")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
