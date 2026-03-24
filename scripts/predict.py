import argparse
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained YOLOv8 model")
    parser.add_argument("--weights", required=True, help="Path to trained model weights")
    parser.add_argument("--source", required=True, help="Image, video, webcam, or directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="", help="Device, e.g., 0 or cpu")
    parser.add_argument("--save", action="store_true", help="Save results")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=args.save,
    )


if __name__ == "__main__":
    main()
