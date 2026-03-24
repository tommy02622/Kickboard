import argparse
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLOv8 model for on-device deployment")
    parser.add_argument("--weights", required=True, help="Path to trained model weights")
    parser.add_argument("--format", default="onnx", choices=["onnx", "tflite"], help="Export format")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.export(format=args.format, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
