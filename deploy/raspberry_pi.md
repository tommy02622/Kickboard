# Raspberry Pi Deployment (Guide)

이 가이드는 YOLOv8 모델을 라즈베리파이에 올리기 위한 최소 흐름입니다.

## 1) 모델 내보내기
```
python scripts/export_rpi.py --weights runs/detect/train/weights/best.pt --format onnx
```

## 2) 라즈베리파이 환경
```
sudo apt update
sudo apt install python3-pip
pip3 install onnxruntime opencv-python
```

## 3) 추론 실행
간단한 추론 스크립트는 `scripts/predict.py`를 참고해 구성할 수 있습니다.  
필요하면 라즈베리파이 전용 `infer.py`도 만들어줄게요.

## 참고
- TFLite 변환은 추가 패키지 설치가 필요할 수 있습니다.
- 성능 최적화를 위해 이미지 크기 축소(예: 416) 또는 `yolov8n` 모델 사용을 권장합니다.
