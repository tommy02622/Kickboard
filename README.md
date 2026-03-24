# Kickboard Safety Detection (YOLOv8)

전동킥보드 안전 위반(헬멧 미착용, 2인 이상 탑승)을 자동 탐지하기 위한 프로젝트입니다.  
직접 수집한 이미지 + 인터넷 공개 데이터로 학습 데이터를 구축하고, YOLOv8을 학습한 뒤 라즈베리파이 온디바이스 환경에 배포하는 흐름을 정리했습니다.

## Highlights
- YOLOv8 기반 객체 탐지 모델 학습
- 안전 위반 시나리오(헬멧 미착용 / 2인 이상 탑승) 탐지
- ONNX/TFLite 변환 후 라즈베리파이 실행

## 폴더 구조
```
Kickboard/
  data/                # 데이터 설정 및 가이드
  notebooks/           # 학습 노트북
  scripts/             # 학습/추론/변환 스크립트
  deploy/              # 라즈베리파이 배포 가이드
  results/             # 학습 결과 저장 위치
```

## 빠른 시작
1) 의존성 설치
```
pip install -r requirements.txt
```

2) 데이터 경로 설정  
`data/data.yaml`의 `path`를 데이터셋 루트 경로로 수정합니다.

3) 학습
```
python scripts/train.py --data data/data.yaml --model yolov8n.pt --epochs 50 --imgsz 640
```

4) 추론
```
python scripts/predict.py --weights runs/detect/train/weights/best.pt --source path/to/images_or_video
```

5) 온디바이스 변환
```
python scripts/export_rpi.py --weights runs/detect/train/weights/best.pt --format onnx
```

## 데이터셋
`data/data.yaml`은 예시 템플릿입니다.
- 기본 클래스: `more-than-two`, `person`, `scooter`
- 헬멧 탐지를 추가하려면 `helmet`, `no-helmet` 클래스를 추가하세요.

로컬 경로 예시는 `data/data_local.example.yaml`에 있습니다.  
실제 경로 파일은 `data/data_local.yaml`로 두고 깃에는 올리지 않도록 `.gitignore`에 포함되어 있습니다.

## 온디바이스 배포
라즈베리파이 배포 흐름은 `deploy/raspberry_pi.md`에 정리되어 있습니다.

## 복원된 자료
기존 노트북과 yaml은 `notebooks/`와 `data/`에 포함되어 있습니다.
- `notebooks/kickboard.ipynb` (Roboflow API 키는 placeholder로 대체됨)

## 앞으로 추가하면 좋은 것들
- 성능 요약표(mAP/Precision/Recall)
- 데모 영상 및 스크린샷
- 라벨링 가이드 / 데이터 수집 정책

---
필요하면 결과 표/데모/실험 로그까지 붙여서 포트폴리오용 리포지토리로 완성해줄게요.
