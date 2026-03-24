# Dataset Notes

현재 복원된 `data.yaml` 기준 클래스:
- `more-than-two`: 2인 이상 탑승
- `person`: 사람
- `scooter`: 킥보드

헬멧 미착용 탐지를 추가하려면:
1. 라벨 클래스 추가: `helmet`, `no-helmet`
2. `data.yaml`의 `nc`와 `names`를 업데이트
3. 라벨링된 데이터로 재학습

데이터 경로는 환경에 맞게 `data.yaml`의 `train/val/test`를 수정하세요.
