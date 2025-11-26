import cv2
import numpy as np
from ultralytics import YOLO

# 1. YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

# 2. 비디오 파일 열기
video_path = "C:/Users/gunhu/dev/14637596_2160_3840_24fps.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"오류: {video_path}를 열 수 없습니다.")
    exit()

# 3. 위험 구역(다각형) 정의
danger_zone = np.array([
    [6, 2404],
    [651, 2424],
    [627, 3012],
    [6, 3019],
], np.int32)

# 4. 영상 프레임 처리 루프
while True:
    ret, frame = cap.read()
    if not ret:
        print("영상의 끝입니다.")
        break

    # 5. YOLOv8 객체 감지 수행 (person 클래스만)
    results = model(frame, classes=[0], verbose=False)

    is_danger = False

    # 감지된 객체 정보 처리
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_point = (int((x1 + x2) / 2), y2)

            # 6. 기준점이 위험 구역 내에 있는지 확인
            is_inside = cv2.pointPolygonTest(danger_zone, person_point, False)
            box_color = (0, 255, 0) # Green

            if is_inside >= 0:
                is_danger = True
                box_color = (0, 0, 255) # Red
                cv2.circle(frame, person_point, 7, box_color, -1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, 'Person', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # 7. 시각화: 위험 구역 그리기
    cv2.polylines(frame, [danger_zone], isClosed=True, color=(255, 0, 0), thickness=2)

    # 8. 위험 알림 표시
    if is_danger:
        cv2.putText(frame, "!!! DANGER ZONE ALERT !!!", (50, 50), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)

    # 9. 결과 화면 출력 (🚨 수정된 부분)
    # 원본 프레임(frame)이 너무 크므로, 화면에 보여줄 크기로 축소합니다.
    
    # (1) 보여주고 싶은 창의 가로 크기 지정 (예: 1280 픽셀)
    display_width = 640
    
    # (2) 원본 영상의 비율에 맞게 세로 크기 계산
    try:
        original_height, original_width = frame.shape[:2]
        aspect_ratio = original_height / original_width
        display_height = int(display_width * aspect_ratio)
    except Exception as e:
        print(f"프레임 크기 계산 오류: {e} - 프레임을 찾을 수 없습니다.")
        break # 프레임이 비었으면 루프 종료

    # (3) 프레임 리사이즈
    frame_resized = cv2.resize(frame, (display_width, display_height))

    # (4) 축소된 프레임(frame_resized)을 화면에 표시
    cv2.imshow('Danger Zone Detection', frame_resized) 

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 10. 자원 해제
cap.release()
cv2.destroyAllWindows()