from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/runs/volleyball_yolov85/weights/best.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Không mở được camera. Thử đổi 0 thành 1.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không lấy được frame từ camera.")
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("Volleyball Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()