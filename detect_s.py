import cv2
from ultralytics import YOLO
import pyttsx3
import time
import threading

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Load YOLOv8 model
model = YOLO('best.pt')

cap = cv2.VideoCapture("vid1x.mp4")

min_confidence = 0.5
speak_delay = 0.5  # seconds

spoken_objects = {}   # class_name -> last spoken timestamp

# Speak function (non-blocking)
def speak_text(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait()).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    # Define center region (40% width and height)
    center_x1 = int(w * 0.3)
    center_y1 = int(h * 0.3)
    center_x2 = int(w * 0.7)
    center_y2 = int(h * 0.7)

    # Draw center region rectangle
    cv2.rectangle(frame, (center_x1, center_y1), (center_x2, center_y2), (255, 0, 0), 2)

    results = model(frame)[0]
    current_time = time.time()
    speak_list = []

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < min_confidence:
                continue

            class_name = model.names[cls_id]
            class_name = class_name.replace("pedestrian", "human")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Compute center of object
            obj_cx = (x1 + x2) // 2
            obj_cy = (y1 + y2) // 2

            # Only speak if center is inside the center region
            if center_x1 <= obj_cx <= center_x2 and center_y1 <= obj_cy <= center_y2:
                if class_name not in spoken_objects or current_time - spoken_objects[class_name] > speak_delay:
                    speak_list.append(class_name)
                    spoken_objects[class_name] = current_time

    if speak_list:
        sentence = ', '.join(speak_list)
        speak_text(sentence)

    cv2.imshow('YOLOv8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
