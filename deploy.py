import cv2
import time
import os
from ultralytics import YOLO

MODEL_PATH = "models/cardd_yolo11s_1024_final.pt"

VIDEO_SOURCE = 0

OUTPUT_PATH = "deployment_demo_output.mp4"
CONF_THRESHOLD = 0.4  

if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")
    exit()

model = YOLO(MODEL_PATH)
print("Model successfully loaded")

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Could not open video source: {VIDEO_SOURCE}")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_input, (width, height))


prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break 
        
    start = time.time()
    results = model(frame, imgsz=1024, conf=CONF_THRESHOLD, verbose=False)
    
    annotated_frame = results[0].plot()
    
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    
    cv2.putText(annotated_frame, f"Model: Exp4 (1024px)", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("CarDD Deployment System", annotated_frame)
    out.write(annotated_frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output saved to: {OUTPUT_PATH}")
