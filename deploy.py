import cv2
import time
import os
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Load your PRODUCTION model (The one we just saved)
MODEL_PATH = "models/cardd_yolo11s_1024_final.pt"

# 2. Input Source
# Option A: Webcam (Best for live demo) -> Set to 0
# Option B: Video File (Best for thesis video) -> Set to "test_video.mp4"
VIDEO_SOURCE = 0

# 3. Output Settings
OUTPUT_PATH = "deployment_demo_output.mp4"
CONF_THRESHOLD = 0.4  # Confidence threshold (0.25 is default, 0.4 is cleaner)

# ==========================================
# SETUP
# ==========================================
print(f"[INFO] Loading model: {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found at {MODEL_PATH}")
    print("Did you run the save script in the notebook?")
    exit()

model = YOLO(MODEL_PATH)
print("[SUCCESS] Model loaded!")

# Setup Video Capture
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"[ERROR] Could not open video source: {VIDEO_SOURCE}")
    print("Try setting VIDEO_SOURCE = 0 for webcam.")
    exit()

# Setup Video Writer (to save the result)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_input = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_input, (width, height))

print("-" * 50)
print("[INFO] Starting Inference Loop...")
print("[INFO] Press 'Q' to quit.")
print("-" * 50)

# ==========================================
# MAIN LOOP
# ==========================================
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break # End of video
        
    # 1. Run Inference
    start = time.time()
    results = model(frame, imgsz=1024, conf=CONF_THRESHOLD, verbose=False)
    
    # 2. Draw Bounding Boxes
    annotated_frame = results[0].plot()
    
    # 3. Calculate FPS (Real-time speed)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    
    # 4. Draw FPS on Screen (Crucial for Thesis)
    # Green text with black outline for visibility
    cv2.putText(annotated_frame, f"Model: Exp4 (1024px)", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 5. Show & Save
    cv2.imshow("CarDD Deployment System", annotated_frame)
    out.write(annotated_frame)
    
    # Quit on 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[INFO] Output saved to: {OUTPUT_PATH}")
