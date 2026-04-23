import cv2
import os
import time
import numpy as np
from picamera2 import Picamera2

# === CONFIGURATION ===
CHECKERBOARD_SIZE = (8, 6) 

NUM_FRAMES = 40    
WIDTH, HEIGHT = 800, 800
CAPTURE_DELAY = 3.0   # Seconds to wait between valid captures

# CAM0 is mathematically the LEFT camera, CAM1 is the RIGHT camera
CAM0_LEFT_DIR = "cam0_calib_0305_4"
CAM1_RIGHT_DIR = "cam1_calib_0305_4"

os.makedirs(CAM0_LEFT_DIR, exist_ok=True)
os.makedirs(CAM1_RIGHT_DIR, exist_ok=True)

# === INITIALIZE CAMERAS ===
print("Initializing cameras...")
# You noted: picam0 is RIGHT, picam1 is LEFT
picam_right = Picamera2(0) 
picam_left = Picamera2(1)

for p in [picam_right, picam_left]:
    p.configure(p.create_preview_configuration(main={"size": (WIDTH, HEIGHT)}))
    p.start()

print(f"✅ Searching for {CHECKERBOARD_SIZE} inner corners.")
print("  - GREEN lines = Good detection.")
print("  - PRESS 's' to start/stop Auto-Save mode.")
print("  - PRESS 'q' to quit.")

capturing = False
frame_count = 0
last_capture_time = 0

detect_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_NORMALIZE_IMAGE + 
                cv2.CALIB_CB_FILTER_QUADS)

try:
    while True:
        # 1. Capture Frames
        frame_left = cv2.cvtColor(picam_left.capture_array(), cv2.COLOR_RGB2BGR)
        frame_right = cv2.cvtColor(picam_right.capture_array(), cv2.COLOR_RGB2BGR)

        # 2. Convert to Grayscale
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # 3. Detect Corners (Robust Mode)
        ret_l, corners_l = cv2.findChessboardCorners(gray_left, CHECKERBOARD_SIZE, detect_flags)
        ret_r, corners_r = cv2.findChessboardCorners(gray_right, CHECKERBOARD_SIZE, detect_flags)

        # 4. Visual Feedback
        vis_left = frame_left.copy()
        vis_right = frame_right.copy()

        if ret_l:
            cv2.drawChessboardCorners(vis_left, CHECKERBOARD_SIZE, corners_l, ret_l)
        if ret_r:
            cv2.drawChessboardCorners(vis_right, CHECKERBOARD_SIZE, corners_r, ret_r)

        # 5. Save Logic (Only if BOTH see the board)
        current_time = time.time()
        
        if capturing:
            if ret_l and ret_r:
                if (current_time - last_capture_time >= CAPTURE_DELAY):
                    frame_count += 1
                    
                    # Save Left to CAM0, Right to CAM1
                    cv2.imwrite(os.path.join(CAM0_LEFT_DIR, f"img_{frame_count:02d}.jpg"), frame_left)
                    cv2.imwrite(os.path.join(CAM1_RIGHT_DIR, f"img_{frame_count:02d}.jpg"), frame_right)
                    
                    last_capture_time = current_time
                    print(f"📸 Saved pair {frame_count}/{NUM_FRAMES}")
                    
                    # Flash Effect
                    cv2.rectangle(vis_left, (0,0), (WIDTH, HEIGHT), (255, 255, 255), 10)
                    cv2.rectangle(vis_right, (0,0), (WIDTH, HEIGHT), (255, 255, 255), 10)
            else:
                cv2.putText(vis_left, "SHOW BOARD", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 6. Status Display
        status_text = f"MODE: {'AUTO-SAVE' if capturing else 'PREVIEW'} | Count: {frame_count}"
        color = (0, 255, 0) if capturing else (0, 255, 255)
        
        # Display Left visually on the left, Right visually on the right
        combined = cv2.hconcat([vis_left, vis_right])
        cv2.putText(combined, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Scale down if 1600x800 is too big for your screen
        display_frame = cv2.resize(combined, (WIDTH, HEIGHT // 2))
        cv2.imshow("Stereo Collection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('s'): 
            capturing = not capturing
            last_capture_time = time.time()

        if frame_count >= NUM_FRAMES:
            print("Collection Complete!")
            break

finally:
    picam_right.stop()
    picam_left.stop()
    cv2.destroyAllWindows()
