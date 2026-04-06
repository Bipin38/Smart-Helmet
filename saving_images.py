import cv2
import numpy as np
import json
import os
import time
import threading
import logging
import pygame  # NEW: Import pygame for audio
from picamera2 import Picamera2
from gpiozero import Button

# --- 0. Logging & Audio Setup ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"logs/session_{time.strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# NEW: Initialize pygame mixer and define blocking playback function
pygame.mixer.init()

def play_audio_blocking(filename):
    """Plays an audio file and blocks execution until it finishes."""
    if os.path.exists(filename):
        try:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            # This while loop is what forces the code to wait
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            logger.error(f"Audio playback error for {filename}: {e}")
    else:
        logger.warning(f"Audio file not found: {filename}")


# --- 1. Threaded Thermal Camera ---
class ThreadedThermalCamera:
    def __init__(self, src=16):
        self.capture = cv2.VideoCapture(src, cv2.CAP_V4L2)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', '1', '6', ' '))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 80)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 60)
        self.capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock() 

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                if status:
                    with self.lock: self.frame = frame
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def stop(self):
        self.stopped = True
        self.capture.release()

# --- 2. Calibration & Globals ---
CALIB_FILE = "stereo_calibration_0303_v2.json"
try:
    with open(CALIB_FILE, "r") as f:
        calib = json.load(f)
except Exception as e:
    logger.critical(f"Config error: {e}"); exit()

R1, R2, P1, P2, Q = [np.array(calib[k]) for k in ["R1", "R2", "P1", "P2", "Q"]]
mtx0, dist0 = np.array(calib["cam0"]["camera_matrix"]), np.array(calib["cam0"]["dist_coeffs"])
mtx1, dist1 = np.array(calib["cam1"]["camera_matrix"]), np.array(calib["cam1"]["dist_coeffs"])

W, H = 800, 800
DASH_W, DASH_H = 320, 320
ROOT_IMG_DIR = "Dataset"
os.makedirs(ROOT_IMG_DIR, exist_ok=True)

def get_next_scene_num():
    existing = [d for d in os.listdir(ROOT_IMG_DIR) if d.startswith("Scene_")]
    nums = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    return max(nums) + 1 if nums else 0

scene_num = get_next_scene_num()
trigger_capture_flag = False
running_flag = True
clicked_point, measure_text = None, ""
global_depth_map, global_raw_warped = None, None

# --- 3. Hardware Button ---
def request_capture():
    global trigger_capture_flag
    trigger_capture_flag = True

def request_shutdown():
    global running_flag
    running_flag = False

try:
    btn = Button(26, bounce_time=0.1, hold_time=3.0)
    btn.when_pressed = request_capture
    btn.when_held = request_shutdown
except Exception as e:
    logger.error(f"GPIO Error: {e}")

# --- 4. Mouse Callback ---
def on_mouse_click(event, x, y, flags, param):
    global clicked_point, measure_text, global_depth_map, global_raw_warped
    if event == cv2.EVENT_LBUTTONDOWN and x < DASH_W and y < DASH_H:
        orig_x, orig_y = int(x * (W / DASH_W)), int(y * (H / DASH_H))
        clicked_point = (orig_x, orig_y)
        d_str, t_str = "N/A", "N/A"
        if global_depth_map is not None:
            disp = global_depth_map[orig_y, orig_x]
            if disp > 0:
                h = Q @ np.array([orig_x, orig_y, disp, 1.0])
                d_str = f"{(h[2] / h[3]) / 1000.0:.2f}m"
        if global_raw_warped is not None:
            t_str = f"{(global_raw_warped[orig_y, orig_x] / 100.0) - 273.15:.1f}C"
        measure_text = f"{d_str} | {t_str}"

# --- 5. Initialization ---
picam0, picam1 = Picamera2(1), Picamera2(0)
for p in [picam0, picam1]:
    p.configure(p.create_preview_configuration(main={"size": (W, H)}))
    p.start()

try:
    thermal_cam = ThreadedThermalCamera(3).start()
except:
    thermal_cam = ThreadedThermalCamera(16).start()

time.sleep(1.0)

# NEW: Play audio when cameras are successfully started
play_audio_blocking("camera_started.mp3")

matcher_left = cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=11)
matcher_right = cv2.ximgproc.createRightMatcher(matcher_left)
wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher_left)
wls.setLambda(8000.0); wls.setSigmaColor(2.5)

mapL1, mapL2 = cv2.initUndistortRectifyMap(mtx0, dist0, R1, P1, (W, H), cv2.CV_32FC1)
mapR1, mapR2 = cv2.initUndistortRectifyMap(mtx1, dist1, R2, P2, (W, H), cv2.CV_32FC1)
grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
grid_x, grid_y = grid_x.astype(np.float32), grid_y.astype(np.float32)

cv2.namedWindow("Glasses Dashboard")
cv2.setMouseCallback("Glasses Dashboard", on_mouse_click)
cv2.namedWindow("Fine Tune")
def nothing(x): pass
cv2.createTrackbar("ROI X", "Fine Tune", 143, W, nothing)
cv2.createTrackbar("ROI Y", "Fine Tune", 18, H, nothing)
cv2.createTrackbar("ROI W", "Fine Tune", 473, W, nothing)
cv2.createTrackbar("ROI H", "Fine Tune", 583, H, nothing)
cv2.createTrackbar("Parallax", "Fine Tune", 100, 200, nothing)
cv2.createTrackbar("Hot Thresh", "Fine Tune", 190, 255, nothing)
cv2.createTrackbar("Cold Thresh", "Fine Tune", 80, 255, nothing)

# --- 6. Main Loop ---
try:
    while running_flag:
        rectL = cv2.remap(cv2.cvtColor(picam0.capture_array()[:,:,:3], cv2.COLOR_RGB2BGR), mapL1, mapL2, cv2.INTER_LINEAR)
        rectR = cv2.remap(cv2.cvtColor(picam1.capture_array()[:,:,:3], cv2.COLOR_RGB2BGR), mapR1, mapR2, cv2.INTER_LINEAR)
        
        grayL, grayR = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
        dispL = matcher_left.compute(grayL, grayR).astype(np.float32) / 16.0
        dispR = matcher_right.compute(grayR, grayL).astype(np.float32) / 16.0
        depth_map = wls.filter(dispL, grayL, None, dispR)
        global_depth_map = depth_map

        ret, th_raw = thermal_cam.read()
        thermal_warped = np.zeros((H, W, 3), dtype=np.uint8)
        th_vis_full = np.zeros((H, W, 3), dtype=np.uint8)
        detected_objects = [] 

        if ret:
            th_f = th_raw.astype(np.float32)
            th_norm = cv2.normalize(np.clip(th_f, np.percentile(th_f, 2), np.percentile(th_f, 98)), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            th_vis_full = cv2.rotate(cv2.resize(cv2.applyColorMap(th_norm, cv2.COLORMAP_INFERNO), (W, H)), cv2.ROTATE_180)
            th_vis_gray = cv2.rotate(cv2.resize(th_norm, (W, H)), cv2.ROTATE_180)
            
            rx, ry = cv2.getTrackbarPos("ROI X", "Fine Tune"), cv2.getTrackbarPos("ROI Y", "Fine Tune")
            rw, rh = cv2.getTrackbarPos("ROI W", "Fine Tune"), cv2.getTrackbarPos("ROI H", "Fine Tune")
            px = (cv2.getTrackbarPos("Parallax", "Fine Tune") - 100) * 5.0
            
            crop_vis = th_vis_full[ry:ry+rh, rx:rx+rw]
            crop_gray = th_vis_gray[ry:ry+rh, rx:rx+rw]
            crop_raw = cv2.rotate(cv2.resize(th_f, (W, H)), cv2.ROTATE_180)[ry:ry+rh, rx:rx+rw]

            if crop_vis.size > 0:
                t_f, g_f, r_f = cv2.resize(crop_vis, (W, H)), cv2.resize(crop_gray, (W, H)), cv2.resize(crop_raw, (W, H))
                shift_x = grid_x + (cv2.boxFilter(depth_map, -1, (5,5)) * (px / 1000.0))
                thermal_warped = cv2.remap(t_f, shift_x, grid_y, cv2.INTER_LINEAR)
                g_warped = cv2.remap(g_f, shift_x, grid_y, cv2.INTER_LINEAR)
                global_raw_warped = cv2.remap(r_f, shift_x, grid_y, cv2.INTER_NEAREST)

                for (name, thr, meth, col) in [("HOT", cv2.getTrackbarPos("Hot Thresh", "Fine Tune"), cv2.THRESH_BINARY, (0,0,255)), 
                                               ("COLD", cv2.getTrackbarPos("Cold Thresh", "Fine Tune"), cv2.THRESH_BINARY_INV, (255,0,0))]:
                    _, b_mask = cv2.threshold(g_warped, thr, 255, meth)
                    cnts, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in cnts:
                        if cv2.contourArea(c) < 50: continue
                        x, y, w, h = cv2.boundingRect(c)
                        raw_v = np.max(global_raw_warped[y:y+h, x:x+w]) if name=="HOT" else np.min(global_raw_warped[y:y+h, x:x+w])
                        temp_c = (raw_v / 100.0) - 273.15
                        if (name == "HOT" and temp_c < 35.0) or (name == "COLD" and temp_c > 10.0): continue
                        
                        avg_d = np.mean(depth_map[y:y+h, x:x+w][depth_map[y:y+h, x:x+w]>0.1]) or 0
                        homog = Q @ np.array([x+w/2, y+h/2, avg_d, 1.0])
                        detected_objects.append(((x,y,w,h), f"{name} {(homog[2]/homog[3])/1000.0:.1f}m | {temp_c:.1f}C", col))

        rgb_viz = rectL.copy()
        for (rect, label, col) in detected_objects:
            cv2.rectangle(rgb_viz, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), col, 2)
            cv2.putText(rgb_viz, label, (rect[0], rect[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        if clicked_point:
            cv2.circle(rgb_viz, clicked_point, 5, (0,255,0), -1)
            cv2.putText(rgb_viz, measure_text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Dashboard Prep
        gray_d = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        color_d = cv2.applyColorMap(gray_d, cv2.COLORMAP_JET)
        t1, t2, t3 = cv2.resize(rgb_viz, (DASH_W, DASH_H)), cv2.resize(rectR, (DASH_W, DASH_H)), cv2.resize(cv2.cvtColor(gray_d, cv2.COLOR_GRAY2BGR), (DASH_W, DASH_H))
        t4, t5, t6 = cv2.resize(color_d, (DASH_W, DASH_H)), cv2.resize(thermal_warped, (DASH_W, DASH_H)), cv2.resize(th_vis_full, (DASH_W, DASH_H))
        cv2.imshow("Glasses Dashboard", np.vstack((np.hstack((t1, t2, t3)), np.hstack((t4, t5, t6)))))

        # --- TRIGGER CAPTURE LOGIC (SPECIFIC FOLDERS) ---
        if trigger_capture_flag:
            # NEW: Play audio to acknowledge button press *before* saving code runs
            play_audio_blocking("captured.mp3")
            
            base = os.path.join(ROOT_IMG_DIR, f"Scene_{scene_num}")
            subfolders = ["left_cam", "right_cam", "depth", "thermal", "annotated"]
            for s in subfolders: os.makedirs(os.path.join(base, s), exist_ok=True)
            
            cv2.imwrite(os.path.join(base, "left_cam/left.png"), rectL)
            cv2.imwrite(os.path.join(base, "right_cam/right.png"), rectR)
            cv2.imwrite(os.path.join(base, "depth/depth.png"), gray_d)
            np.save(os.path.join(base, "depth/depth.npy"), depth_map)
            cv2.imwrite(os.path.join(base, "thermal/thermal.png"), thermal_warped)
            if global_raw_warped is not None:
                np.save(os.path.join(base, "thermal/thermal.npy"), (global_raw_warped / 100.0) - 273.15)
            cv2.imwrite(os.path.join(base, "annotated/annotated.png"), rgb_viz)
            
            with open(os.path.join(base, "metadata.json"), "w") as f:
                json.dump({"Q": Q.tolist(), "scene": scene_num, "timestamp": time.time()}, f, indent=4)
            
            logger.info(f"Captured Scene {scene_num}"); scene_num += 1
            trigger_capture_flag = False

            # NEW: Play audio once the I/O saving operations are complete
            play_audio_blocking("images_saved.mp3")

        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    play_audio_blocking("script_stopped.mp3")
    picam0.stop(); picam1.stop(); thermal_cam.stop(); cv2.destroyAllWindows()
    pygame.mixer.quit() # NEW: Clean up the audio mixer