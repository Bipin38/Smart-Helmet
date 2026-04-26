import cv2
import numpy as np
import os
import json

# === CONFIGURATION ===
CHESSBOARD_SIZE = (8, 6)     # Inner corners (columns, rows)
SQUARE_SIZE = 25.0           # millimeters per chessboard square
NUM_IMAGES = 40
EXT = ".jpg"
SAVE_FILE = "stereo_calibration_0305_v6.json"

CAM_FOLDERS = {
    "cam1": "cam0_calib_0305",
    "cam0": "cam1_calib_0305"
}


# === HELPER: FIND CHESSBOARD CORNERS ===
def find_corners(folder_path, num_images, chessboard_size, square_size, file_prefix="img_"):
    """Find chessboard corners for calibration images."""
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # <-- scale to real-world mm

    objpoints = []
    imgpoints = []
    valid_indices = []

    for i in range(1, num_images + 1):
        filename = os.path.join(folder_path, f"{file_prefix}{i:02d}{EXT}")
        img = cv2.imread(filename)
        if img is None:
            print(f"❌ Could not read {filename}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            # Refine corner accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners_subpix)
            valid_indices.append(i)

            cv2.drawChessboardCorners(img, chessboard_size, corners_subpix, ret)
            cv2.imshow("Corners", img)
            cv2.waitKey(50)
        else:
            print(f"⚠️ Chessboard not found in {filename}")

    cv2.destroyAllWindows()
    return objpoints, imgpoints, valid_indices, gray.shape[::-1]  # resolution (width, height)


# === FIND CORNERS FOR BOTH CAMERAS ===
objpoints_left, imgpoints_left, valid_left, res_left = find_corners(
    CAM_FOLDERS["cam1"], NUM_IMAGES, CHESSBOARD_SIZE, SQUARE_SIZE
)
objpoints_right, imgpoints_right, valid_right, res_right = find_corners(
    CAM_FOLDERS["cam0"], NUM_IMAGES, CHESSBOARD_SIZE, SQUARE_SIZE
)

# === FILTER VALID PAIRS ===
valid_indices = list(set(valid_left).intersection(set(valid_right)))
print(f"✅ Using {len(valid_indices)} valid image pairs for stereo calibration.")

objpoints_stereo = []
imgpoints_left_stereo = []
imgpoints_right_stereo = []

for idx in valid_indices:
    i_left = valid_left.index(idx)
    i_right = valid_right.index(idx)
    objpoints_stereo.append(objpoints_left[i_left])
    imgpoints_left_stereo.append(imgpoints_left[i_left])
    imgpoints_right_stereo.append(imgpoints_right[i_right])

# === CALIBRATE INDIVIDUAL CAMERAS ===
print("📷 Calibrating left and right cameras individually...")
retL, mtxL, distL, _, _ = cv2.calibrateCamera(
    objpoints_stereo, imgpoints_left_stereo, res_left, None, None
)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(
    objpoints_stereo, imgpoints_right_stereo, res_right, None, None
)

# === STEREO CALIBRATION ===
print("🎯 Performing stereo calibration...")
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints_stereo,
    imgpoints_left_stereo,
    imgpoints_right_stereo,
    mtxL, distL,
    mtxR, distR,
    res_left,
    criteria=criteria_stereo,
    flags=flags
)

# === STEREO RECTIFICATION ===
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR,
    res_left, R, T, alpha=0
)

# === COMPUTE BASELINE ===
baseline = np.linalg.norm(T)
print(f"📏 Baseline (distance between cameras): {baseline:.2f} mm")

# === SAVE TO JSON ===
calibration_data = {
    "square_size_mm": SQUARE_SIZE,
    "cam0": {
        "resolution": list(res_left),
        "camera_matrix": mtxL.tolist(),
        "dist_coeffs": distL.tolist()
    },
    "cam1": {
        "resolution": list(res_right),
        "camera_matrix": mtxR.tolist(),
        "dist_coeffs": distR.tolist()
    },
    "R": R.tolist(),
    "T": T.tolist(),
    "R1": R1.tolist(),
    "R2": R2.tolist(),
    "P1": P1.tolist(),
    "P2": P2.tolist(),
    "Q": Q.tolist(),
    "baseline_mm": float(baseline)
}

with open(SAVE_FILE, "w") as f:
    json.dump(calibration_data, f, indent=4)

print(f"🎉 Stereo calibration completed and saved to {SAVE_FILE}!")

