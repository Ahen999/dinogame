import warnings
warnings.filterwarnings('ignore')
import cv2, math, pyautogui
import numpy as np
from processing_pipeline import process
from game_control import get_config_data

# Since there's only one camera, we set this to 0.
camera_to_use = 0  
camera = None

def connectToCamera():
    global camera
    print("Connecting to camera", camera_to_use)
    # Using CAP_DSHOW for Windows
    camera = cv2.VideoCapture(camera_to_use, cv2.CAP_DSHOW)
    if camera.isOpened():
        print("Camera connected")
    else:
        print("Failed to connect to camera", camera_to_use)

# Create window and trackbars
cv2.namedWindow('image')
cv2.createTrackbar('Lower Threshold', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('Upper Threshold', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('Game On', 'image', 0, 1, lambda x: None)

config = get_config_data()
connectToCamera()

while True:
    # Safely check if window exists; if not, exit the loop.
    try:
        if cv2.getWindowProperty('image', 0) < 0:
            print("Window closed. Exiting.")
            break
    except cv2.error:
        break

    if camera is not None and camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            print("Failed to read frame from camera.")
            continue

        # Read trackbar positions
        thresh_lower = cv2.getTrackbarPos('Lower Threshold', 'image')
        thresh_upper = cv2.getTrackbarPos('Upper Threshold', 'image')
        game_on = cv2.getTrackbarPos('Game On', 'image')

        try:
            # Process the frame to get ROI and count defects (fingers-1)
            roi_1, drawing, thresh_img, crop_img, count_defects = process(frame, thresh_lower, thresh_upper)
            fingers = count_defects + 1
            out_frame = frame.copy()
            cv2.putText(out_frame, f"Fingers: {fingers}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # If game mode is enabled and a valid key is mapped, send the key press.
            if game_on == 1 and config.get(str(fingers), "None") != "None":
                pyautogui.press(config[str(fingers)])
        except Exception as e:
            print("Error in processing:", e)
            out_frame = frame.copy()
    else:
        out_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    cv2.imshow('image', out_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc key to exit
        break

if camera is not None:
    camera.release()
cv2.destroyAllWindows()
