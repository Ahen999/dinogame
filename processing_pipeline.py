'''
import warnings
warnings.filterwarnings('ignore')
import cv2, math
import numpy as np

def process(image, thresh_lower, thresh_upper):
    # Draw ROI rectangle
    cv2.rectangle(image, (60, 60), (300, 300), (0, 255, 0), 4)
    roi = image[70:300, 70:300]
    crop_img = roi.copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Convert ROI to LUV color space for segmentation
    luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)
    lower_val = np.array([0, 150, 50])
    upper_val = np.array([195, 255, 255])
    mask = cv2.inRange(luv, lower_val, upper_val)
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)
    # Apply Otsu's thresholding with additional thresholds from trackbars
    ret, thresh_img = cv2.threshold(blurred, thresh_lower, thresh_upper, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours (compatible with OpenCV 4.x)
    contours, hierarchy = cv2.findContours(thresh_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        # If no contours are found, return default outputs
        return roi.copy(), np.zeros_like(roi), thresh_img, crop_img, 0
    cnt = max(contours, key=lambda c: cv2.contourArea(c))
    x, y, w, h = cv2.boundingRect(cnt)
    roi_1 = roi.copy()
    cv2.rectangle(roi_1, (x, y), (x + w, y + h), (0, 0, 255), 1)
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(roi.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 255, 255), 0)
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return roi_1, drawing, thresh_img, crop_img, 0
    defects = cv2.convexityDefects(cnt, hull_indices)
    cv2.drawContours(thresh_img, contours, -1, (0, 255, 0), 3)
    count_defects = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        if 2 * b * c == 0:
            angle = 0
        else:
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * (180 / math.pi)
        cv2.circle(crop_img, far, 4, [0, 0, 255], -1)
        if angle <= 90:
            count_defects += 1
        cv2.line(crop_img, start, end, [0, 255, 0], 3)
    return roi_1, drawing, thresh_img, crop_img, count_defects
'''
import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import math

def process(image, thresh_lower=60, thresh_upper=200):
    """Processes an image to detect hand contours and count fingers."""
    
    # Define ROI (Region of Interest)
    cv2.rectangle(image, (60, 60), (300, 300), (0, 255, 0), 4)
    roi = image[70:300, 70:300]
    crop_img = roi.copy()

    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)

    # Apply thresholding for segmentation
    _, thresh_img = cv2.threshold(blurred, thresh_lower, thresh_upper, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return roi.copy(), np.zeros_like(roi), thresh_img, crop_img, 0  # Ensures 5 return values

    # Get largest contour
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt, returnPoints=False)

    if hull is None or len(hull) < 3:
        return roi.copy(), np.zeros_like(roi), thresh_img, crop_img, 0  # Ensures 5 return values

    # Compute convexity defects
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start, end, far = tuple(cnt[s][0]), tuple(cnt[e][0]), tuple(cnt[f][0])

            # Compute angles
            a = np.linalg.norm(np.array(end) - np.array(start))
            b = np.linalg.norm(np.array(far) - np.array(start))
            c = np.linalg.norm(np.array(end) - np.array(far))

            if 2 * b * c == 0:
                angle = 0
            else:
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * (180 / math.pi)

            if angle <= 90:  
                count_defects += 1

    drawing = np.zeros(roi.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 1)

    return roi.copy(), drawing, thresh_img, crop_img, count_defects + 1  # Ensures 5 return values
