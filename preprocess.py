import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def segment_skin(frame):
    lower = np.array([0, 40, 30], dtype="uint8")
    upper = np.array([43, 255, 254], dtype="uint8")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=1)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    return cv2.bitwise_and(frame, frame, mask=skinMask)

def segment_skin_2(image):
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(img, lower, upper)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    skin = cv2.bitwise_and(img, img, mask=skin_mask)
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

def cluster(image):
    red = image[:, :, 0]
    X = red.flatten().reshape(-1, 1)
    y_pred = KMeans(n_clusters=2, random_state=170).fit_predict(X)
    y_pred = np.invert(y_pred.reshape(image.shape[0], image.shape[1]).astype(np.uint8))
    return y_pred

def invert(img):
    m, n = img.shape
    for i in range(m):
        for j in range(n):
            img[i][j] = 0 if img[i][j] == 1 else 1
    return img

def element_wise_multiply(img, mask):
    return img * mask

def preprocess_final(img):
    red = img[:, :, 0]
    X = red.flatten().reshape(-1, 1)
    y_pred = KMeans(n_clusters=2, random_state=10).fit_predict(X)
    y_pred = y_pred.reshape(img.shape[0], img.shape[1])
    mask = invert(y_pred)
    return element_wise_multiply(red, mask)
