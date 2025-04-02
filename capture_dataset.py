'''
import cv2
import os

# Define gesture labels
GESTURE_LABELS = {
    1: "1_finger",
    2: "2_fingers",
    3: "3_fingers",
    4: "4_fingers",
    5: "5_fingers"
}

# Create dataset directories
dataset_path = "HandGestureDataset"
if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

for label in GESTURE_LABELS.values():
    folder_path = os.path.join(dataset_path, label)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

# Start capturing
cap = cv2.VideoCapture(0)
count = 0
current_gesture = 1  # Change this manually for different gestures

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display instructions
    cv2.putText(frame, f"Showing Gesture: {GESTURE_LABELS[current_gesture]}", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Dataset Capture", frame)

    key = cv2.waitKey(1)
    
    if key == ord("s"):  # Press 's' to save the frame
        file_path = os.path.join(dataset_path, GESTURE_LABELS[current_gesture], f"img_{count}.jpg")
        cv2.imwrite(file_path, frame)
        print(f"Saved: {file_path}")
        count += 1

    elif key == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
'''
import cv2
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Define gesture labels
GESTURE_LABELS = {
    1: "1_finger",
    2: "2_fingers",
}

# Create dataset directories
dataset_path = "HandGestureDataset"
if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

for label in GESTURE_LABELS.values():
    folder_path = os.path.join(dataset_path, label)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

# Initialize webcam
cap = cv2.VideoCapture(0)
count = 0
current_gesture = 1  # Change manually for different gestures

# Gesture image count dictionary
image_counts = defaultdict(int)

# Function to update histogram
def update_histogram():
    plt.clf()
    plt.bar(image_counts.keys(), image_counts.values(), color="skyblue")
    plt.xlabel("Gesture Type")
    plt.ylabel("Number of Captured Images")
    plt.title("Live Gesture Capture Stats")
    plt.xticks(rotation=20)
    plt.pause(0.1)  # Refresh plot

# Open Matplotlib interactive window
plt.ion()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display instructions
    cv2.putText(frame, f"Gesture: {GESTURE_LABELS[current_gesture]} | Press 's' to Save", 
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Dataset Capture", frame)

    key = cv2.waitKey(1)

    if key == ord("s"):  # Save frame when 's' is pressed
        file_path = os.path.join(dataset_path, GESTURE_LABELS[current_gesture], f"img_{count}.jpg")
        cv2.imwrite(file_path, frame)
        image_counts[GESTURE_LABELS[current_gesture]] += 1  # Update count
        print(f"Saved: {file_path}")
        count += 1
        update_histogram()  # Refresh the histogram

    elif key == ord("q"):  # Quit when 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()  # Show final histogram
