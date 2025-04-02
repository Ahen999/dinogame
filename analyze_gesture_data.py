'''
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ensure Matplotlib displays plots correctly
plt.ion()

# Set dataset path
DATASET_PATH = "HandGestureDataset"

# Gesture labels
GESTURE_LABELS = {
    "1_finger": 1,
    "2_fingers": 2,
    "3_fingers": 3,
    "4_fingers": 4,
    "5_fingers": 5
}

# Store gesture counts for visualization
gesture_counts = {label: 0 for label in GESTURE_LABELS.keys()}
image_data = []

# Ensure dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"❌ Dataset folder '{DATASET_PATH}' not found! Please check the path.")
    exit()

print("✅ Dataset found. Checking images...")

# Read dataset images
for gesture in GESTURE_LABELS.keys():
    folder_path = os.path.join(DATASET_PATH, gesture)

    if not os.path.exists(folder_path):
        print(f"⚠️ Skipping {gesture}: Folder not found")
        continue

    image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png"))]

    if len(image_files) == 0:
        print(f"⚠️ No images found for {gesture}")
        continue

    gesture_counts[gesture] = len(image_files)

    for img_file in image_files[:5]:  # Load 5 sample images per gesture
        img_path = os.path.join(folder_path, img_file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image_data.append((gesture, image))

# Stop execution if no images are found
if sum(gesture_counts.values()) == 0:
    print("❌ No images found in any category. Make sure you have captured data.")
    exit()

print("✅ Image analysis complete. Generating visualizations...")

# Plot histogram of detected gestures
plt.figure(figsize=(8, 5))
plt.bar(gesture_counts.keys(), gesture_counts.values(), color="skyblue")
plt.xlabel("Gesture Type")
plt.ylabel("Number of Captured Images")
plt.title("Final Dataset Distribution")
plt.xticks(rotation=20)
plt.show(block=True)  # Force plot to stay open

# Display sample images
plt.figure(figsize=(10, 5))
for i, (gesture, img) in enumerate(image_data[:5]):
    plt.subplot(1, 5, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(gesture)
    plt.axis("off")
plt.tight_layout()
plt.show(block=True)  # Force plot to stay open

print("✅ Visualization complete. Analysis finished successfully!")'
################################################################################
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Dataset path
DATASET_PATH = "HandGestureDataset"

# Gesture labels
GESTURE_LABELS = {
    "1_finger": 0,
    "2_fingers": 1,
}

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"❌ Dataset folder '{DATASET_PATH}' not found! Please check the path.")
    exit()

# Check if model exists
MODEL_PATH = "gesture_model.pth"
if os.path.exists(MODEL_PATH):
    print("✅ Model found! Loading...")
    model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.eval()
    use_model = True
else:
    print("⚠️ No model found! Proceeding without prediction analysis.")
    use_model = False

# Store gesture counts for visualization
gesture_counts = {label: 0 for label in GESTURE_LABELS.keys()}
image_data = []
true_labels = []
predicted_labels = []

# Define image transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Read dataset images
for gesture, label in GESTURE_LABELS.items():
    folder_path = os.path.join(DATASET_PATH, gesture)

    if not os.path.exists(folder_path):
        print(f"⚠️ Skipping {gesture}: Folder not found")
        continue

    image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png"))]

    if len(image_files) == 0:
        print(f"⚠️ No images found for {gesture}")
        continue

    gesture_counts[gesture] = len(image_files)

    for img_file in image_files[:10]:  # Load 10 sample images per gesture
        img_path = os.path.join(folder_path, img_file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image_data.append((gesture, image))
            true_labels.append(label)

            # If model is available, make predictions
            if use_model:
                img_tensor = transform(image).unsqueeze(0)
                prediction = torch.argmax(model(img_tensor)).item()
                predicted_labels.append(prediction)

# Stop execution if no images are found
if sum(gesture_counts.values()) == 0:
    print("❌ No images found in any category. Make sure you have captured data.")
    exit()

print("✅ Image analysis complete. Generating visualizations...")

# Plot histogram of detected gestures
plt.figure(figsize=(8, 5))
plt.bar(gesture_counts.keys(), gesture_counts.values(), color="skyblue")
plt.xlabel("Gesture Type")
plt.ylabel("Number of Captured Images")
plt.title("Final Dataset Distribution")
plt.xticks(rotation=20)
plt.show()

# Display sample images per gesture
plt.figure(figsize=(10, 5))
for i, (gesture, img) in enumerate(image_data[:5]):
    plt.subplot(1, 5, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(gesture)
    plt.axis("off")
plt.tight_layout()
plt.show()

# Generate confusion matrix if model predictions are available
if use_model and len(predicted_labels) > 0:
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=GESTURE_LABELS.keys(), yticklabels=GESTURE_LABELS.keys())
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix of Model Predictions")
    plt.show()

    # Print classification report
    print("\n🔹 **Model Classification Report:**")
    print(classification_report(true_labels, predicted_labels, target_names=GESTURE_LABELS.keys()))

print("✅ Full analysis complete!")
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from train_gesture_model import GestureCNN  # Import the model architecture

# ✅ Load Model Properly
print("✅ Model found! Loading...")
model = GestureCNN()  # Create an instance of the model
model.load_state_dict(torch.load("gesture_model.pth"))  # Load trained weights
model.eval()  # Set model to evaluation mode

# Define dataset path
DATASET_PATH = "HandGestureDataset"

# Define data transformation (same as training)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load dataset
test_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ✅ Perform Gesture Analysis
actual_labels = []
predicted_labels = []

print("\nAnalyzing gesture dataset...")
for images, labels in test_loader:
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        actual_labels.append(labels.item())
        predicted_labels.append(predicted.item())

# ✅ Compute Confusion Matrix
class_names = test_dataset.classes
cm = confusion_matrix(actual_labels, predicted_labels)
report = classification_report(actual_labels, predicted_labels, target_names=class_names)

# ✅ Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Gesture Recognition")
plt.show()

# ✅ Print Classification Report
print("\n=== Classification Report ===")
print(report)

# ✅ Save Analysis Report
with open("gesture_analysis_report.txt", "w") as f:
    f.write("=== Gesture Classification Report ===\n\n")
    f.write(report)

print("\n✅ Gesture analysis completed! Results saved in 'gesture_analysis_report.txt'")
