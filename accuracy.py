import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Define the same CNN model used for training
class GestureCNN(nn.Module):
    def __init__(self, num_classes):
        super(GestureCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define transformations (same as used during training)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load test dataset
DATASET_PATH = "HandGestureDataset"
test_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load the trained model
num_classes = len(test_dataset.classes)
model = GestureCNN(num_classes)
model.load_state_dict(torch.load("gesture_model.pth", map_location=torch.device("cpu")))
model.eval()  # Set model to evaluation mode

# Evaluate model accuracy
true_labels, predicted_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted.numpy())

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"✅ Model Accuracy on Test Set: {accuracy * 100:.2f}%")
