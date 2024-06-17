# Install the Ultralytics library using pip
!pip install ultralytics

# Install the Roboflow library using pip
!pip install roboflow

# Import the necessary modules
from roboflow import Roboflow
from ultralytics import YOLO

# Import other required libraries
import os
import glob

# Define the Roboflow API key
api_key = "YOUR_API_KEY"

# Initialize Roboflow and access the project
rf = Roboflow(api_key)
workspace_name = "YOUR_WORKSPACE_NAME"
project_name = "YOUR_PROJECT_NAME"
project = rf.workspace(workspace_name).project(project_name)

# Download the dataset associated with version 1 of the project using YOLOv8 format
version = 1
form = "yolov8"
dataset = project.version(version).download(form)

# Initialize the YOLO model by loading the pre-trained weights from 'yolov8s.pt'
model = YOLO('yolov8s.pt')

# Train the model with the specified parameters
results = model.train(
    data='PATH_TO_YAML_FILE',  # Path to the training data YAML file
    epochs=200,  # Number of training epochs
    batch=32,  # Batch size for training
    imgsz=640,  # Input image size
    seed=32,  # Random seed for reproducibility
    optimizer='NAdam',  # Optimizer algorithm
    weight_decay=1e-4,  # Weight decay for regularization
    momentum=0.937,  # Initial momentum for the optimizer
    cos_lr=True,  # Use cosine learning rate scheduling
    lr0=0.01,  # Initial learning rate
    lrf=1e-5,  # Final learning rate
    warmup_epochs=10,  # Number of warmup epochs
    warmup_momentum=0.5,  # Momentum during warm-up epochs
    close_mosaic=20,  # Parameter for close mosaic augmentation
    label_smoothing=0.2,  # Label smoothing parameter for regularization
    dropout=0.5,  # Dropout rate to prevent overfitting
    verbose=True  # Print verbose training information
)

# Save the trained model to a file
model.save('PATH_TO_SAVED_MODEL')
