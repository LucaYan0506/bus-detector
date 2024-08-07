import os
from ultralytics import YOLO

#get the path of the pre-trained model
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

# Run inference on 'traffic road.jpg' with arguments
model.predict("traffic road.jpg", save=True, imgsz=320, conf=0.5)