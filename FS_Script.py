#!pip install torch 
import torch
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    #model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

    return model

def load_image(image_path):
    img = cv2.imread('D:/Newfolder/Dogs.jpg')
    return img

import cv2

def detect_dogs(model, img, dog_boxes=False):
    # Run inference on the image
    results = model(img)
    
    # Extract predicted bounding boxes and class names
    pred_boxes = results.xyxy[0].cpu().numpy()
    pred_classes = [model.names[int(x[-1])] for x in results.xyxy[0]]
    
    # Draw bounding boxes around the detected dogs
    dog_boxes = []
    for box, cls in zip(pred_boxes, pred_classes):
        if cls == 'dog':
            x1, y1, x2, y2 = box[:4]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            dog_boxes.append(box)
    
    return img, dog_boxes





   

if __name__ == "__main__":
    # Load the image and YOLOv5 model
    img = load_image('D:/Newfolder/Dogs.jpg')
    model = load_model()

    # Detect dogs in the image and draw bounding boxes
    img, dog_boxes = detect_dogs(model, img)

    # Display the image with bounding boxes
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()








