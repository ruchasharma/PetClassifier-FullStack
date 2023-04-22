import torch
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import joblib 
from flask import Flask
from flask import jsonify, render_template, make_response
import io
import base64
from flask import request

# Import load_model function from FS_Scripy.py file
from FS_Script import load_model
from FS_Script import detect_dogs
#img = cv2.imread('D:/Newfolder/Dogs.jpg')

app = Flask(__name__)
model = load_model()

@app.route('/')
def home():
    #return "<h1>Welcome to Dog Classifier. Use '/classify' route to classify. </h1>"
    return render_template('index.html')

#@app.route("/classify")
#def classify():
    
    #img = cv2.imread('D:/Newfolder/Dogs.jpg')
    # Resize the image to a larger size
    #new_size = (400, 400)
    #img = cv2.resize(img, new_size)

    # Get detections and draw bounding boxes on the image
    #img_with_detections = detect_dogs(model, img)
    # Detect dogs and get predicted boxes
    #img_with_detections, boxes = detect_dogs(model, img, return_boxes=True)

    # Print predicted boxes
    #print(boxes)
    
    # Detect dogs and get predicted boxes
    #img_with_detections, boxes = detect_dogs(model, img, dog_boxes=True)

    # Print predicted boxes
    #print(boxes)

    # Convert the image to JPEG format
    #img_bytes = cv2.imencode('.jpg', img_with_detections)[1].tobytes()
    
    # Create a response with the image data and set content type to JPEG
    #response = make_response(img_bytes)
    #response.headers['Content-Type'] = 'image/jpeg'
    
    # Display the image in the browser
    return response




    
    # Convert the image to JPEG format
    img_bytes = cv2.imencode('.jpg', img_with_detections)[1].tobytes()
    
    # Create a response with the image data and set content type to JPEG
    response = make_response(img_bytes)
    response.headers['Content-Type'] = 'image/jpeg'
    
    # Display the image in the browser
    #return response

@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded image from the request object
    image = request.files['image'].read()
    image = Image.open(io.BytesIO(image))

    # Preprocess the image for the model
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Make the prediction with the model
    img_with_detections, boxes = detect_dogs(model, image, dog_boxes=True)

    # Draw bounding boxes on the image
    # Draw bounding boxes on the image
    for box in boxes:
    # Get box coordinates
        try:
         x1, y1, x2, y2, *_ = box
        except ValueError:
        # If the box does not have 4 coordinates, skip it
         continue
    
        print(x1, y1, x2, y2)


    # Draw box on the image
    img_with_detections = cv2.rectangle(img_with_detections, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)


    # Convert the image to JPEG format
    img_bytes = cv2.imencode('.jpg', img_with_detections)[1].tobytes()

    # Create a response with the image data and set content type to JPEG
    response = make_response(img_bytes)
    response.headers['Content-Type'] = 'image/jpeg'

    # Display the image in the browser
    return response




if __name__ == '__main__':
    app.run()
