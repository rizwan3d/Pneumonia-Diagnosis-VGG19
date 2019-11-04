from __future__ import division, print_function
# coding=utf-8
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
import argparse
import imutils
import cv2
import os

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = '../models/model.h5'


img_width, img_height = 128, 128

model = load_model(MODEL_PATH)
model.compile(loss = "binary_crossentropy", 
              optimizer = SGD(lr=0.001, momentum=0.9), 
              metrics=["accuracy"])


def model_predict(img_path, model):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (img_width, img_height))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)  
    # generate prediction
    result = model.predict(image)
    pred = np.argmax(result, axis=1)
    prediction = "UNRECOGNIZABLE"
    if(pred[0] == 0):
        prediction = "Normal"
    else:
        prediction = "Pneumonia"

    return prediction


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        return preds
    return None


# if __name__ == '__main__':
#     # app.run(port=5002, debug=True)

#     # Serve the app with gevent
#     http_server = WSGIServer(('0.0.0.0', 5000), app)
#     http_server.serve_forever()
