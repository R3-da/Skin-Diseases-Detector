from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
import json  

# Pour Le modèle CNN
def decode_prediction(preds, top=2, class_list_path=None):
    if len(preds.shape) != 2 or preds.shape[1] != 3:
        raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples, 3)). '
                     'Found array with shape: ' + str(preds.shape))
    CLASS_INDEX = json.load(open(class_list_path))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results

model = load_model('static/SD_model_5_epochs_3M.h5')

COUNT = 0

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/resultat/', methods=['POST'])
def resultat():    
    # CNN
    if request.form['methode']=="methode1":
        global COUNT
        img = request.files['inference']

        img.save('static/{}.jpg'.format(COUNT))    
        img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

        img_arr = cv2.resize(img_arr, (224,224))
        img_arr = img_arr / 255.0
        img_arr = img_arr.reshape(1, 224,224,3)

        predictions = model.predict(img_arr)

        results = decode_prediction(predictions, 2, 'static/imagenet_class_index_3M.json')
        data = []
        data.append(results)

        COUNT += 1
        return render_template('resultat.html', data=results)

@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))

@app.route('/detecter/')
def detection():
    return render_template('detecter.html')

@app.route('/comprendre/')
def comprendre():
    return render_template('comprendre.html')

@app.route('/apropos/')
def apropos():
    return render_template('apropos.html')

if __name__ == '__main__':
    app.run(debug=True)



