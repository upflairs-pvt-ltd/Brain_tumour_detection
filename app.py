import os
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

def getResult(img):
    # Check if the file exists before attempting to access it
    if os.path.exists(img):
        image = cv2.imread(img)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        image = np.array(image)
        input_img = np.expand_dims(image, axis=0)
        result = model.predict(input_img)
        return result
    else:
        # Handle the case where the file does not exist
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        #To save the file, which are inputed to this system.
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        value = getResult(file_path)
        result = get_className(value) 


        return render_template('import.html',Result=result)
    return None

if __name__ == '__main__':
    app.run(debug=True)
