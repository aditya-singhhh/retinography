import tensorflow as tf
from flask import Flask, render_template, request
import os
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_class(path):
    img = cv2.imread(path)
    RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    RGBImg = cv2.resize(RGBImg, (224, 224))
    image = np.array(RGBImg) / 255.0

    # Load the model using tf.saved_model.load for inference-only
    model = tf.saved_model.load("64x3-CNN.model")
    infer = model.signatures["serving_default"]
    
    predict = infer(tf.constant([image], dtype=tf.float32))
    probabilities = predict['dense_1'].numpy()[0].tolist()
    diagnosis = "No Diabetic Retinopathy Detected" if np.argmax(probabilities) == 1 else " Diabetic Retinopathy Detected"
    return diagnosis, probabilities

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            diagnosis, probabilities = predict_class(file_path)
            return render_template('predict.html', diagnosis=diagnosis, probabilities=probabilities, user_image=file_path)
    return render_template('index.html', message='Error occurred')

if __name__ == '__main__':
    app.run(debug=True)
