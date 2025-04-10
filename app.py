from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = load_model('model/model.h5')  # make sure this path is correct

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = Image.open(image_path).resize((64, 64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            img = preprocess_image(image_path)
            result = model.predict(img)[0][0]
            prediction = "Crop" if result < 0.5 else "Weed"

    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
