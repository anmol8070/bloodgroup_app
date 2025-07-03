from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model('model/blood_group_cnn.h5')

# Replace this with your model’s class label order
class_labels = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # adjust size if needed
    img = img / 255.0  # adjust normalization if needed
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        
        # For now just redirect to login
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        if email == 'admin@example.com' and password == 'admin123':
            return redirect(url_for('hospitalhome'))
        else:
            return render_template('login.html', error="Invalid credentials. Use admin@example.com / admin123")
    
    return render_template('login.html')

@app.route('/hospitalhome', methods=['GET', 'POST'])
def hospitalhome():
    if request.method == 'POST':
        pname = request.form['pname']
        age = request.form['age']
        gender = request.form['gender']
        file = request.files['file']

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = preprocess_image(filepath)
            pred = model.predict(img)

            # Debug prints
            print("Prediction probabilities:", pred)
            class_idx = np.argmax(pred)
            print("Predicted class index:", class_idx)
            blood_group = class_labels[class_idx]
            print("Predicted blood group:", blood_group)

            return render_template('result.html', pname=pname, age=age, gender=gender, result=blood_group)
    
    return render_template('hospitalhome.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
