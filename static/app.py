# app.py
from flask import Flask, request, render_template
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model from the joblib file
model = joblib.load('model.joblib')

def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize image to match input shape for the model (5x4 for this example)
    resized_image = cv2.resize(gray_image, (5, 4))  # This will create 20 pixels total (5*4)

    # Normalize pixel values to range [0, 1]
    normalized_image = resized_image / 255.0

    # Flatten image for model input (should result in 20 features)
    return normalized_image.flatten().reshape(1, -1)

def clean_image(image):
    # Apply Gaussian Blur to reduce noise and improve feature extraction
    cleaned_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return cleaned_image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']
        
        # Read and process the images
        img1 = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)

        # Clean images before preprocessing
        cleaned_img1 = clean_image(img1)
        cleaned_img2 = clean_image(img2)

        # Preprocess images to match the input shape of the model (should yield 20 features)
        processed_img1 = preprocess_image(cleaned_img1)
        processed_img2 = preprocess_image(cleaned_img2)

        # Predict using the loaded model
        prediction1 = model.predict(processed_img1)[0]
        prediction2 = model.predict(processed_img2)[0]

        # Determine match status and similarity score (dummy logic)
        match = prediction1 == prediction2
        similarity_score = np.random.rand() * 100  # Replace with actual calculation if needed

        return render_template('index.html', match=match, score=similarity_score)

    return render_template('index.html', match=None)

if __name__ == '__main__':
    app.run(debug=True)
