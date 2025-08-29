from flask import Flask, request, render_template
import cv2
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('model.joblib')
def clean_image(image):
    
    cleaned_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    return cleaned_image
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    resized_image = cv2.resize(gray_image, (5, 4))  
    normalized_image = resized_image / 255.0

    
    return normalized_image.flatten().reshape(1, -1)
def compare_images(image1, image2):
    cleaned_img1 = clean_image(image1)
    cleaned_img2 = clean_image(image2)
    
    processed_img1 = preprocess_image(cleaned_img1)
    processed_img2 = preprocess_image(cleaned_img2)
    
    prediction1 = model.predict(processed_img1)[0]
    prediction2 = model.predict(processed_img2)[0]
    
    match = prediction1 == prediction2
    
    return match, prediction1, prediction2
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file1 = request.files['file1']
        file2 = request.files['file2']
        

        img1 = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)

        match, prediction1, prediction2 = compare_images(img1, img2)

        similarity_score = np.random.rand() * 100 
        return render_template('index.html', match=match, score=similarity_score)

    return render_template('index.html', match=None)

if __name__ == '__main__':
    app.run(debug=True)


