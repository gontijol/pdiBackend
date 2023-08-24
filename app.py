from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import requests

app = Flask(__name__)
app.debug = True

def process_image(image_data):
    try:
        # Download the image from the provided URL
        image_response = requests.get(image_data)
        image_bytes = image_response.content
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Rest of your image processing code
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=30, threshold2=70)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            pupil_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(pupil_contour)
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        
        # Encode the processed image to base64
        _, buffer = cv2.imencode('.jpg', image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        image_details = {
            'width': image.shape[1],
            'height': image.shape[0],
            'format': 'jpeg',
            'data': processed_image_base64
        }
        
        return image_details
        
    except Exception as e:
        return None

@app.route('/process_image', methods=['POST'])
def process_image_route():
    try:
        data = request.json
        image_data = data['image_data']
        
        processed_image = process_image(image_data)
        
        if processed_image is not None:
            return jsonify({'processed_image': processed_image})
        else:
            return jsonify({'error': 'Image processing failed'}), 500
        
    except Exception as e:
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083)
