from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from collections import deque, Counter
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

app = Flask(__name__)

# Load the pre-trained ASL model
model = load_model('asl_sign_recognition_model.h5')

# Prepare MediaPipe for real-time hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Define possible classes and encoder
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space", "nothing"]
label_encoder = LabelEncoder()
label_encoder.fit(classes)

# State for buffering predictions
text_box = ""
prediction_queue = deque(maxlen=5)  
accept_threshold =  3      
last_accepted = None



def preprocess_keypoints(keypoints):
    # Reshape landmarks into model input
    return np.array(keypoints).reshape(1, -1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global text_box, prediction_queue, last_accepted
    logging.debug('Predict endpoint called')

    # Ensure frame provided
    if 'frame' not in request.files:
        logging.error('No frame provided in request')
        return jsonify({'error': 'No frame provided'}), 400

    # Decode the uploaded JPEG
    file = request.files['frame']
    img_data = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    logging.debug(f'Received frame shape: {frame.shape}')

    # Save a debug frame image
    cv2.imwrite('debug_frame.jpg', frame)

    # Prepare image for MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    logging.debug('Landmarks present: %s', bool(results.multi_hand_landmarks))

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]
        logging.debug('Extracted %d keypoints', len(keypoints))
        data = preprocess_keypoints(keypoints)

        preds = model.predict(data, verbose=0)[0]
        idx = int(np.argmax(preds))
        label = label_encoder.inverse_transform([idx])[0]
        logging.debug('Model predicted: %s (%.2f)', label, preds[idx])

        prediction_queue.append(label)
        common, count = Counter(prediction_queue).most_common(1)[0]
        if count >= accept_threshold and common != last_accepted and common != 'nothing':
            logging.debug('Accepted label: %s', common)
            if common == 'space':
                text_box += ' '
            elif common == 'del':
                text_box = text_box[:-1]
            else:
                text_box += common
            last_accepted = common
            prediction_queue.clear()
    else:
        logging.debug('No hand detected in frame')
        prediction_queue.clear()
        last_accepted = None

    return jsonify({'text': text_box})


if __name__ == '__main__':
    # Disable reloader to capture logging in single process
    app.run(debug=True, use_reloader=False)

@app.route('/clear', methods=['POST'])
def clear():
    global text_box, prediction_queue, last_accepted
    logging.debug('Clear endpoint called')
    text_box = ""
    prediction_queue.clear()
    last_accepted = None
    return jsonify({'status': 'cleared'})
