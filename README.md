# Hand2Text: Real-Time ASL Gesture Recognition using MediaPipe and ANN

> 👋 A Final Year Project by Vangala Rishwanth  
> 🔤 Real-time American Sign Language (ASL) fingerspelling to text converter using MediaPipe and Artificial Neural Networks (ANN).

## 🚀 Features

- Real-time hand gesture detection using **MediaPipe**
- Gesture classification using a custom **ANN model**
- Supports **A–Z**, **Space**, **Delete**, and **No Gesture**
- Optional **Text-to-Speech (TTS)** support
- Built using **Flask** (backend) + **JavaScript** (frontend)
- Simple and responsive UI – works in a browser

---

## 🧠 Tech Stack

- Python
- Flask
- MediaPipe
- OpenCV
- TensorFlow / Keras
- NumPy
- scikit-learn
- JavaScript (Webcam + API handling)

---

## 📁 Project Structure
📦 Hand2Text/
┣ 📜 app.py # Flask backend
┣ 📜 asl_sign_recognition_model.h5 # Trained ANN model
┣ 📜 requirements.txt # Dependencies
┣ 📂 static/
┃ ┗ 📂 js/
┃ ┗ 📜 app.js # Frontend JS (camera, API calls)
┣ 📂 templates/
┃ ┗ 📜 index.html # Frontend HTML




---

## 🔧 Setup & Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/VRishwanth/Mediapipe-and-ANNs-for-Real-Time-ASL-Gesture-Recognition.git
cd Mediapipe-and-ANNs-for-Real-Time-ASL-Gesture-Recognition

python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
pip install -r requirements.txt

**Requirements**

flask
mediapipe
opencv-python
numpy
tensorflow
scikit-learn


