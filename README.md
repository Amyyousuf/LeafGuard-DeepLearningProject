🌿 LeafGuard – Intelligent Plant Disease Detection System

A Deep Learning + Streamlit application for detecting plant diseases from leaf images.

🚀 Overview

LeafGuard is an AI-powered plant disease detection system that uses a Custom Convolutional Neural Network (CNN) trained on the PlantVillage dataset. The app allows users to upload a leaf image and instantly get:

✔ Disease prediction

✔ Confidence score

✔ Healthy/Diseased status

✔ AI-generated treatment/cure suggestions

✔ Drift monitoring charts to track prediction trends

This project aims to assist farmers, students, and researchers in early disease detection and better crop management.

🧠 Features
🌱 Deep Learning Prediction

Custom CNN model built using TensorFlow/Keras

Classifies 38+ diseases and healthy conditions

Preprocessing pipeline ensures consistent image quality

🤖 AI-Generated Cure Suggestions

Automatically generates short treatment steps

Powered by OpenAI API 

📊 Drift Monitoring Dashboard

Tracks model predictions over time

Includes:

📍 Class-frequency bar chart

📍 Disease distribution pie chart

📍 Daily prediction trend line

🖥 Modern Streamlit Interface

Clean UI with custom CSS

Sidebar navigation

Image preview + results panel

Responsive layout

📂 Project Structure
LeafGuard/
│── app.py                     # Main Streamlit application
│── style.css                  # Custom UI styling
│── Project_DL.ipynb           # Model training and evaluation notebook
│── models/
│     └── best_custom_cnn_model.keras   # Trained CNN model
│── drift_data/
│     ├── train_distribution.json
│     └── drift_history.json
│── requirements.txt           # Dependencies 
│── README.md
│── LICENSE

🛠 Technologies Used

Python

TensorFlow / Keras

NumPy

Streamlit

Matplotlib & Seaborn

PIL

OpenAI GPT-4o 

🖼️ How It Works

User uploads a leaf image

Image is resized and preprocessed

CNN model predicts the class

Probability scores are computed

Application displays:

Class name

Confidence

Health status

Cure/solution

Prediction is logged for drift tracking

Dashboard displays data trends

▶️ How to Run the App Locally

Make sure Python 3.8+ is installed.

1. Install dependencies:
pip install -r requirements.txt

2. Run the app:
streamlit run app.py

3. Open the URL shown in terminal (usually http://localhost:8501/)
🔐 License

This project is protected under All Rights Reserved.
No part of this code may be used, copied, or modified without written permission.

👩‍💻 Authors

Iqra Nawaz, Sameen Fatima & Laiba Nadeem

🙌 Acknowledgements

PlantVillage Dataset

TensorFlow documentation

Streamlit community
