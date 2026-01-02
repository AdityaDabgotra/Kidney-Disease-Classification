# 🩺 Kidney Disease Classification using CNN

![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black.svg)
![CNN](https://img.shields.io/badge/Model-CNN-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

![GitHub stars](https://img.shields.io/github/stars/AdityaDabgotra/Kidney-Disease-Classification?style=social)
![GitHub forks](https://img.shields.io/github/forks/AdityaDabgotra/Kidney-Disease-Classification?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/AdityaDabgotra/Kidney-Disease-Classification)

This project implements a **Convolutional Neural Network (CNN)** to classify kidney medical images into different disease categories.  
It also includes a **Flask-based web application** that allows users to upload images and get real-time predictions with confidence scores.

---

## 📌 Project Description

Early detection of kidney-related diseases is critical for effective treatment.  
This project leverages **Deep Learning** to automatically classify kidney images into predefined classes such as **Normal** and **Stone**.

The system includes:
- Model training and evaluation pipeline
- Saved trained CNN model
- Flask backend for inference
- Simple frontend for image upload and prediction

---

## 🗂️ Project Structure
```bash
Kidney-Disease-Classification/
│
├── artifacts/ # Trained models and generated outputs
├── cnnClassifier/ # Core source code
│ ├── components/ # Model and data components
│ ├── config/ # Configuration management
│ ├── entity/ # Configuration entities
│ ├── pipeline/ # Training & evaluation pipelines
│ └── utils/ # Utility functions
│
├── config.yaml # Global configuration file
├── params.yaml # Model parameters
├── app.py # Flask application
├── main.py # Training pipeline entry point
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

## ⚙️ Tech Stack

- **Python 3.8**
- **TensorFlow / Keras**
- **Flask**
- **NumPy & Pandas**
- **OpenCV / PIL**
- **HTML, CSS, JavaScript**

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/AdityaDabgotra/Kidney-Disease-Classification.git
cd Kidney-Disease-Classification
```

### 2️⃣ Create & Activate Conda Environment
```bash
conda create -n cnncls python=3.8 -y
conda activate cnncls
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model (Optional)

### If you want to train the model from scratch:
```bash
python main.py
```
### 5️⃣ Run the Flask Application
```bash
python app.py
```

Open your browser and visit:
```bash
http://127.0.0.1:5000/
```

### 🧪 Prediction Output

The web application provides:

Predicted kidney condition

Confidence percentage

Prediction results

## 📦 Requirements

All required libraries are listed in:

requirements.txt

## 🙌 Author

Aditya Dabgotra
📧 Email: adityadabgotra2004@gmail.com

🔗 GitHub: https://github.com/AdityaDabgotra

## 📄 License

This project is open-source and intended for educational and research purposes.

⭐ If you find this project useful, consider giving it a star!
