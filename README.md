# NeroCare_AI

NeroCare_AI is an AI-powered healthcare diagnostic platform designed to **predict stroke risk** using patient medical data and provide **personalized medication recommendations**. By leveraging machine learning, NeroCare_AI helps in early detection and preventive care for stroke-prone individuals.

---

## 🚀 Project Overview

NeroCare_AI analyzes patient demographic, physiological, and lifestyle parameters to determine the likelihood of stroke.  
If stroke risk is detected, the system automatically recommends the most suitable **medications**, **preventive guidelines**, and **lifestyle adjustments**.

The system is built to be fast, accurate, and intuitive, designed for both students and healthcare-based academic projects.

---

## ✨ Features

- 🧠 **Stroke Risk Prediction** — High, Medium, or Low  
- 💊 **Medication Recommendation Engine**  
- 📊 **Parameter-Based Health Evaluation**  
- ⚡ **Instant Predictions Using ML Models**  
- 📱 **Modern and Responsive Web UI**  
- 🔐 **Secure Patient Data Handling**

---

## 🛠️ Tech Stack

### **Frontend**
- HTML5, CSS3  
- Bootstrap / Tailwind CSS  
- JavaScript  
- Jinja2 (Flask Template Engine)

### **Backend**
- Python Flask  
- Scikit-Learn / XGBoost (Prediction Model)  
- NumPy, Pandas  
- Joblib for model loading

### **System Workflow**
1. User enters health parameters  
2. ML model predicts stroke probability  
3. Recommender engine provides medication suggestions  
4. Results are visualized on the UI  

---

## 📁 Directory Structure

```
NeroCare_AI/
│── app.py
│── requirements.txt
│── /models
│   ├── stroke_prediction_model.pkl
│   ├── medications.pkl
│── /templates
│   ├── index.html
│   ├── result.html
│── /static
│   ├── css/
│   ├── js/
│   ├── images/
```

---

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/bkit-solutions/NeuroCare_AI.git
cd NeroCare_AI
```

### 2️⃣ Create a Virtual Environment  
```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate       # Windows
```

### 3️⃣ Install Required Packages  
```bash
pip install -r requirements.txt
```

---

## 🔧 Setup Instructions

- Ensure your ML model files (`.pkl`) are inside `/models`  
- Ensure images/assets are placed in `/static`  
- Update any file paths in `app.py` if needed  

---

## ▶️ How to Run the Project  
Start the Flask server:

```bash
python app.py
```

Your app will be available at:

```
http://127.0.0.1:5000
```

---

## 🧠 How NeroCare_AI Works

1. Patient fills out required medical details  
2. Backend processes the input and feeds into ML model  
3. Stroke risk is predicted as:  
   - **Low**
   - **Medium**
   - **High**
4. If Medium/High → Medication & lifestyle recommendations appear  
5. Full results are shown neatly on the UI  

---

## 🩺 Medication Recommendation Engine

Based on predicted stroke risk, system suggests:

- **Antiplatelets** (e.g., Aspirin)  
- **Statins** for cholesterol management  
- **Antihypertensives** (BP control)  
- **Blood Sugar regimen** for diabetes patients  
- **Diet & lifestyle optimization guidelines**

> ⚠️ *Disclaimer: This system is for educational and project use only.*  

---

## 🔮 Future Enhancements

- Live ECG-based stroke detection  
- Stroke-zone localization using AI on CT/MRI scans  
- Integration with wearable health trackers  
- Smart doctor dashboard with patient analytics  
- Auto-generated PDF reports  

---

## 📜 License  
MIT License  

---

## 👥 Contributors  
Team NeroCare_AI  | BKIT
AI Developers • Data Scientists • Healthcare Enthusiasts  
