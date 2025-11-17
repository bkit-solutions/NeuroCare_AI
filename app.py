import os
import io
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, redirect, url_for
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model paths
CNN_MODEL_PATH = "models/brain_stroke_model.h5"
RECOMMENDATION_MODEL_PATH = "models/CLAUDE_RECOMMENDATOIN_MODEL"

# Input and output columns
INPUT_COLUMNS = [
    'age', 'gender', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
    'smoking_status', 'medication_use', 'symptoms_duration'
]

OUTPUT_COLUMNS = [
    'stroke_type', 'stroke_stage', 'medication', 
    'recommended_duration', 'recommended_doctor'
]

# Medication plan mapping
def get_comprehensive_medication_plan(stroke_type, severity):
    medication_mappings = {
        "Ischemic": {
            "primary": "Aspirin",
            "secondary": ["Clopidogrel", "Statins"],
            "additional": ["Blood Pressure Medication"],
            "monitoring": {"Mild": "Weekly", "Moderate": "Every 3-5 Days", "Severe": "Daily"}
        },
        "Hemorrhagic": {
            "primary": "Blood Pressure Medication",
            "secondary": ["Antihypertensive Drugs"],
            "additional": ["Calcium Channel Blockers"],
            "monitoring": {"Mild": "Bi-Weekly", "Moderate": "Weekly", "Severe": "Daily"}
        },
        "TIA": {
            "primary": "Antiplatelet Agents",
            "secondary": ["Aspirin", "Clopidogrel"],
            "additional": ["Statins"],
            "monitoring": {"Mild": "Monthly", "Moderate": "Bi-Weekly", "Severe": "Weekly"}
        }
    }
    
    details = medication_mappings.get(stroke_type, {})
    return {
        "primary_medication": details.get("primary", ""),
        "secondary_medications": details.get("secondary", []),
        "additional_medications": details.get("additional", []),
        "monitoring_frequency": details.get("monitoring", {}).get(severity, "As Needed"),
        "intensive_care_required": severity == "Severe"
    }

# Model loading
def load_models():
    try:
        cnn_model = load_model(CNN_MODEL_PATH)
        logger.info("✅ CNN Stroke Detection Model loaded successfully!")
        
        with open(RECOMMENDATION_MODEL_PATH, 'rb') as f:
            model_package = joblib.load(f)
        
        recommendation_model = model_package['model']
        logger.info("✅ Recommendation Model loaded successfully!")
        
        return (
            cnn_model,
            model_package['model'],  # Pipeline
            model_package['label_encoders']  # Direct access to encoders
        )
    
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return None, None, None

CNN_MODEL, RECOMMENDATION_MODEL, LABEL_ENCODERS = load_models()

# Image preprocessing
def preprocess_image(image):
    image = image.convert("L").resize((128, 128))
    return np.expand_dims(img_to_array(image) / 255.0, axis=0)

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/patient')
def patient():
    return render_template('patient.html')

@app.route('/analysis')
def analysis():
    return render_template('index.html')  # Your existing analysis page

# @app.route('/landing')
# def landing():
#     return render_template('landing.html')

@app.route('/predict_stroke', methods=['POST'])
def predict_stroke():
    try:
        if 'image' not in request.files:
            return "No image uploaded", 400
        
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
        processed_img = preprocess_image(img)
        
        prediction = CNN_MODEL.predict(processed_img)
        stroke_detected = prediction[0][0] > 0.5
        
        return render_template('index.html',
            stroke_detected=stroke_detected,
            message='Stroke Detected - Immediate Attention Required!' if stroke_detected 
                   else 'Normal Brain Activity - No Further Action Needed!'
        )
    
    except Exception as e:
        return render_template('index.html', 
            error_message=f"Analysis Error: {str(e)}"), 500

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    try:
        input_data = request.form.to_dict()
        converted_data = {}

        # Data conversion and validation
        for col in INPUT_COLUMNS:
            try:
                if col in ['age', 'avg_glucose_level', 'bmi', 'symptoms_duration']:
                    converted_data[col] = float(input_data.get(col, 0))
                elif col in ['hypertension', 'heart_disease', 'ever_married', 'medication_use']:
                    value = str(input_data.get(col, '0')).lower()
                    converted_data[col] = 1 if value in ['1', 'yes', 'true'] else 0
                else:
                    converted_data[col] = input_data.get(col, 'Unknown')
            except Exception as e:
                error_msg = f"Invalid value for {col}: {input_data.get(col)}"
                logger.error(error_msg)
                return render_template('error.html', error_message=error_msg), 400

        # Make predictions
        df_input = pd.DataFrame([converted_data])
        predictions = RECOMMENDATION_MODEL.predict(df_input)

        # Decode predictions
        decoded_predictions = {}
        for i, col in enumerate(OUTPUT_COLUMNS):
            decoder = LABEL_ENCODERS[col]  # Use the loaded encoders
            decoded_predictions[col] = decoder.inverse_transform([predictions[0][i]])[0]

        # Prepare results
        results = {
            **decoded_predictions,
            'medication_plan': get_comprehensive_medication_plan(
                decoded_predictions['stroke_type'],
                decoded_predictions['stroke_stage']
            )
        }

        return render_template('results.html', results=results)

    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        return render_template(
            'error.html',
            error_message=f"System Error: {str(e)}. Please check your inputs and try again."
        ), 500

if __name__ == '__main__':
    app.run(debug=True)