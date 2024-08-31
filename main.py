from flask import Flask, request, jsonify
from openai import OpenAI
from deep_translator import GoogleTranslator
from joblib import load
import keras
import numpy as np
import json
import os

app = Flask(__name__)

models = {
    "lung_cancer":{
        "model": keras.saving.load_model('./models/lung_cancer_model.keras'),
        "scaler": load('./models/scalers/lung_cancer_scaler.joblib'),
        "symptoms": ["GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC_DISEASE", 
                    "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMPTION", "COUGHING", "SHORTNESS_OF_BREATH", 
                    "SWALLOWING_DIFFICULTY", "CHEST_PAIN"],
        "metrics": json.load(open("./metrics/lung_cancer_metrics.json"))
    },
    "heart_disease":{
        "model": keras.saving.load_model('./models/heart_disease_model.keras'),
        "scaler": load('./models/scalers/heart_disease_scaler.joblib'),
        "symptoms": ["AGE", "SEX", "CHEST_PAIN", "RESTING_BLOOD_PRESSURE", "CHOLESTEROL", "FASTING_BLOOD_SUGAR", 
                    "RESTING_ECG", "MAX_HEART_RATE", "EXERCISE_INDUCED_ANGINA", "OLDPEAK", "SLOPE", 
                    "NUMBER_OF_MAJOR_VESSELS", "THALLIUM_STRESS_TEST"],
        "metrics": json.load(open("./metrics/lung_cancer_metrics.json"))
    },
    "liver_disease":{
        "model": keras.saving.load_model('./models/liver_disease_model.keras'),
        "scaler": load('./models/scalers/liver_disease_scaler.joblib'),
        "symptoms": ["AGE", "GENDER", "BMI", "ALCOHOL_CONSUMPTION", "SMOKING", "GENETIC_RISK", "PHYSICAL_ACTIVITIES", 
                    "DIABETES", "HYPERTENSION", "LIVER_FUNCTION_TEST"],
        "metrics": json.load(open("./metrics/liver_disease_metrics.json"))
    }
}

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.json
    language = data.get('language', 'en') or 'en'
    if language not in ['en','ga','fr','es','pt','it','pl','nl','de','ru','sv','zh','ja','ko']:
        return jsonify({"error": f"Language '{language}' is not supported"}), 400
    illness = data.get('illness')

    if illness not in models:
        return jsonify({"error": f"Illness '{illness}' not supported"}), 400
    
    model_info = models[illness]
    symptoms = model_info["symptoms"]

    input_features = []
    for symptom in symptoms:
        value = data.get(symptom)
        if value is None:
            return jsonify({"error": f"Symptom '{symptom}' is missing"}), 400
        input_features.append(value)
    
    input_array = np.array(input_features).reshape(1,-1)
    input_scaled = model_info["scaler"].transform(input_array)

    prediction_prob = model_info["model"].predict(input_scaled)
    prediction = (prediction_prob > 0.5).astype(int)[0][0]

    result = "positive" if prediction == 1 else "negative"

    advice = generate_advice(illness, result, language)

    return jsonify({
        "prediction": result,
        "advice": advice,
        "model_accuracy": model_info['metrics']['accuracy']
        })

@app.route('/report_accuracy', methods=["GET"])
def report_accuracy():
    illness = request.args.get("illness")

    if illness not in models:
        return jsonify({"error": f"Illness '{illness}' not supported"}), 400
    
    metrics = models[illness]["metrics"]
    
    return jsonify(metrics)
    

def generate_advice(illness, prediction, language):
    client = OpenAI()
    
    response = client.chat.completions.create(
        model = "gpt-4o-mini-2024-07-18",
        messages = [
            {"role": "system", "content": f"You are Doc-Bot, you give adive based on the result of the prediction model."},
           
            {"role": "user", "content": "The prediction model indicates a positive result for heart disease."},
            {"role": "assistant", "content": "Based on your symptoms, you are likely experiencing heart disease. It is important to consult a healthcare provider immediately. They may recommend lifestyle changes, medications, or further tests. Regular follow-ups and a heart-healthy diet can help manage the condition. Please note that this advice is general and should not replace professional medical consultation. Always seek personalized guidance from your healthcare provider."},
            {"role": "user", "content": "The prediction model indicates a negative result for heart disease."},
            {"role": "assistant", "content": "Your symptoms suggests that you do not have heart disease. However, it’s important to maintain a healthy lifestyle to reduce future risk. Regular exercise, a balanced diet, and routine check-ups are advisable. If you have any symptoms or concerns, consult your healthcare provider. This advice is general and should not be used as a substitute for professional medical advice."},
            
            {"role": "user", "content": "The prediction model indicates a positive result for lung cancer."},
            {"role": "assistant", "content": "Based on your symptoms, you may have lung cancer. It is crucial to seek consultation with an oncologist promptly. Early diagnosis and treatment are key for better outcomes. Follow your doctor’s recommendations, which may include imaging tests, biopsies, and a personalized treatment plan. Please note that this advice is general and should not replace professional medical consultation. Always seek personalized guidance from your healthcare team."},
            {"role": "user", "content": "The prediction model indicates a negative result for lung cancer."},
            {"role": "assistant", "content": "Your symptoms suggests that you do not have lung cancer. Continue monitoring your health by avoiding smoking and environmental toxins, and get regular check-ups. Discuss any symptoms or risk factors with your healthcare provider. This advice is general and should not be used as a substitute for professional medical advice."},
            
            {"role": "user", "content": "The prediction model indicates a positive result for liver disease."},
            {"role": "assistant", "content": "According to your symptoms, you may have liver disease. Immediate consultation with a healthcare provider is necessary. They may suggest further tests, lifestyle changes, or medications depending on the type and severity of the liver condition. Regular monitoring and adherence to medical advice are essential. Please note that this advice is general and should not replace professional medical consultation. Always seek personalized guidance from your doctor."},
            {"role": "user", "content": "The prediction model indicates a negative result for liver disease."},
            {"role": "assistant", "content": "Your symptoms indicates that you do not have liver disease. To maintain liver health, follow a balanced diet, avoid excessive alcohol consumption, and get regular medical check-ups. Consult your healthcare provider if you have any symptoms or risk factors. This advice is general and should not be used as a substitute for professional medical advice."},

            {"role": "user", "content": f"The prediction model indicates a {prediction} result for {illness}."}
        ]
    )

    advice = response.choices[0].message.content

    if language != 'en':
        advice = GoogleTranslator(source='en', target=language).translate(text=advice)

    return advice

        

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)