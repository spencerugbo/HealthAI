from flask import Flask, request, jsonify
from openai import OpenAI
from translate import Translator
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
    if language not in ['en','ga','fr','es','pt','it','pl','nl','de']:
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
    translator = Translator(provide='libre', from_lang='en', to_lang=language, secret_access_key=os.environ['LT_API_KEYS'])

    prompt = (
        f"The user has been tested for {illness} based on their symptoms, and the result is {prediction}. "
        "Break the news to the user and provide appropriate advice, including necessary medical disclaimers, and suggest the next steps they should take."
    )
    
    response = client.chat.completions.create(
        model = "gpt-4o-mini-2024-07-18",
        messages = [
            {"role": "system", "content": f"You are Doc-Bot, a symptom-checker used to detect the likelihood of a user having any of the conditions currently being checked . {prompt}"},
        ]
    )

    advice = response.choices[0].message.content

    if language != 'en':
        advice = translator.translate(advice)

    return advice

        

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)