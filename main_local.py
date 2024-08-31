from openai import OpenAI
import keras
import numpy as np
from joblib import load
import os



def getUserSymptoms():
    disease_check_choice = input("Which disease do you want to check for?\n1. Heart Disease\n2. Lung Cancer\n3. Liver Disease\n")
    match disease_check_choice:
        case "1":
            prediction = get_heart_disease_prediction()
            advice = generate_advice('heart disease', prediction)
            print(f"\n\nDoc-Bot\n{advice}")
        case "2":
            prediction = get_lung_cancer_prediction()
            advice = generate_advice('lung cancer', prediction)
            print(f"\n\nDoc-Bot\n{advice}")
        case "3":
            prediction = get_liver_disease_prediction()
            advice = generate_advice('liver disease', prediction)
            print(f"\n\nDoc-Bot\n{advice}")
        case _:
            print("You did not enter a valid option")


def get_lung_cancer_prediction():
    gender = get_gender_input("What is your gender? (M/F): ", 1, 0)
    age = get_int_input("How old are you? ")
    smoking = get_yes_no_input("Do you smoke? (Y/N): ", 2, 1)
    yellow_fingers = get_yes_no_input("Do you have yellow fingers? (Y/N): ", 2, 1)
    anxiety = get_yes_no_input("Do you experience anxiety? (Y/N): ", 2, 1)
    peer_presseure = get_yes_no_input("Do you feel peer pressure to smoke? (Y/N): ", 2, 1)
    chronic_disease = get_yes_no_input("Do you have any chronic illnesses? (Y/N): ", 2, 1)
    fatigue = get_yes_no_input("Are you feeling fatigued? (Y/N): ", 2, 1)
    allergy = get_yes_no_input("Do you have any allergies? (Y/N): ", 2, 1)
    wheezing = get_yes_no_input("Do you experience wheezing? (Y/N): ", 2, 1)
    alcohol_consumption = get_yes_no_input("Do you consume alcohol? (Y/N): ", 2, 1)
    coughing = get_yes_no_input("Are you coughing? (Y/N): ", 2, 1)
    shortness_of_breath = get_yes_no_input("Do you experience a shortness of breath? (Y/N): ", 2, 1)
    swallowing_difficulty = get_yes_no_input("Do you have difficulty swallowing? (Y/N): ", 2, 1)
    chest_pain = get_yes_no_input("Do you experience chest pain? (Y/N): ", 2, 1)
    model_file_path = './models/lung_cancer_model.keras'
    model = keras.saving.load_model(model_file_path)
    input_features = np.array([[gender, age, smoking, yellow_fingers, anxiety, peer_presseure, chronic_disease, 
                                fatigue, allergy, wheezing, alcohol_consumption, coughing, 
                                shortness_of_breath, swallowing_difficulty, chest_pain]])
    prediction_prob = model.predict(input_features)
    prediction = (prediction_prob > 0.5).astype(int)[0][0]
    return prediction

def get_heart_disease_prediction():
    sex = get_gender_input("What is your sex? (M/F): ", 1, 0)
    age = get_int_input("How old are you? ")
    chest_pain = get_range_input("Rate your chest pain (0 = None, 1 = Light, 2 = Moderate, 3 = Severe): ", 0, 3)
    resting_bp = get_int_input("What is your resting blood pressure? (in mm/Hg): ")
    cholesterol = get_int_input("What is your serum cholesterol? (in mg/dl): ")
    fasting_blood_sugar = get_yes_no_input("Is your fasting blood sugar above 120 mg/dl? (Y/N): ", 1, 0)
    resting_ecg = get_range_input("What is your resting electrocardiographic (ECG) result? (0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy): ", 0, 2)
    maximum_bpm = get_int_input("What is your maximum heart rate achieved during exercise? ")
    excercise_induced_angina = get_yes_no_input("Did you experience exercise-induced angina? (Y/N): ", 1, 0)
    old_peak = get_float_input("What is the ST depression induced by exercise relative to rest? ")
    slope = get_range_input("What is the slope of the peak exercise ST segment? (0 = Upsloping, 1 = Flat, 2 = Downsloping): ", 0, 2)
    major_vessels = get_range_input("How many major vessels were colored by fluoroscopy? (Enter a number from 0 to 3): ", 0, 3)
    thallium_stress_test = get_range_input("What was the result of your thallium stress test? (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect): ", 1, 3)
    model_file_path = './models/heart_disease_model.keras'
    model = keras.saving.load_model(model_file_path)
    input_features = np.array(([[age,sex,chest_pain,resting_bp,cholesterol,fasting_blood_sugar,
                                resting_ecg, maximum_bpm,excercise_induced_angina,old_peak,slope,
                                major_vessels,thallium_stress_test]]))
    prediction_prob = model.predict(input_features)
    prediction = (prediction_prob > 0.5).astype(int)[0][0]
    return prediction

def get_liver_disease_prediction():
    age = get_int_input("How old are you? ")
    gender = get_gender_input("What is your gender? (M/F): ", 0, 1)
    bmi = get_float_input("What is your BMI? ")
    alcohol_consumption = get_range_float_input("What is your weekly alcohol consumption? (0 to 20 units per week): ", 0, 20)
    smoking = get_yes_no_input("Do you smoke? ", 1, 0)
    genetic_risk = get_range_input("Are you at genetic risk? (0 = Low, 1 = Medium 2, = High): ", 0, 2)
    physical_activity = get_range_float_input("How many hours of physical activity do you do per week? (0 to 10 hours): ", 0, 10)
    diabetes = get_yes_no_input("Do you have diabetes? ", 1, 0)
    hypertension = get_yes_no_input("Do you have hypertension (high blood pressure)?  ", 1, 0)
    liver_function_test = get_range_float_input("Please enter your liver function test result (20 to 100): ", 20, 100)
    model_file_path = "./models/liver_disease_model.keras"
    model = keras.saving.load_model(model_file_path)
    input_features = np.array([[age, gender, bmi, alcohol_consumption, smoking, genetic_risk, 
                                physical_activity, diabetes, hypertension, liver_function_test]])
    prediction_prob = model.predict(input_features)
    prediction = (prediction_prob > 0.5).astype(int)[0][0]
    return prediction


def generate_advice(illness, prediction):
    client = OpenAI()

    result = 'positive' if prediction == 1 else 'negative'

    prompt = (
        f"The user has been tested for {illness} based on their symptoms, and the result is {result}. "
        "Break the news to the user and provide appropriate advice, including necessary medical disclaimers, and suggest the next steps they should take."
    )
    
    response = client.chat.completions.create(
        model = "gpt-4o-mini-2024-07-18",
        messages = [
            {"role": "system", "content": f"You are Doc-Bot, a symptom-checker used to detect the likelihood of a user having any of the conditions currently being checked . {prompt}"},
        ]
    )

    return response.choices[0].message.content




def get_int_input(prompt):
    while True:
        try:
            user_input = int(input(prompt).strip())
            if user_input >= 0:
                return user_input
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a valid integer")

def get_yes_no_input(prompt, yes_value, no_value):
    while True:
        user_input = input(prompt).strip().upper()
        if user_input == 'Y':
            return yes_value
        elif user_input == 'N':
            return no_value
        else:
            print("Invalid input. Please enter 'Y' for Yes or 'N' for No.")

def get_gender_input(prompt, male_value, female_value):
    while True:
        user_input = input(prompt).strip().upper()
        if user_input == 'M':
            return male_value
        elif user_input == 'F':
            return female_value
        else:
            print("Invalid input. Please enter 'M' for Male or 'F' for Female.")

def get_range_input(prompt, lowest_value, highest_value):
    while True:
        try:
            user_input = int(input(prompt).strip())
            if user_input in range(lowest_value, highest_value + 1):
                return user_input
            else:
                print(f"Invalid input. Please enter a number between {lowest_value} and {highest_value}")
        except ValueError:
            print("Invalid input. Please enter in a valid integer")

def get_range_float_input(prompt, lowest_value, highest_value):
    while True:
        try:
            user_input = float(input(prompt).strip())
            if lowest_value <= user_input <= highest_value:
                return user_input
            else:
                print(f"Invalid input. Please enter a number between {lowest_value} and {highest_value}")
        except ValueError:
            print("Invalid input. Please enter in a valid value")

def get_float_input(prompt):
    while True:
        try:
            user_input = float(input(prompt).strip())
            if user_input >= 0:
                return user_input
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a valid floating number")

        
def main():
    getUserSymptoms()

if __name__ == "__main__":
    main()