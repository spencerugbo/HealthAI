import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import joblib

def create_lung_cancer_model():
    file_path = './datasets/lung_cancer.csv'
    df = pd.read_csv(file_path)
    df = df.dropna()

    label_encoder = LabelEncoder()
    df['LUNG_CANCER'] = label_encoder.fit_transform(df['LUNG_CANCER'])
    df['GENDER'] = label_encoder.fit_transform(df['GENDER'])

    X = df.drop('LUNG_CANCER', axis=1).values
    y = df['LUNG_CANCER'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_filename = './models/scalers/lung_cancer_scaler.joblib'
    joblib.dump(scaler, scaler_filename)

    model = keras.Sequential()
    model.add(keras.layers.Dense(units=32, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(keras.layers.Dense(units=16, activation='relu'))
    model.add(keras.layers.Dense(units=8, activation='relu'))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    with open('./metrics/lung_cancer_metrics.json', 'w') as f:
        json.dump(metrics, f)

    model_filename = './models/lung_cancer_model.keras'
    model.save(model_filename)

def create_heart_disease_model():
    file_path = './datasets/heart_disease.xlsx'
    df = pd.read_excel(file_path)
    df = df.dropna()

    target_column = 'Heart Disease'
    df[target_column] = df[target_column].map({'Yes': 1, 'No': 0})
    categorical_columns = ['sex', 'cp', 'restecg', 'exang', 'slope']
    label_encoders = {}
    for column in categorical_columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
        label_encoders[column] = label_encoder

    X = df.drop(target_column, axis=1).values
    y = df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_filename = './models/scalers/heart_disease_scaler.joblib'
    joblib.dump(scaler, scaler_filename)

    model = keras.Sequential()
    model.add(keras.layers.Dense(units=32, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(keras.layers.Dense(units=16, activation='relu'))
    model.add(keras.layers.Dense(units=8, activation='relu'))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    y_pred_prob = model.predict(X_test_scaled) 
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    with open('./metrics/heart_disease_metrics.json', 'w') as f:
        json.dump(metrics, f)

    model_filename = './models/heart_disease_model.keras'
    model.save(model_filename)


def create_liver_disease_model():
    file_path = './datasets/liver_disease.csv'
    df = pd.read_csv(file_path)
    df = df.dropna()

    labelEncoder = LabelEncoder()
    df['Gender'] = labelEncoder.fit_transform(df['Gender'])

    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_filename = "./models/scalers/liver_disease_scaler.joblib"
    joblib.dump(scaler, scaler_filename)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = keras.Sequential()
    model.add(keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    with open('./metrics/liver_disease_metrics.json', 'w') as f:
        json.dump(metrics, f)

    model_filename = "./models/liver_disease_model.keras"
    model.save(model_filename)
    

def main():
    create_lung_cancer_model()
    create_heart_disease_model()
    create_liver_disease_model()

if __name__ == "__main__":
    main()