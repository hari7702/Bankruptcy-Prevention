#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(Bankruptcy)

# Define paths to the models
MODEL_PATHS = {
    'Support Vector Classifier': '"C:\Users\haria\Documents\ExcelR\ExcelR Project\Bank\best_support_vector_classifier_model.pkl"'
}

# Function to load model
def load_model(best_support_vector_classifier):
    model_path = MODEL_PATHS[best_support_vector_classifier]
    return joblib.load(model_path)

# Load the best model initially (you can change this as needed)
best_model_name = 'Random Forest Classifier'
model = load_model(best_model_name)

@app.route('/')
def home():
    return "Welcome to the Bankruptcy Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data.get('features')
    if not features or len(features) != 6:
        return jsonify({'error': 'Invalid input. Please provide exactly 6 feature values.'}), 400
    prediction = model.predict(np.array(features).reshape(1, -1))
    result = 'Bankruptcy' if prediction[0] == 1 else 'Non-Bankruptcy'
    return jsonify({'prediction': result})

@app.route('/load_model', methods=['POST'])
def load_new_model():
    global model
    data = request.get_json(force=True)
    new_model_name = data.get('model_name')
    if new_model_name not in MODEL_PATHS:
        return jsonify({'error': 'Model not found. Please provide a valid model name.'}), 400
    model = load_model(new_model_name)
    return jsonify({'message': f'Loaded model: {new_model_name}'})

if __name__ == '__main__':
    app.run(debug=True)

