"""
Flask API for Autism Prediction Model
This API serves predictions for autism screening based on user inputs.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Load model and encoders
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ML_model', 'best_model.pkl')
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), '..', 'ML_model', 'encoders.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    with open(ENCODERS_PATH, 'rb') as f:
        encoders = pickle.load(f)
    print("Encoders loaded successfully!")
except Exception as e:
    print(f"Error loading encoders: {e}")
    encoders = None


def preprocess_input(data):
    """
    Preprocess input data to match the training pipeline.
    
    Args:
        data: Dictionary containing user input
        
    Returns:
        numpy array ready for model prediction
    """
    # Create a DataFrame from input
    df = pd.DataFrame([data])
    
    # Handle country name mapping (same as training)
    if 'contry_of_res' in df.columns:
        mapping = {
            'Viet Nam': 'Vietnam',
            'AmericanSamoa': 'United States',
            'Hong Kong': 'China'
        }
        df['contry_of_res'] = df['contry_of_res'].replace(mapping)
    
    # Handle ethnicity mapping
    if 'ethnicity' in df.columns:
        df['ethnicity'] = df['ethnicity'].replace({'?': 'Others', 'others': 'Others'})
    
    # Handle relation mapping
    if 'relation' in df.columns:
        df['relation'] = df['relation'].replace({
            '?': 'Others',
            'Relative': 'Others',
            'Parent': 'Others',
            'Health care professional': 'Others'
        })
    
    # Apply label encoding using saved encoders
    if encoders:
        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    # Handle unseen categories
                    unique_values = df[col].unique()
                    for val in unique_values:
                        if val not in encoder.classes_:
                            # Replace with most common class or 'Others'
                            if col == 'ethnicity':
                                df[col] = df[col].replace(val, 'Others')
                            elif col == 'relation':
                                df[col] = df[col].replace(val, 'Others')
                            else:
                                # Use the first class as default
                                df[col] = df[col].replace(val, encoder.classes_[0])
                    
                    df[col] = encoder.transform(df[col])
                except Exception as e:
                    print(f"Error encoding {col}: {e}")
                    # Use default value
                    df[col] = 0
    
    # Convert age to int
    if 'age' in df.columns:
        df['age'] = df['age'].astype(int)
    
    # Ensure correct column order (matching training data)
    expected_columns = [
        'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
        'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
        'age', 'gender', 'ethnicity', 'jaundice', 'austim',
        'contry_of_res', 'used_app_before', 'result', 'relation'
    ]
    
    # Ensure all expected columns exist, add missing ones with default values
    for col in expected_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' is missing, adding with default value 0")
            df[col] = 0
    
    # Reorder columns to match training data
    df = df[expected_columns]
    
    return df.values


@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Autism Prediction API is running',
        'model_loaded': model is not None,
        'encoders_loaded': encoders is not None
    })


@app.route('/predict', methods=['POST', 'OPTIONS', 'GET'])
@app.route('/predict/', methods=['POST', 'OPTIONS', 'GET'])
def predict():
    """
    Predict endpoint for autism screening.
    
    Expected JSON payload:
    {
        "A1_Score": 1,
        "A2_Score": 0,
        "A3_Score": 1,
        "A4_Score": 0,
        "A5_Score": 1,
        "A6_Score": 0,
        "A7_Score": 1,
        "A8_Score": 0,
        "A9_Score": 1,
        "A10_Score": 1,
        "age": 25,
        "gender": "m",
        "ethnicity": "White-European",
        "jaundice": "no",
        "austim": "no",
        "contry_of_res": "United States",
        "used_app_before": "no",
        "result": 6.5,
        "relation": "Self"
    }
    """
    # Handle CORS preflight requests
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    # Handle GET requests (provide usage information)
    if request.method == 'GET':
        return jsonify({
            'message': 'This endpoint requires a POST request with JSON data',
            'usage': {
                'method': 'POST',
                'content_type': 'application/json',
                'endpoint': '/predict',
                'required_fields': [
                    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
                    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
                    'age', 'gender', 'ethnicity', 'jaundice', 'austim',
                    'contry_of_res', 'used_app_before', 'result', 'relation'
                ],
                'example': {
                    'A1_Score': 1,
                    'A2_Score': 0,
                    'A3_Score': 1,
                    'A4_Score': 0,
                    'A5_Score': 1,
                    'A6_Score': 0,
                    'A7_Score': 1,
                    'A8_Score': 0,
                    'A9_Score': 1,
                    'A10_Score': 1,
                    'age': 25,
                    'gender': 'm',
                    'ethnicity': 'White-European',
                    'jaundice': 'no',
                    'austim': 'no',
                    'contry_of_res': 'United States',
                    'used_app_before': 'no',
                    'result': 6.5,
                    'relation': 'Self'
                }
            }
        }), 200
    
    if model is None or encoders is None:
        return jsonify({
            'error': 'Model or encoders not loaded. Please check server logs.'
        }), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Required fields
        required_fields = [
            'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
            'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
            'age', 'gender', 'ethnicity', 'jaundice', 'austim',
            'contry_of_res', 'used_app_before', 'result', 'relation'
        ]
        
        # Check if all required fields are present
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Preprocess input
        processed_data = preprocess_input(data)
        
        # Validate processed data shape
        if processed_data.shape[1] != 19:
            return jsonify({
                'error': f'Processed data has incorrect shape. Expected 19 features, got {processed_data.shape[1]}'
            }), 500
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Get prediction probabilities if available
        try:
            prediction_proba = model.predict_proba(processed_data)[0]
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Autism Detected' if prediction == 1 else 'No Autism',
                'probability': {
                    'no_autism': float(prediction_proba[0]),
                    'autism': float(prediction_proba[1])
                },
                'confidence': float(max(prediction_proba))
            }
        except AttributeError:
            # Model doesn't support predict_proba
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Autism Detected' if prediction == 1 else 'No Autism',
                'probability': None,
                'confidence': None
            }
        
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response, 200
        
    except KeyError as e:
        return jsonify({
            'error': f'Missing required field in data: {str(e)}'
        }), 400
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Prediction error: {error_trace}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'details': error_trace if app.debug else None
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'encoders_loaded': encoders is not None
    })


@app.errorhandler(405)
def method_not_allowed(e):
    """Handle 405 Method Not Allowed errors"""
    return jsonify({
        'error': 'Method not allowed',
        'message': f'The {request.method} method is not allowed for this endpoint. Use POST instead.',
        'endpoint': request.path,
        'allowed_methods': ['POST', 'OPTIONS']
    }), 405


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

