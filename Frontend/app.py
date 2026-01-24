"""
Streamlit Frontend for Autism Prediction Model
A user-friendly interface for autism screening predictions.
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path to import from APIs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Autism Prediction Model",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'ML_model', 'best_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_encoders():
    """Load the label encoders"""
    encoders_path = os.path.join(os.path.dirname(__file__), '..', 'ML_model', 'encoders.pkl')
    try:
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        return encoders
    except Exception as e:
        st.error(f"Error loading encoders: {e}")
        return None

def preprocess_input(data, encoders):
    """Preprocess input data to match training pipeline"""
    df = pd.DataFrame([data])
    
    # Handle country name mapping
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
    
    # Apply label encoding
    if encoders:
        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    # Handle unseen categories
                    unique_values = df[col].unique()
                    for val in unique_values:
                        if val not in encoder.classes_:
                            if col == 'ethnicity':
                                df[col] = df[col].replace(val, 'Others')
                            elif col == 'relation':
                                df[col] = df[col].replace(val, 'Others')
                            else:
                                df[col] = df[col].replace(val, encoder.classes_[0])
                    
                    df[col] = encoder.transform(df[col])
                except Exception as e:
                    st.warning(f"Error encoding {col}: {e}")
                    df[col] = 0
    
    # Convert age to int
    if 'age' in df.columns:
        df['age'] = df['age'].astype(int)
    
    # Ensure correct column order
    expected_columns = [
        'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
        'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
        'age', 'gender', 'ethnicity', 'jaundice', 'austim',
        'contry_of_res', 'used_app_before', 'result', 'relation'
    ]
    
    df = df[expected_columns]
    return df.values

# Main app
def main():
    st.markdown('<div class="main-header">🧩 Autism Prediction Model</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Autism Screening Tool
    
    This application uses a machine learning model to predict the likelihood of autism spectrum disorder 
    based on screening questionnaire responses and demographic information. Please fill in all the fields below.
    
    **Note:** This is a screening tool and should not be used as a substitute for professional medical diagnosis.
    """)
    
    # Load model and encoders
    model = load_model()
    encoders = load_encoders()
    
    if model is None or encoders is None:
        st.error("⚠️ Model or encoders could not be loaded. Please check the file paths.")
        return
    
    # Create form for user input
    with st.form("prediction_form"):
        st.markdown("### Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=25, step=1)
            gender = st.selectbox("Gender", ["m", "f"], format_func=lambda x: "Male" if x == "m" else "Female")
            ethnicity = st.selectbox("Ethnicity", [
                "White-European", "Middle Eastern ", "Pasifika", "Black", "Others",
                "Hispanic", "Asian", "Turkish", "South Asian", "Latino"
            ])
            jaundice = st.selectbox("Born with Jaundice?", ["no", "yes"], format_func=lambda x: "No" if x == "no" else "Yes")
        
        with col2:
            austim = st.selectbox("Family member with Pervasive Development Disorder?", ["no", "yes"], format_func=lambda x: "No" if x == "no" else "Yes")
            contry_of_res = st.selectbox("Country of Residence", [
                "United States", "United Kingdom", "India", "Canada", "Australia",
                "New Zealand", "Brazil", "South Africa", "Jordan", "Austria",
                "United Arab Emirates", "Ukraine", "Iraq", "France", "Malaysia",
                "Vietnam", "Egypt", "Netherlands", "Afghanistan", "Oman", "Italy",
                "Bahamas", "Saudi Arabia", "Ireland", "Aruba", "Sri Lanka", "Russia",
                "Bolivia", "Azerbaijan", "Armenia", "Serbia", "Ethiopia", "Sweden",
                "Iceland", "Angola", "China", "Germany", "Spain", "Tonga", "Pakistan",
                "Iran", "Argentina", "Japan", "Mexico", "Nicaragua", "Sierra Leone",
                "Czech Republic", "Niger", "Romania", "Cyprus", "Belgium", "Burundi",
                "Bangladesh", "Kazakhstan"
            ])
            used_app_before = st.selectbox("Used screening app before?", ["no", "yes"], format_func=lambda x: "No" if x == "no" else "Yes")
            relation = st.selectbox("Who is completing the test?", ["Self", "Others"])
        
        st.markdown("### Screening Questionnaire (A1-A10)")
        st.markdown("Please answer each question (0 = No, 1 = Yes):")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            a1 = st.selectbox("A1 Score", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            a2 = st.selectbox("A2 Score", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col2:
            a3 = st.selectbox("A3 Score", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            a4 = st.selectbox("A4 Score", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col3:
            a5 = st.selectbox("A5 Score", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            a6 = st.selectbox("A6 Score", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col4:
            a7 = st.selectbox("A7 Score", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            a8 = st.selectbox("A8 Score", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col5:
            a9 = st.selectbox("A9 Score", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            a10 = st.selectbox("A10 Score", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        st.markdown("### Additional Information")
        result = st.number_input("Screening Test Result Score", min_value=-10.0, max_value=20.0, value=0.0, step=0.1,
                                help="The result score from the screening test")
        
        submitted = st.form_submit_button("🔍 Get Prediction", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = {
                'A1_Score': a1,
                'A2_Score': a2,
                'A3_Score': a3,
                'A4_Score': a4,
                'A5_Score': a5,
                'A6_Score': a6,
                'A7_Score': a7,
                'A8_Score': a8,
                'A9_Score': a9,
                'A10_Score': a10,
                'age': age,
                'gender': gender,
                'ethnicity': ethnicity,
                'jaundice': jaundice,
                'austim': austim,
                'contry_of_res': contry_of_res,
                'used_app_before': used_app_before,
                'result': result,
                'relation': relation
            }
            
            # Preprocess and predict
            try:
                processed_data = preprocess_input(input_data, encoders)
                prediction = model.predict(processed_data)[0]
                prediction_proba = model.predict_proba(processed_data)[0]
                
                # Display results
                st.markdown("---")
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("## ⚠️ Autism Detected")
                        st.markdown(f"**Confidence:** {prediction_proba[1]*100:.2f}%")
                    else:
                        st.success("## ✅ No Autism Detected")
                        st.markdown(f"**Confidence:** {prediction_proba[0]*100:.2f}%")
                
                with col2:
                    st.markdown("### Probability Breakdown")
                    st.progress(prediction_proba[0])
                    st.markdown(f"**No Autism:** {prediction_proba[0]*100:.2f}%")
                    st.progress(prediction_proba[1])
                    st.markdown(f"**Autism:** {prediction_proba[1]*100:.2f}%")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Disclaimer
                st.info("""
                **Important Disclaimer:** This prediction is based on a machine learning model and should not be 
                considered a medical diagnosis. Please consult with a qualified healthcare professional for 
                proper evaluation and diagnosis.
                """)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()

