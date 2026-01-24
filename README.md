# Autism Prediction Model - End-to-End Deployment

This project contains a complete machine learning pipeline for autism spectrum disorder (ASD) prediction, including a trained model, API backend, and web frontend.

## 📋 Project Structure

```
autism_prediction_model/
├── ML_model/
│   ├── autistic_prediction.ipynb    # Model training notebook
│   ├── best_model.pkl                # Trained RandomForest model
│   └── encoders.pkl                  # Label encoders for categorical features
├── APIs/
│   └── app.py                        # Flask REST API backend
├── Frontend/
│   └── app.py                        # Streamlit web frontend
├── Dataset/
│   └── train - Copy.csv              # Training dataset
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Features

- **Machine Learning Model**: RandomForest classifier trained on autism screening data
- **REST API**: Flask-based API for programmatic access
- **Web Interface**: Streamlit-based user-friendly frontend
- **Data Preprocessing**: Automated preprocessing pipeline matching training data
- **Model Persistence**: Saved model and encoders for easy deployment

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone or navigate to the project directory:**
   ```bash
   cd autism_prediction_model
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Option 1: Streamlit Web Interface (Recommended for End Users)

The easiest way to use the model is through the Streamlit web interface:

```bash
streamlit run Frontend/app.py
```

This will start a local web server (usually at `http://localhost:8501`). Open the URL in your browser to access the interface.

**Features:**
- User-friendly form interface
- All required input fields
- Real-time predictions
- Probability scores
- Visual feedback

### Option 2: Flask API (For Developers/Integration)

Start the Flask API server:

```bash
python APIs/app.py
```

The API will be available at `http://localhost:5000`

#### API Endpoints

**1. Health Check:**
```bash
GET http://localhost:5000/
```

**2. Prediction:**
```bash
POST http://localhost:5000/predict
Content-Type: application/json

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
```

**Response:**
```json
{
    "prediction": 0,
    "prediction_label": "No Autism",
    "probability": {
        "no_autism": 0.85,
        "autism": 0.15
    },
    "confidence": 0.85
}
```

#### Example API Usage (Python)

```python
import requests
import json

url = "http://localhost:5000/predict"
data = {
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

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2))
```

## 📊 Model Details

- **Algorithm**: RandomForest Classifier
- **Features**: 19 features including:
  - A1-A10 screening scores (binary)
  - Age (integer)
  - Gender, Ethnicity, Jaundice, Autism history, Country, App usage, Relation (categorical)
  - Result score (float)
- **Accuracy**: ~84% on test set
- **Preprocessing**: Label encoding, outlier handling, SMOTE for class imbalance

## 🌐 Deployment Options

### Local Deployment

Both the Streamlit frontend and Flask API can run locally as shown above.

### Cloud Deployment

#### Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Deploy!

#### Heroku

1. Create a `Procfile`:
   ```
   web: python APIs/app.py
   ```
2. Deploy using Heroku CLI:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

#### AWS/GCP/Azure

- Use containerization (Docker) for easier deployment
- Deploy Flask API as a web service
- Use load balancers for production

### Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "APIs/app.py"]
```

Build and run:
```bash
docker build -t autism-prediction .
docker run -p 5000:5000 autism-prediction
```

## 🔧 Configuration

### API Configuration

Edit `APIs/app.py` to change:
- Port number (default: 5000)
- Host (default: 0.0.0.0)
- Debug mode (default: True)

### Model Paths

Ensure the model files are in the correct locations:
- Model: `ML_model/best_model.pkl`
- Encoders: `ML_model/encoders.pkl`

## ⚠️ Important Notes

1. **Medical Disclaimer**: This model is for screening purposes only and should not replace professional medical diagnosis.

2. **Data Privacy**: Ensure proper data handling and privacy measures when deploying in production.

3. **Model Updates**: If you retrain the model, ensure the preprocessing pipeline matches the training pipeline.

## 🐛 Troubleshooting

### Model/Encoder Loading Errors

- Verify that `best_model.pkl` and `encoders.pkl` exist in `ML_model/` directory
- Check file paths are correct relative to the script location

### Port Already in Use

- Change the port in `APIs/app.py` (line with `app.run()`)
- Or stop the process using the port

### Missing Dependencies

- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again

## 📝 License

This project is for educational and research purposes.

## 👤 Author

Built as part of an autism prediction model project.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

---

**For questions or issues, please open an issue on the repository.**

