# Quick Start Guide

Get your Autism Prediction Model up and running in 3 simple steps!

## 🚀 Quick Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Choose Your Interface

#### Option A: Web Interface (Easiest)
```bash
streamlit run Frontend/app.py
```
Then open your browser to `http://localhost:8501`

#### Option B: API Server
```bash
python APIs/app.py
```
API will be available at `http://localhost:5000`

### Step 3: Test It Out!

**For Web Interface:**
- Fill in the form with patient information
- Click "Get Prediction"
- View results!

**For API:**
- Use the test script:
  ```bash
  python test_api.py
  ```
- Or use curl/Postman to send POST requests to `http://localhost:5000/predict`

## 📝 Example API Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## ⚠️ Troubleshooting

**Problem:** Model not found
- **Solution:** Make sure `ML_model/best_model.pkl` and `ML_model/encoders.pkl` exist

**Problem:** Port already in use
- **Solution:** Change the port in `APIs/app.py` or stop the process using that port

**Problem:** Import errors
- **Solution:** Make sure you've activated your virtual environment and installed all requirements

## 🎯 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Customize the API endpoints as needed
- Deploy to cloud (see README for deployment options)

Happy predicting! 🧩

