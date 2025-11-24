# main.py
from fastapi import FastAPI
from pydantic import BaseModel, validator
import joblib
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize FastAPI App
app = FastAPI(title="Dynamic Accessibility Scoring Engine")

# --- 1. Load the Model and NLP Tools ---

# Load the trained ML model (ensure 'scoring_model.pkl' is in the same directory)
try:
    model = joblib.load('scoring_model.pkl')
    print("ML Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Handle case where model fails to load

# Import and initialize NLTK tools
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# If you installed it correctly in Step 1, it should load fine now.
# We keep this line simple for local API testing.
sia = SentimentIntensityAnalyzer()


# --- 2. Define Request Schema ---
# This defines the data structure the API expects from the user
class VenueData(BaseModel):
    Doorway_Width_cm: float
    Ramp_Angle_deg: float
    Clear_Path_Blocked: int # 0 or 1
    Compliance_Violations: int
    User_Review_Text: str

    # Optional: Basic validation to ensure inputs are reasonable
    @validator('Doorway_Width_cm')
    def validate_width(cls, value):
        if not 75 <= value <= 120:
            raise ValueError('Doorway width must be between 75 and 120 cm.')
        return value

# --- 3. Define the Prediction Endpoint ---

@app.post("/api/score")
def predict_score(data: VenueData):
    """
    Accepts venue data and returns the calculated Accessibility Score.
    """
    if model is None:
        return {"error": "Scoring model is not loaded."}
    
    # 1. Extract the sentiment score from the review text (NLP Feature Engineering)
    review_score = sia.polarity_scores(data.User_Review_Text)['compound']
    
    # 2. Prepare the features for the ML model (must match the order used in training!)
    features = np.array([
        data.Doorway_Width_cm, 
        data.Ramp_Angle_deg, 
        data.Clear_Path_Blocked,
        data.Compliance_Violations, 
        review_score
    ]).reshape(1, -1)
    
    # 3. Predict the score using the loaded ML model
    predicted_score = model.predict(features)[0]
    
    # 4. Format and return the result
    return {
        "venue_score": round(float(predicted_score), 2),
        "score_unit": "0-100",
        "review_sentiment": round(review_score, 2),
        "model_used": "LinearRegression"
    }

# --- 4. Root/Health Check Endpoint (Optional) ---
@app.get("/")
def home():
    return {"message": "Welcome to the Dynamic Accessibility Scoring Engine API. Use /api/score for predictions."}