import pandas as pd
import numpy as np
from textblob import TextBlob
import joblib
import json
import re
import os
from datetime import datetime

class TweetPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.company_encoder = None
        self.username_encoder = None
        self.feature_columns = []
        self.model_metadata = {}
        
        # Model paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = os.path.join(self.base_dir, "ai_model")
        
        # Individual component paths
        self.model_path = os.path.join(self.model_dir, "tweet_prediction_model.pkl")
        self.scaler_path = os.path.join(self.model_dir, "feature_scaler.pkl")
        self.company_encoder_path = os.path.join(self.model_dir, "company_encoder.pkl")
        self.username_encoder_path = os.path.join(self.model_dir, "username_encoder.pkl")
        self.metadata_path = os.path.join(self.model_dir, "model_metadata.json")
    
    def load_model(self):
        """Load LightGBM model components individually (avoiding pickle issues)"""
        try:
            print("📦 Loading LightGBM model components...")
            
            # Load individual components
            if all(os.path.exists(path) for path in [
                self.model_path, self.scaler_path, 
                self.company_encoder_path, self.username_encoder_path
            ]):
                
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.company_encoder = joblib.load(self.company_encoder_path)
                self.username_encoder = joblib.load(self.username_encoder_path)
                
                # Load metadata from JSON
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        self.model_metadata = json.load(f)
                    self.feature_columns = self.model_metadata.get('feature_columns', [])
                else:
                    # Fallback feature columns
                    self.feature_columns = [
                        'word_count', 'char_count', 'sentiment', 'subjectivity',
                        'emoji_count', 'hashtag_count', 'mention_count', 'url_count',
                        'has_media', 'hour', 'day_of_week', 'is_weekend', 'month',
                        'company_encoded', 'username_encoded'
                    ]
                
                print(f"✅ Loaded LightGBM model successfully")
                if self.model_metadata:
                    print(f"📊 R² Score: {self.model_metadata.get('r2_score', 'N/A')}")
                return True
            else:
                print("❌ Model component files not found!")
                print(f"   Model: {os.path.exists(self.model_path)}")
                print(f"   Scaler: {os.path.exists(self.scaler_path)}")
                print(f"   Company Encoder: {os.path.exists(self.company_encoder_path)}")
                print(f"   Username Encoder: {os.path.exists(self.username_encoder_path)}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_features(self, content, username, company, media, hour):
        """Extract features for LightGBM prediction"""
        features = {}
        
        # Text features
        features['word_count'] = len(content.split())
        features['char_count'] = len(content)
        features['sentiment'] = TextBlob(content).sentiment.polarity
        features['subjectivity'] = TextBlob(content).sentiment.subjectivity
        
        # Special character counts
        features['emoji_count'] = len(re.findall(r'[😀-🙏🌀-🗿🚀-🛿]', content))
        features['hashtag_count'] = len(re.findall(r'#\w+', content))
        features['mention_count'] = len(re.findall(r'@\w+', content))
        features['url_count'] = len(re.findall(r'http\S+|www\S+', content))
        
        # Media feature
        features['has_media'] = 1 if media else 0
        
        # Time features
        features['hour'] = hour if hour is not None else 12
        features['day_of_week'] = datetime.now().weekday()
        features['is_weekend'] = 1 if datetime.now().weekday() >= 5 else 0
        features['month'] = datetime.now().month
        
        # Categorical features
        try:
            features['company_encoded'] = self.company_encoder.transform([company.lower()])[0]
        except:
            features['company_encoded'] = 0
            
        try:
            features['username_encoded'] = self.username_encoder.transform([username.lower()])[0]
        except:
            features['username_encoded'] = 0
        
        return features
    
    def predict(self, content, username, company, media, hour=None):
        """Make prediction using LightGBM model"""
        if self.model is None:
            raise ValueError("LightGBM model not loaded! Call load_model() first.")
        
        # Extract features
        features = self.extract_features(content, username, company, media, hour)
        
        # Create feature vector
        feature_vector = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        
        # LightGBM prediction (no scaling needed)
        predicted_likes = self.model.predict(feature_vector)[0]
        predicted_likes = max(0, int(predicted_likes))
        
        # Determine engagement level
        if predicted_likes < 50:
            engagement_level = "Low"
            confidence = 0.65
        elif predicted_likes < 200:
            engagement_level = "Medium"
            confidence = 0.80
        else:
            engagement_level = "High"
            confidence = 0.90
        
        return {
            "predicted_likes": predicted_likes,
            "engagement_level": engagement_level,
            "confidence": confidence
        }
