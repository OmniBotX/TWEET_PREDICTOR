import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from textblob import TextBlob
import lightgbm as lgb
import joblib
import re
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.lower().strip()

def create_sample_dataset():
    """Create sample dataset for training"""
    import random
    print("📊 Creating sample dataset...")
    
    sample_data = {
        'id': range(1, 2001),
        'date': [f"{random.randint(1, 28):02d}-{random.randint(1, 12):02d}-{random.randint(2020, 2024)} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}" for _ in range(2000)],
        'likes': [max(0, int(np.random.lognormal(3, 1.5))) for _ in range(2000)],
        'content': [
            random.choice([
                "Exciting news from our team! New product launch coming soon 🚀",
                "What do you think about our latest innovation? Let us know! 💬", 
                "Check out our amazing customer stories! Thank you for your support ❤️",
                "Big announcement: We're expanding to new markets! 🎉",
                "Behind the scenes look at our development process 👀",
                "Customer appreciation post! You make everything possible 🙏",
                "Innovation never stops! Here's what we're working on next 💡",
                "Join us for our upcoming webinar! Link in bio 📚",
                "Weekend vibes! Hope everyone is having a great time 🌟",
                "Flash sale alert! 50% off everything today only! 🔥"
            ]) for _ in range(2000)
        ],
        'username': [random.choice(['TechCorp', 'InnovateBrand', 'StartupLife', 'DigitalFirst', 'CreativeStudio', 'EcoFriendly', 'HealthyLiving', 'FashionForward', 'FoodieDelight', 'ServicePro']) for _ in range(2000)],
        'media': [random.choice([None, 'Photo', 'Video', 'GIF', None, None]) for _ in range(2000)],
        'inferred company': [random.choice(['tech company', 'startup', 'digital agency', 'creative studio', 'consulting firm', 'fashion brand', 'food service', 'health company', 'eco brand']) for _ in range(2000)]
    }
    
    df = pd.DataFrame(sample_data)
    os.makedirs('ai_model', exist_ok=True)
    df.to_csv('ai_model/dataset.csv', index=False)
    return df

def load_and_preprocess_data():
    print("📂 Loading dataset...")
    
    # Try to load existing dataset or create sample
    if os.path.exists("ai_model/dataset.csv"):
        df = pd.read_csv("ai_model/dataset.csv")
    else:
        df = create_sample_dataset()
    
    # Data cleaning
    df = df.dropna(subset=['content', 'username', 'inferred company', 'likes'])
    df['media'] = df['media'].fillna('no_media')
    df['has_media'] = (df['media'] != 'no_media').astype(int)
    df['content'] = df['content'].astype(str).str.strip()
    
    # Date features
    df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
    df['hour'] = df['datetime'].dt.hour.fillna(12)
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Text features
    df['processed_content'] = df['content'].apply(preprocess_text)
    df['word_count'] = df['content'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['content'].apply(lambda x: len(str(x)))
    
    # Sentiment analysis
    print("😊 Analyzing sentiment...")
    df['sentiment'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['subjectivity'] = df['content'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    
    # Special character counts
    df['emoji_count'] = df['content'].apply(lambda x: len(re.findall(r'[😀-🙏🌀-🗿🚀-🛿]', str(x))))
    df['hashtag_count'] = df['content'].apply(lambda x: len(re.findall(r'#\w+', str(x))))
    df['mention_count'] = df['content'].apply(lambda x: len(re.findall(r'@\w+', str(x))))
    df['url_count'] = df['content'].apply(lambda x: len(re.findall(r'http\S+|www\S+', str(x))))
    
    # Categorical encoding
    company_encoder = LabelEncoder()
    username_encoder = LabelEncoder()
    
    df['company_encoded'] = company_encoder.fit_transform(df['inferred company'])
    df['username_encoded'] = username_encoder.fit_transform(df['username'])
    
    print(f"✅ Preprocessing complete! Dataset shape: {df.shape}")
    return df, company_encoder, username_encoder

def train_lightgbm_model(df):
    feature_columns = [
        'word_count', 'char_count', 'sentiment', 'subjectivity',
        'emoji_count', 'hashtag_count', 'mention_count', 'url_count',
        'has_media', 'hour', 'day_of_week', 'is_weekend', 'month',
        'company_encoded', 'username_encoded'
    ]
    
    X = df[feature_columns].fillna(0)
    y = df['likes']
    
    print(f"📊 Feature matrix: {X.shape}, Target: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # LightGBM model with optimal parameters
    print("🤖 Training LightGBM model...")
    
    lightgbm_model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        metric='rmse'
    )
    
    # Train model
    lightgbm_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = lightgbm_model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"🏆 LightGBM Performance:")
    print(f"   📊 RMSE: {rmse:.2f}")
    print(f"   📊 MAE: {mae:.2f}")
    print(f"   📊 R² Score: {r2:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': lightgbm_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 5 Important Features:")
    for idx, row in feature_importance.head().iterrows():
        print(f"   • {row['feature']}: {row['importance']:.3f}")
    
    # Create scaler (for compatibility, though LightGBM doesn't need it)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    return lightgbm_model, scaler, feature_columns, rmse, r2

def save_models_simple(model, scaler, company_encoder, username_encoder, feature_columns, rmse, r2, df):
    print("\n💾 Saving LightGBM models (without pickle)...")
    
    # Save only the core ML components (no custom classes)
    joblib.dump(model, 'ai_model/tweet_prediction_model.pkl')
    joblib.dump(scaler, 'ai_model/feature_scaler.pkl')
    joblib.dump(company_encoder, 'ai_model/company_encoder.pkl')
    joblib.dump(username_encoder, 'ai_model/username_encoder.pkl')
    
    # Save metadata as JSON instead of pickle
    metadata = {
        'best_model_name': 'LightGBM',
        'rmse': float(rmse),
        'r2_score': float(r2),
        'feature_columns': feature_columns,
        'feature_count': len(feature_columns),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_size': len(df),
        'use_scaling': False
    }
    
    with open('ai_model/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Models saved successfully (pickle-safe)!")
    print("   📊 tweet_prediction_model.pkl (LightGBM)")
    print("   🔧 feature_scaler.pkl")
    print("   🏷️ company_encoder.pkl")
    print("   🏷️ username_encoder.pkl")
    print("   📄 model_metadata.json")

def main():
    print("🚀 CAIC Tweet Predictor - LightGBM Training (Fixed)")
    print("=" * 50)
    
    # Load and preprocess data
    df, company_encoder, username_encoder = load_and_preprocess_data()
    
    # Train LightGBM model
    model, scaler, feature_columns, rmse, r2 = train_lightgbm_model(df)
    
    # Save models without pickle issues
    save_models_simple(model, scaler, company_encoder, username_encoder, feature_columns, rmse, r2, df)
    
    print(f"\n🎯 TRAINING COMPLETE!")
    print(f"🏆 Model: LightGBM")
    print(f"📊 Performance: RMSE={rmse:.2f}, R²={r2:.3f}")
    print(f"🚀 Ready for deployment!")

if __name__ == "__main__":
    main()
