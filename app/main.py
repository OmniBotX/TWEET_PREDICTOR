from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.schema import TweetPredictionRequest, TweetGenerationRequest, TweetPredictionResponse, TweetGenerationResponse
from app.services.predictor import TweetPredictor
from app.services.generator import TweetGenerator

app = FastAPI(
    title="Tweet Predictor - CAIC 2025 (LightGBM)",
    description="AI-powered tweet engagement prediction and content generation using LightGBM",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
predictor = TweetPredictor()
generator = TweetGenerator()

@app.on_event("startup")
async def startup_event():
    """Load LightGBM model on startup"""
    print("🚀 Starting Tweet Predictor API with LightGBM...")
    model_loaded = predictor.load_model()
    
    if model_loaded:
        print("✅ LightGBM model loaded successfully")
    else:
        print("❌ Warning: LightGBM model could not be loaded")

@app.get("/")
async def root():
    return {
        "message": "Tweet Predictor API - LightGBM Ready",
        "model": "LightGBM", 
        "tasks": ["Behavior Simulation", "Content Generation"]
    }

@app.get("/health")
async def health_check():
    model_status = "loaded" if predictor.model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model": "LightGBM",
        "model_status": model_status,
        "services": ["predictor", "generator"]
    }

# Task 1: Predict Likes (Behavior Simulation)
@app.post("/predict", response_model=TweetPredictionResponse)
async def predict_likes(request: TweetPredictionRequest):
    """Predict tweet engagement using LightGBM"""
    try:
        prediction = predictor.predict(
            content=request.content,
            username=request.username,
            company=request.company,
            media=request.media,
            hour=request.hour
        )
        return TweetPredictionResponse(**prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LightGBM prediction error: {str(e)}")

# Task 2: Generate Content (Content Simulation)
@app.post("/generate", response_model=TweetGenerationResponse)
async def generate_tweet(request: TweetGenerationRequest):
    """Generate tweet content"""
    try:
        generated = generator.generate_tweet(
            company=request.company,
            tweet_type=request.tweet_type,
            topic=request.topic,
            username=request.username,
            include_media=request.include_media
        )
        return TweetGenerationResponse(**generated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content generation error: {str(e)}")

# Combined endpoint
@app.post("/generate_and_predict")
async def generate_and_predict(request: TweetGenerationRequest):
    """Generate content and predict engagement in one call"""
    try:
        # Generate tweet
        generated = generator.generate_tweet(
            company=request.company,
            tweet_type=request.tweet_type,
            topic=request.topic,
            username=request.username,
            include_media=request.include_media
        )
        
        # Predict engagement
        prediction = predictor.predict(
            content=generated["generated_tweet"],
            username=request.username or "default_user",
            company=request.company,
            media=request.include_media,
            hour=12
        )
        
        return {
            "generated_tweet": generated["generated_tweet"],
            "tweet_type": generated["tweet_type"],
            "predicted_likes": prediction["predicted_likes"],
            "engagement_level": prediction["engagement_level"],
            "confidence": prediction["confidence"],
            "model": "LightGBM"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Combined operation error: {str(e)}")
