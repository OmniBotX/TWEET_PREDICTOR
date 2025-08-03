import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Tweet Predictor"
    debug: bool = True
    data_path: str = "ai_model/dataset.csv"
    model_path: str = "ai_model/model.pkl"
    
    class Config:
        env_file = ".env"

settings = Settings()
