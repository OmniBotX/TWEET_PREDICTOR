from pydantic import BaseModel, Field
from typing import Optional, Literal

# Request Models
class TweetPredictionRequest(BaseModel):
    content: str = Field(..., description="Tweet content text")
    username: str = Field(..., description="Username posting the tweet")
    company: str = Field(..., description="Company/brand name")
    media: bool = Field(default=False, description="Whether tweet includes media")
    hour: Optional[int] = Field(default=12, description="Hour of posting (0-23)")

class TweetGenerationRequest(BaseModel):
    company: str = Field(..., description="Company/brand name")
    tweet_type: Literal["announcement", "question", "promotional", "engagement"] = Field(
        default="promotional", description="Type of tweet to generate"
    )
    topic: str = Field(..., description="Topic or theme for the tweet")
    username: Optional[str] = Field(None, description="Target username style")
    include_media: bool = Field(default=False, description="Whether to include media reference")

# Response Models
class TweetPredictionResponse(BaseModel):
    predicted_likes: int
    engagement_level: str  # Low, Medium, High
    confidence: float

class TweetGenerationResponse(BaseModel):
    generated_tweet: str
    tweet_type: str
    word_count: int
    character_count: int
