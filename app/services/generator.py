import random
import pickle
import os

class TweetGenerator:
    def __init__(self):
        self.generator = None
        self._load_generator()
        
        # Fallback templates
        self.templates = {
            'announcement': [
                "🚀 Exciting news from {company}! {topic} is here! {media_ref}",
                "Big announcement: {company} is launching {topic}! 🎉 {media_ref}",
                "📢 {company} proudly presents {topic}! Don't miss out {media_ref}"
            ],
            'question': [
                "What do you think about {topic}? {company} wants to hear from you! 💬 {media_ref}",
                "Quick question from {company}: How do you feel about {topic}? 🤔 {media_ref}"
            ],
            'promotional': [
                "Check out {company}'s latest {topic}! Limited time offer 🌟 {media_ref}",
                "Don't miss {company}'s exclusive {topic}! Get yours today 💯 {media_ref}"
            ],
            'engagement': [
                "Tag someone who needs to see {company}'s {topic}! 👆 {media_ref}",
                "Double tap if you love {company}'s {topic}! ❤️ {media_ref}"
            ]
        }
    
    def _load_generator(self):
        """Load generator from model package"""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            package_path = os.path.join(base_dir, "ai_model", "complete_tweet_predictor.pkl")
            
            if os.path.exists(package_path):
                with open(package_path, 'rb') as f:
                    package = pickle.load(f)
                self.generator = package.get('tweet_generator')
                print("✅ Loaded tweet generator from LightGBM package")
        except Exception as e:
            print(f"⚠️ Could not load generator: {e}")
    
    def generate_tweet(self, company, tweet_type="promotional", topic="new product", 
                      username=None, include_media=False):
        """Generate tweet using loaded generator or templates"""
        
        # Try using loaded generator first
        if self.generator:
            try:
                return self.generator.generate_tweet(company, tweet_type, topic, username, include_media)
            except:
                pass
        
        # Fallback to template generation
        templates = self.templates.get(tweet_type, self.templates['promotional'])
        template = random.choice(templates)
        
        media_ref = ""
        if include_media:
            media_refs = ["Check out the photo!", "See image below 📸", "Watch the video!"]
            media_ref = random.choice(media_refs)
        
        tweet = template.format(company=company.title(), topic=topic, media_ref=media_ref).strip()
        
        if len(tweet) > 280:
            tweet = tweet[:277] + "..."
        
        return {
            'generated_tweet': tweet,
            'tweet_type': tweet_type,
            'word_count': len(tweet.split()),
            'character_count': len(tweet),
            'includes_media_ref': include_media
        }
