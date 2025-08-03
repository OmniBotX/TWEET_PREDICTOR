import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="TweetIQ - LightGBM Powered",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main { font-family: 'Inter', sans-serif; }
    
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .tweet-preview {
        background: linear-gradient(135deg, #1DA1F2 0%, #0d8bd9 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(29, 161, 242, 0.3);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    
    .lightgbm-badge {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8000"

def check_api_connection():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def predict_tweet_engagement(content, username, company, media, hour):
    """Call LightGBM prediction API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "content": content,
                "username": username,
                "company": company,
                "media": media,
                "hour": hour
            }
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Prediction API error: {e}")
        return None

def generate_tweet_content(company, tweet_type, topic, username, include_media):
    """Call content generation API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={
                "company": company,
                "tweet_type": tweet_type,
                "topic": topic,
                "username": username,
                "include_media": include_media
            }
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Generation API error: {e}")
        return None

def generate_and_predict(company, tweet_type, topic, username, include_media):
    """Call combined API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate_and_predict",
            json={
                "company": company,
                "tweet_type": tweet_type,
                "topic": topic,
                "username": username,
                "include_media": include_media
            }
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Combined API error: {e}")
        return None

def main():
    # Hero Section
    st.markdown("""
    <div class="hero-header">
        <div class="lightgbm-badge">⚡ Powered by LightGBM</div>
        <div class="hero-title">TweetIQ</div>
        <div class="hero-subtitle">AI-Powered Content Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API connection
    api_connected, api_info = check_api_connection()
    
    if not api_connected:
        st.error("⚠️ **Backend API is not running!** Please start the FastAPI server.")
        st.info("💡 **Run:** `uvicorn app.main:app --reload` from your project directory")
        return
    else:
        model_info = api_info.get('model', 'Unknown') if api_info else 'Unknown'
        st.success(f"✅ API Connected! Model: **{model_info}**")
    
    # Navigation
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Engagement Predictor", key="nav_predict", help="Predict tweet performance with LightGBM"):
            st.session_state.current_page = "predict"
    
    with col2:
        if st.button("✨ Content Generator", key="nav_generate", help="Generate engaging content"):
            st.session_state.current_page = "generate"
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home"
    
    # Page routing
    if st.session_state.current_page == "predict":
        engagement_predictor_page()
    elif st.session_state.current_page == "generate":
        content_generator_page()
    else:
        homepage()

def homepage():
    """Professional landing page"""
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Combined Intelligence Demo
    st.markdown("## 🚀 Complete Content Intelligence Workflow")
    st.markdown("*Generate content and predict its performance in one seamless flow*")
    
    with st.form("intelligence_workflow"):
        col1, col2 = st.columns(2)
        
        with col1:
            company = st.text_input("🏢 Company Name", placeholder="Nike, Apple, Starbucks...")
            tweet_type = st.selectbox("📱 Content Type", ["promotional", "announcement", "question", "engagement"])
        
        with col2:
            topic = st.text_input("🎯 Topic/Theme", placeholder="product launch, sale, innovation...")
            include_media = st.checkbox("🖼️ Include Media Reference")
        
        username = st.text_input("👤 Username (optional)", placeholder="brand_official")
        
        workflow_btn = st.form_submit_button("🤖 Generate & Predict with LightGBM", use_container_width=True)
    
    if workflow_btn and company and topic:
        with st.spinner("🤖 LightGBM is analyzing and generating..."):
            result = generate_and_predict(company, tweet_type, topic, username, include_media)
            
            if result:
                st.success("✅ **LightGBM Analysis Complete!**")
                
                # Display results in two columns
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.markdown("### 📱 Generated Content")
                    st.markdown(f'<div class="tweet-preview">{result["generated_tweet"]}</div>', 
                               unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 📊 LightGBM Prediction")
                    
                    # Metrics
                    st.metric("💙 Predicted Likes", f"{result['predicted_likes']:,}")
                    st.metric("📈 Engagement", result['engagement_level'])
                    st.metric("🎯 Confidence", f"{result['confidence']:.1%}")
                
                # Confidence visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = result['confidence'] * 100,
                    title = {'text': "LightGBM Confidence"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#28a745"},
                        'steps': [
                            {'range': [0, 60], 'color': "lightgray"},
                            {'range': [60, 80], 'color': "yellow"}, 
                            {'range': [80, 100], 'color': "green"}
                        ],
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("❌ Failed to process request. Please try again.")
    
    # Feature showcase
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <span style="font-size: 2.5rem;">🎯</span>
            <h3>LightGBM Prediction Engine</h3>
            <p>Advanced gradient boosting model trained on tweet engagement data. 
            Delivers fast, accurate predictions with superior performance.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <span style="font-size: 2.5rem;">✨</span>
            <h3>Smart Content Generation</h3>
            <p>AI-powered tweet creation tailored to your brand voice. 
            Generate engaging content optimized for maximum interaction.</p>
        </div>
        """, unsafe_allow_html=True)

def engagement_predictor_page():
    """LightGBM engagement prediction interface"""
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("## 🎯 LightGBM Engagement Predictor")
        st.markdown("*Forecast tweet performance with gradient boosting AI*")
    with col2:
        if st.button("← Back", key="back_from_predict"):
            st.session_state.current_page = "home"
    
    with st.form("prediction_form"):
        content = st.text_area(
            "💬 Tweet Content",
            height=120,
            placeholder="Enter your tweet content here...",
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("👤 Account Handle", placeholder="your_brand")
            company = st.text_input("🏢 Company Name", placeholder="Your Company")
        
        with col2:
            media = st.checkbox("🖼️ Visual Content")
            hour = st.selectbox("⏰ Post Time", options=list(range(24)), index=12, format_func=lambda x: f"{x:02d}:00")
        
        predict_btn = st.form_submit_button("🤖 Analyze with LightGBM", use_container_width=True)
    
    if predict_btn and content and username and company:
        with st.spinner("🤖 LightGBM analyzing your content..."):
            prediction = predict_tweet_engagement(content, username, company, media, hour)
            
            if prediction:
                st.success("✅ **LightGBM Analysis Complete!**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <span style="font-size: 2rem; font-weight: 700;">{prediction['predicted_likes']:,}</span>
                        <div style="color: #6c757d; margin-top: 0.5rem;">💙 Predicted Likes</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    engagement_colors = {"Low": "🔴", "Medium": "🟡", "High": "🟢"}
                    color = engagement_colors.get(prediction['engagement_level'], '⚪')
                    st.markdown(f"""
                    <div class="metric-container">
                        <span style="font-size: 2rem;">{color}</span>
                        <div style="color: #6c757d; margin-top: 0.5rem;">📈 {prediction['engagement_level']} Engagement</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <span style="font-size: 2rem; font-weight: 700;">{prediction['confidence']:.0%}</span>
                        <div style="color: #6c757d; margin-top: 0.5rem;">🎯 Model Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
            else:
                st.error("❌ Prediction failed. Please check your inputs.")

def content_generator_page():
    """Content generation interface"""
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("## ✨ AI Content Generator")
        st.markdown("*Create compelling social media content*")
    with col2:
        if st.button("← Back", key="back_from_generate"):
            st.session_state.current_page = "home"
    
    with st.form("generation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            company = st.text_input("🏢 Brand Name", placeholder="Your Company")
            tweet_type = st.selectbox(
                "📱 Content Category",
                options=["promotional", "announcement", "question", "engagement"],
                format_func=lambda x: {
                    "promotional": "🎯 Promotional",
                    "announcement": "📢 Announcement", 
                    "question": "❓ Question",
                    "engagement": "💬 Engagement"
                }[x]
            )
        
        with col2:
            topic = st.text_input("🎯 Campaign Theme", placeholder="product launch, sale...")
            username = st.text_input("👤 Voice Style", placeholder="professional, casual...")
        
        include_media = st.checkbox("🖼️ Include Visual References")
        
        generate_btn = st.form_submit_button("✨ Generate Content", use_container_width=True)
    
    if generate_btn and company and topic:
        with st.spinner("🤖 AI crafting your content..."):
            generated = generate_tweet_content(company, tweet_type, topic, username, include_media)
            
            if generated:
                st.success("✅ **Content Created Successfully!**")
                
                # Display generated content
                st.markdown(f"""
                <div class="tweet-preview">
                    <div style="opacity: 0.8; margin-bottom: 0.5rem;">@{username or company.lower().replace(' ', '_')}</div>
                    <div style="font-size: 1.1rem; line-height: 1.5;">{generated["generated_tweet"]}</div>
                    <div style="opacity: 0.8; margin-top: 1rem;">🕒 Generated with AI</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Content metrics
                col1, col2, col3, col4 = st.columns(4)
                
                metrics = [
                    ("📝", "Words", generated['word_count']),
                    ("📊", "Characters", f"{generated['character_count']}/280"),
                    ("📱", "Type", generated['tweet_type'].title()),
                    ("🖼️", "Media", "Yes" if generated.get('includes_media_ref') else "No")
                ]
                
                for i, (icon, label, value) in enumerate(metrics):
                    with [col1, col2, col3, col4][i]:
                        st.markdown(f"""
                        <div class="metric-container">
                            <span style="font-size: 1.5rem;">{icon}</span>
                            <div style="font-weight: 600; margin: 0.5rem 0;">{value}</div>
                            <div style="color: #6c757d; font-size: 0.9rem;">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
            else:
                st.error("❌ Content generation failed. Please try again.")

if __name__ == "__main__":
    main()
