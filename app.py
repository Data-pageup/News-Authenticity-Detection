import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import re

# ======================
# Page Configuration
# ======================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Color scheme */
    :root {
        --primary: #E63946;
        --secondary: #457B9D;
        --accent: #F1FAEE;
        --success: #06A77D;
        --warning: #F77F00;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #E63946 0%, #457B9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 20px 0;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .fake-card {
        background: linear-gradient(135deg, #E63946 0%, #C71F2F 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 6px 12px rgba(230, 57, 70, 0.3);
    }
    
    .real-card {
        background: linear-gradient(135deg, #06A77D 0%, #00BFA5 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 6px 12px rgba(6, 167, 125, 0.3);
    }
    
    .info-box {
        background: white;
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #457B9D;
        color: #1A1A2E;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        margin: 15px 0;
    }
    
    .prediction-box {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #457B9D 0%, #1D3557 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #457B9D;
        font-size: 1.05rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #E63946 0%, #457B9D 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 15px 40px;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ======================
# Sidebar
# ======================
st.sidebar.markdown("""
    <div style='text-align: center; padding: 30px 20px; background: rgba(255,255,255,0.1); border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; font-size: 1.8rem; margin: 0; font-weight: 700;'>
            📰 Fake News Detector
        </h1>
        <p style='color: #E0E0E0; font-size: 0.95rem; margin-top: 8px;'>
            AI-Powered Verification
        </p>
    </div>
""", unsafe_allow_html=True)

section = st.sidebar.radio(
    "🧭 Navigate",
    ["🏠 Home", "🔍 Live Detector", "📊 Model Performance", "📈 EDA & Insights"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
    <p style='margin: 0; font-size: 0.9rem;'>💡 <strong>How to use</strong></p>
    <p style='margin: 5px 0 0 0; font-size: 0.85rem; opacity: 0.9;'>
        Paste news text in Live Detector to check authenticity using AI models
    </p>
</div>
""", unsafe_allow_html=True)

# Simple text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Mock prediction function (replace with your actual model)
def predict_news(text, model_type):
    # This is a mock - replace with actual model prediction
    text_clean = preprocess_text(text)
    word_count = len(text_clean.split())
    
    # Mock logic (replace with actual model)
    if model_type == "Logistic Regression":
        # Simulate LR prediction
        fake_prob = 0.15 if word_count > 50 else 0.85
    elif model_type == "Random Forest":
        fake_prob = 0.12 if word_count > 50 else 0.88
    else:  # LSTM
        fake_prob = 0.08 if word_count > 50 else 0.92
    
    real_prob = 1 - fake_prob
    prediction = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = max(fake_prob, real_prob) * 100
    
    return prediction, confidence, fake_prob * 100, real_prob * 100

# ======================
# 🏠 Home Page
# ======================
if section == "🏠 Home":
    st.markdown("<h1 class='main-header'>Fake News Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-Powered News Verification using Machine Learning & Deep Learning</p>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Best Accuracy", "99.75%", "Random Forest")
    with col2:
        st.metric("🤖 Models Trained", "3", "ML + DL")
    with col3:
        st.metric("📊 Test Samples", "8,980", "Balanced")
    with col4:
        st.metric("⚡ Features", "5,000", "TF-IDF")
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class='info-box'>
            <h3 style='color: #E63946; margin-top: 0;'>🎯 Project Overview</h3>
            <p style='font-size: 1.05rem; line-height: 1.8;'>
                This project uses <strong>advanced machine learning and deep learning techniques</strong> to detect 
                fake news articles with near-perfect accuracy. The system analyzes text patterns, word frequencies, 
                and contextual information to distinguish between authentic and fabricated news.
            </p>
            <h4 style='color: #457B9D;'>🔬 Technology Stack</h4>
            <ul style='line-height: 2;'>
                <li><strong>Classical ML:</strong> Logistic Regression, Random Forest (99%+ accuracy)</li>
                <li><strong>Deep Learning:</strong> LSTM with embeddings (99.16% validation accuracy)</li>
                <li><strong>Feature Engineering:</strong> TF-IDF vectorization with n-grams</li>
                <li><strong>Text Processing:</strong> Advanced NLP preprocessing pipeline</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='fake-card'>
            <h4 style='margin-top: 0;'>⚠️ Why This Matters</h4>
            <p style='line-height: 1.8;'>
                Fake news spreads 6x faster than real news on social media. 
                Our AI system helps combat misinformation by providing instant, 
                accurate verification of news articles.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='real-card' style='margin-top: 20px;'>
            <h4 style='margin-top: 0;'>✅ Key Features</h4>
            <ul style='line-height: 2; margin: 10px 0;'>
                <li>Real-time news verification</li>
                <li>99.75% accuracy</li>
                <li>Multiple AI models</li>
                <li>Confidence scoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("<h2 style='color: #457B9D; font-size: 2rem; margin: 30px 0 20px 0;'>📊 Model Comparison</h2>", unsafe_allow_html=True)
    
    # Model comparison chart
    models = ['Logistic Regression', 'Random Forest', 'LSTM']
    accuracy = [99.03, 99.75, 99.16]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=accuracy,
            text=[f"{acc}%" for acc in accuracy],
            textposition='auto',
            marker=dict(
                color=['#457B9D', '#E63946', '#F77F00'],
                line=dict(color='white', width=2)
            )
        )
    ])
    
    fig.update_layout(
        title="Model Accuracy Comparison",
        xaxis_title="Model",
        yaxis_title="Accuracy (%)",
        height=400,
        yaxis=dict(range=[98, 100]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ======================
# 🔍 Live Detector
# ======================
elif section == "🔍 Live Detector":
    st.markdown("<h1 class='main-header'>Live News Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Paste any news article to check its authenticity</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Enter News Article")
        news_title = st.text_input("News Title (Optional)", placeholder="Enter headline here...")
        news_text = st.text_area(
            "News Content",
            height=300,
            placeholder="Paste the full news article text here...\n\nThe more text you provide, the more accurate the prediction will be."
        )
        
        model_choice = st.selectbox(
            "🤖 Select Model",
            ["Random Forest (Best)", "Logistic Regression", "LSTM (Deep Learning)"]
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            analyze_btn = st.button("🔍 Analyze News", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    with col2:
        st.markdown("### 💡 Tips for Best Results")
        st.markdown("""
        <div class='info-box'>
            <ul style='line-height: 2;'>
                <li>📰 Include the full article text</li>
                <li>✍️ Add the title for better accuracy</li>
                <li>📏 Longer text = better predictions</li>
                <li>🌐 Works with any language (English best)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if analyze_btn and news_text:
        full_text = f"{news_title} {news_text}" if news_title else news_text
        
        model_map = {
            "Random Forest (Best)": "Random Forest",
            "Logistic Regression": "Logistic Regression",
            "LSTM (Deep Learning)": "LSTM"
        }
        
        with st.spinner("🔍 Analyzing article..."):
            prediction, confidence, fake_prob, real_prob = predict_news(
                full_text, 
                model_map[model_choice]
            )
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == "FAKE":
                st.markdown(f"""
                <div class='fake-card' style='text-align: center; padding: 40px;'>
                    <h2 style='font-size: 3rem; margin: 0;'>⚠️ FAKE NEWS</h2>
                    <p style='font-size: 1.5rem; margin: 20px 0;'>Confidence: {confidence:.2f}%</p>
                    <p style='margin: 0;'>This article shows characteristics of fabricated content</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='real-card' style='text-align: center; padding: 40px;'>
                    <h2 style='font-size: 3rem; margin: 0;'>✅ REAL NEWS</h2>
                    <p style='font-size: 1.5rem; margin: 20px 0;'>Confidence: {confidence:.2f}%</p>
                    <p style='margin: 0;'>This article appears to be authentic</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=real_prob,
                title={'text': "Authenticity Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#06A77D" if prediction == "REAL" else "#E63946"},
                    'steps': [
                        {'range': [0, 33], 'color': "#ffcccc"},
                        {'range': [33, 66], 'color': "#ffffcc"},
                        {'range': [66, 100], 'color': "#ccffcc"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown
        st.markdown("### 📈 Probability Breakdown")
        
        prob_df = pd.DataFrame({
            'Class': ['REAL News', 'FAKE News'],
            'Probability': [real_prob, fake_prob]
        })
        
        fig = px.bar(
            prob_df,
            x='Class',
            y='Probability',
            text='Probability',
            color='Class',
            color_discrete_map={'REAL News': '#06A77D', 'FAKE News': '#E63946'}
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(
            showlegend=False,
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Text statistics
        st.markdown("### 📝 Text Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Word Count", len(full_text.split()))
        with col2:
            st.metric("Character Count", len(full_text))
        with col3:
            st.metric("Model Used", model_map[model_choice])
        with col4:
            st.metric("Processing Time", "0.15s")

# ======================
# 📊 Model Performance
# ======================
elif section == "📊 Model Performance":
    st.markdown("<h1 class='main-header'>Model Performance Analysis</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📈 Logistic Regression", "🌳 Random Forest", "🧠 LSTM"])
    
    with tab1:
        st.markdown("## 📈 Logistic Regression")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div class='info-box'>
                <h4>Model Details</h4>
                <ul style='line-height: 2;'>
                    <li><strong>Type:</strong> Linear Classifier</li>
                    <li><strong>Accuracy:</strong> 99.03%</li>
                    <li><strong>Features:</strong> TF-IDF (5000)</li>
                    <li><strong>N-grams:</strong> (1,2)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Classification Report")
            st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 10px;'>
            
            | Class | Precision | Recall | F1-Score | Support |
            |-------|-----------|--------|----------|---------|
            | <span style='color: #000;'>**FAKE (0)**</span> | <span style='color: #000;'>0.99</span> | <span style='color: #000;'>0.99</span> | <span style='color: #000;'>0.99</span> | <span style='color: #000;'>4,650</span> |
            | <span style='color: #000;'>**REAL (1)**</span> | <span style='color: #000;'>0.99</span> | <span style='color: #000;'>0.99</span> | <span style='color: #000;'>0.99</span> | <span style='color: #000;'>4,330</span> |
            | <span style='color: #000;'>**Accuracy**</span> | | | <span style='color: #000;'>**0.99**</span> | <span style='color: #000;'>**8,980**</span> |
            
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("## 🌳 Random Forest Classifier")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div class='info-box'>
                <h4>Model Details</h4>
                <ul style='line-height: 2;'>
                    <li><strong>Type:</strong> Ensemble Method</li>
                    <li><strong>Accuracy:</strong> 99.75% 🏆</li>
                    <li><strong>Features:</strong> TF-IDF (5000)</li>
                    <li><strong>N-grams:</strong> (1,2)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Classification Report")
            st.markdown("""
            <div style='background: white; padding: 20px; border-radius: 10px; color: #000;'>
            
            | Class | Precision | Recall | F1-Score | Support |
            |-------|-----------|--------|----------|---------|
            | **FAKE (0)** | 1.00 | 1.00 | 1.00 | 4,650 |
            | **REAL (1)** | 1.00 | 1.00 | 1.00 | 4,330 |
            | **Accuracy** | | | **1.00** | **8,980** |
            
            </div>
            """, unsafe_allow_html=True)
        
        st.success("🏆 **Best Performing Model** - Random Forest achieves near-perfect accuracy!")
    
    with tab3:
        st.markdown("## 🧠 LSTM (Deep Learning)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div class='info-box'>
                <h4>Model Architecture</h4>
                <ul style='line-height: 2;'>
                    <li><strong>Type:</strong> Recurrent Neural Network</li>
                    <li><strong>Val Accuracy:</strong> 99.16%</li>
                    <li><strong>Epochs:</strong> 4</li>
                    <li><strong>Batch Size:</strong> 64</li>
                    <li><strong>Layers:</strong> Embedding → LSTM → Dense</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Training history
            epochs = [1, 2, 3, 4]
            train_acc = [90.86, 98.61, 98.86, 99.00]
            val_acc = [98.58, 98.50, 98.78, 99.16]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=train_acc, name='Train Accuracy',
                                    mode='lines+markers', line=dict(color='#457B9D', width=3)))
            fig.add_trace(go.Scatter(x=epochs, y=val_acc, name='Val Accuracy',
                                    mode='lines+markers', line=dict(color='#E63946', width=3)))
            
            fig.update_layout(
                title="LSTM Training History",
                xaxis_title="Epoch",
                yaxis_title="Accuracy (%)",
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.info("✅ **LSTM Advantage**: Captures sequence and context patterns that TF-IDF cannot detect")

# ======================
# 📈 EDA & Insights
# ======================
elif section == "📈 EDA & Insights":
    st.markdown("<h1 class='main-header'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "☁️ Word Clouds", "🔍 Key Insights"])
    
    with tab1:
        st.markdown("## 📊 Dataset Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", "44,898", "Training + Test")
        with col2:
            st.metric("Fake News", "4,650", "51.8%")
        with col3:
            st.metric("Real News", "4,330", "48.2%")
        
        st.markdown("### 📊 Class Distribution")
        
        # Class distribution chart
        labels = ['Fake News', 'Real News']
        values = [4650, 4330]
        colors = ['#E63946', '#06A77D']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.4,
            textinfo='label+percent+value',
            textfont_size=14
        )])
        
        fig.update_layout(
            title="Dataset Balance (Test Set)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class='info-box'>
            <h4>✅ Dataset Quality</h4>
            <ul style='line-height: 2;'>
                <li><strong>Balance:</strong> Nearly balanced dataset (51.8% vs 48.2%)</li>
                <li><strong>Size:</strong> Sufficient samples for robust training</li>
                <li><strong>Features:</strong> Title and full text content</li>
                <li><strong>Preprocessing:</strong> Cleaned, lowercased, stopwords removed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("## ☁️ Word Clouds Analysis")
        
        st.markdown("""
        <div class='info-box'>
            <h4>📝 Visual Text Analysis</h4>
            <p style='line-height: 1.8;'>
                Word clouds reveal the most frequent terms in fake vs real news articles, 
                helping identify linguistic patterns and common themes.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚠️ Fake News Word Cloud")
            try:
                fake_wc = Image.open("fake_wordcloud.png")
                st.image(fake_wc, use_container_width=True)
            except:
                st.warning("📁 Upload 'fake_wordcloud.png' to display")
        
        with col2:
            st.markdown("### ✅ Real News Word Cloud")
            try:
                real_wc = Image.open("real_wordcloud.png")
                st.image(real_wc, use_container_width=True)
            except:
                st.warning("📁 Upload 'real_wordcloud.png' to display")
        
        st.markdown("""
        <div class='info-box'>
            <h4>🔍 Key Observations</h4>
            <ul style='line-height: 2;'>
                <li><strong>Fake News:</strong> Often contains sensational, emotional, or clickbait language</li>
                <li><strong>Real News:</strong> Uses more neutral, factual, and professional terminology</li>
                <li><strong>Patterns:</strong> Distinct vocabulary differences help models classify accurately</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("## 🔍 Key Insights & Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class='info-box'>
                <h4>🎯 Model Performance</h4>
                <ul style='line-height: 2;'>
                    <li><strong>Random Forest:</strong> Best performer (99.75%)</li>
                    <li><strong>LSTM:</strong> Best at capturing context (99.16%)</li>
                    <li><strong>Logistic Regression:</strong> Fast and effective (99.03%)</li>
                    <li><strong>All models:</strong> Achieve >99% accuracy</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='info-box'>
                <h4>🔬 Feature Engineering</h4>
                <ul style='line-height: 2;'>
                    <li><strong>TF-IDF:</strong> Captures word importance effectively</li>
                    <li><strong>Bigrams:</strong> Improves context understanding</li>
                    <li><strong>Title + Text:</strong> Combined features boost accuracy</li>
                    <li><strong>Max Features:</strong> 5000 optimal for performance</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='fake-card'>
                <h4 style='margin-top: 0;'>⚠️ Fake News Characteristics</h4>
                <ul style='line-height: 2; margin: 10px 0;'>
                    <li>Sensational headlines</li>
                    <li>Emotional language</li>
                    <li>Lack of citations</li>
                    <li>Grammatical errors</li>
                    <li>Clickbait patterns</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class='real-card' style='margin-top: 20px;'>
                <h4 style='margin-top: 0;'>✅ Real News Characteristics</h4>
                <ul style='line-height: 2; margin: 10px 0;'>
                    <li>Factual reporting</li>
                    <li>Neutral tone</li>
                    <li>Proper citations</li>
                    <li>Professional writing</li>
                    <li>Credible sources</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div class='info-box'>
            <h4>Future Improvements</h4>
            <ul style='line-height: 2;'>
                <li><strong>Transformers:</strong> Implement BERT/RoBERTa for better accuracy</li>
                <li><strong>Multi-language:</strong> Extend support to non-English news</li>
                <li><strong>Source Checking:</strong> Verify publisher credibility</li>
                <li><strong>Real-time API:</strong> Integration with news aggregators</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 30px 0;'>
    <h3 style='color: #E63946; margin: 0;'>Fake News Detection System</h3>
    <p style='color: #666; font-size: 0.95rem; margin: 10px 0;'>Built with Streamlit • Powered by Machine Learning & Deep Learning</p>
    <p style='color: #999; font-size: 0.85rem; margin: 5px 0;'>Models: Logistic Regression • Random Forest • LSTM</p>
    <p style='color: #457B9D; font-size: 1.1rem; margin: 20px 0; font-weight: 700;'>Built by Amirtha Ganesh R</p>
</div>
""", unsafe_allow_html=True)