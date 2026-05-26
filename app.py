import streamlit as st
import torch
from transformers import pipeline

# Set clean, professional page configuration
st.set_page_config(page_title="Financial Decision Support System", layout="wide")

# ==========================================
# STEP 1: CACHED MODEL INITIALIZATION
# ==========================================
@st.cache_resource
def initialize_pipelines():
    """
    Optimized loader leveraging Streamlit caching mechanism 
    to prevent redundant VRAM/RAM re-allocation on web state reruns.
    """
    device = 0 if torch.cuda.is_available() else -1
    
    # Pipeline 1: Fine-tuned Sentiment Classifier (Core Engine)
    sentiment_pipe = pipeline(
        "text-classification",
        model="chloeleya/finbert-fine-tuned-sentiment-model",
        device=device
    )
    
    # Pipeline 2: Zero-shot Topic Router (Contextual Mapping Engine)
    topic_pipe = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )
    
    return sentiment_pipe, topic_pipe

# Execute deterministic loading
with st.spinner("Initializing Deep Learning Models (HF Hub Synchronizing)..."):
    sentiment_engine, topic_engine = initialize_pipelines()

# ==========================================
# STEP 2: BUSINESS ANCHORS & LEXICONS
# ==========================================
# Verified 20-class target space from twitter-financial-news-topic
TOPIC_LABELS = [
    "analyst rating upgrade downgrade or recommendation",
    "Federal Reserve or central bank monetary policy",
    "company news or product launch announcement",
    "corporate bonds treasury or debt market",
    "dividend payment or distribution announcement",
    "earnings report quarterly results or revenue",
    "energy sector oil gas or petroleum",
    "banking financial services or insurance",
    "currency exchange rate or forex",
    "general financial news or market opinion",
    "gold silver metals or raw materials",
    "initial public offering IPO or stock listing",
    "legal case lawsuit or financial regulation",
    "merger acquisition investment or deal",
    "macroeconomic data GDP inflation or interest rate",
    "stock market index performance or trading",
    "politics government policy or election",
    "executive hire fire or leadership change",
    "stock analysis investor opinion or commentary",
    "stock price movement rise fall or volatility"
]

# ==========================================
# STEP 3: USER INTERFACE & INPUT LAYER
# ==========================================
st.title("📊 AI-Driven Financial News Router & Decision Support System")
st.markdown("---")

st.subheader("Financial News Intelligence Input")
default_text = "Apple shares surged 5% in pre-market trading after the company reported quarterly revenue that significantly beat consensus estimates, driven by strong iPhone 17 demand."
news_input = st.text_area("Paste financial news wire, tweet, or regulatory disclosure below:", value=default_text, height=120)

# ==========================================
# STEP 4: INFERENCE PIPELINES EXECUTION
# ==========================================
if st.button("Generate Trading Intelligence Reference", type="primary"):
    if news_input.strip() == "":
        st.warning("Input buffer empty. Please provide financial text.")
    else:
        with st.spinner("Executing cross-pipeline inference matrix..."):
            
            # Execute Pipeline 1: Sentiment Analysis
            senti_out = sentiment_engine(news_input)[0]
            pred_sentiment = senti_out['label'].upper()  # POSITIVE, NEGATIVE, NEUTRAL
            senti_score = senti_out['score']
            
            # Execute Pipeline 2: Zero-shot Topic Mapping
            topic_out = topic_engine(news_input, candidate_labels=TOPIC_LABELS, truncation=True, max_length=512)
            top_topic = topic_out['labels'][0]
            topic_score = topic_out['scores'][0]
            
            # ==========================================
            # STEP 5: TRADING DECISION SUPPORT LOGIC
            # ==========================================
            # Synthesize technical pipeline outputs into actionable trading references
            if "POS" in pred_sentiment:
                action_signal = "🟢 BULLISH BIAS / LONG REFERENCE"
                action_color = "green"
                strategy_note = "High probability upward momentum. Suitable for long positioning or risk-on allocation depending on volume profile."
            elif "NEG" in pred_sentiment:
                action_signal = "🔴 BEARISH BIAS / SHORT REFERENCE"
                action_color = "red"
                strategy_note = "Downside risk imminent. Consider hedging existing long exposure or strategic short entry positioning."
            else:
                action_signal = "⚪ NEUTRAL BIAS / HOLD REFERENCE"
                action_color = "gray"
                strategy_note = "Market consensus aligned. Maintain current allocation. No tactical entry signals detected."

            # ==========================================
            # STEP 6: DATA VISUALIZATION & OUTPUT DELIVERABLES
            # ==========================================
            st.markdown("### 🎯 Real-Time Trading Intelligence Output")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="Inferred Core Topic", value=top_topic.split(" or ")[0].title())
                st.caption(f"Router Confidence: {topic_score:.2%}")
                
            with col2:
                st.metric(label="Fine-Tuned Market Sentiment", value=pred_sentiment)
                st.caption(f"Sentiment Confidence: {senti_score:.2%}")
                
            with col3:
                st.markdown(f"**Actionable Trading Bias:**\n### :{action_color}[{action_signal}]")
            
            st.markdown("---")
            st.subheader("💡 Quantitative Risk & Strategy Reference")
            st.info(f"**Strategic Guidance:** {strategy_note}\n\n*Disclaimer: This synthesized output is powered by a fine-tuned deep learning model on historical text and serves as a quantitative reference only. It does not constitute formal investment advice.*")
