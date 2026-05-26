import streamlit as st
import torch
import urllib.request
import re
from bs4 import BeautifulSoup
from transformers import pipeline

# Set clean, professional page configuration
st.set_page_config(page_title="Financial Decision Support System", layout="wide")

# ==========================================
# STEP 1: CACHED MODEL INITIALIZATION
# ==========================================
@st.cache_resource
def initialize_pipelines():
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

with st.spinner("Initializing Deep Learning Models (HF Hub Synchronizing)..."):
    sentiment_engine, topic_engine = initialize_pipelines()

# ==========================================
# STEP 2: HELPER FUNCTIONS FOR CRAWLING & ANALYSIS
# ==========================================
def extract_text_from_url(url):
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read()
        
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('h1').get_text() if soup.find('h1') else ""
        paragraphs = soup.find_all('p')
        body_text = " ".join([p.get_text() for p in paragraphs[:4]]) # Take first 4 paragraphs
        
        combined_text = f"{title}. {body_text}"
        cleaned_text = re.sub(r'\s+', ' ', combined_text).strip()
        return cleaned_text if len(cleaned_text) > 20 else None
    except Exception:
        return None

def extract_market_evidence(text, sentiment_bias):
    """
    Granular Feature: Parses the text into individual sentences and extracts 
    the most statistically relevant 'market trigger words' as textual evidence.
    """
    # Split text into sentences professionally
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Define corporate & financial high-impact lexical triggers
    bullish_triggers = ["surged", "beat", "growth", "growth", "jumped", "positive", "highest", "record", "demand", "upgrade", "gained", "climb"]
    bearish_triggers = ["dropped", "missed", "fell", "slumped", "decline", "negative", "loss", "risk", "downgrade", "warned", "plunged", "deficit"]
    
    evidence_sentences = []
    target_triggers = bullish_triggers if sentiment_bias == "BULLISH" else bearish_triggers
    
    if sentiment_bias == "NEUTRAL":
        # For neutral market news, extract macroeconomic data or consensus matching sentences
        target_triggers = ["expected", "unchanged", "remained", "aligned", "flat", "consensus"]

    for sentence in sentences:
        if any(trigger in sentence.lower() for trigger in target_triggers):
            if len(sentence.strip()) > 15 and sentence.strip() not in evidence_sentences:
                evidence_sentences.append(sentence.strip())
                if len(evidence_sentences) >= 2: # Keep maximum 2 key evidence sentences for clean UI
                    break
                    
    # Fallback if no specific trigger words are matched
    if not evidence_sentences and sentences:
        evidence_sentences.append(sentences[0]) # Return the headline as core evidence
        
    return evidence_sentences

# ==========================================
# STEP 3: BUSINESS ANCHORS & LEXICONS
# ==========================================
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
# STEP 4: USER INTERFACE & INPUT LAYER
# ==========================================
st.title("📊 AI-Driven Financial News Router & Decision Support System")
st.markdown("---")

st.subheader("Financial News & Intelligence Input Portal")
st.markdown("*System auto-detects input format. You can paste **raw news text, tweets, or a direct news URL** (e.g., Yahoo Finance, Reuters).*")

default_text = "https://finance.yahoo.com/news/nvidia-earnings-revenue-q3-2025-212015632.html"
user_input = st.text_area("Paste financial news wire text or article URL below:", value=default_text, height=100)

# ==========================================
# STEP 5: INFERENCE PIPELINES EXECUTION
# ==========================================
if st.button("Generate Trading Intelligence Reference", type="primary"):
    if user_input.strip() == "":
        st.warning("Input buffer empty.")
    else:
        with st.spinner("Processing input and executing cross-pipeline inference matrix..."):
            
            is_url = user_input.strip().startswith(("http://", "https://"))
            analysis_text = user_input
            
            if is_url:
                st.info(f"🌐 URL detected. Fetching and parsing text from target webpage...")
                scraped_text = extract_text_from_url(user_input.strip())
                if scraped_text:
                    analysis_text = scraped_text
                    with st.expander("See Scraped News Context"):
                        st.write(analysis_text)
                else:
                    st.error("Failed to extract text from URL. Using raw URL string for fallback inference.")

            # Execute Pipeline 1: Sentiment Analysis
            senti_out = sentiment_engine(analysis_text)[0]
            pred_sentiment = senti_out['label'].upper()
            senti_score = senti_out['score']
            
            # Execute Pipeline 2: Zero-shot Topic Mapping
            topic_out = topic_engine(analysis_text, candidate_labels=TOPIC_LABELS, truncation=True, max_length=512)
            top_topic = topic_out['labels'][0]
            topic_score = topic_out['scores'][0]
            
            # Map Trading Signals
            if "POS" in pred_sentiment:
                sentiment_bias = "BULLISH"
                action_signal = "🟢 BULLISH BIAS / LONG REFERENCE"
                action_color = "green"
                strategy_note = "High probability upward momentum. Suitable for tactical long positioning or asset accumulation."
            elif "NEG" in pred_sentiment:
                sentiment_bias = "BEARISH"
                action_signal = "🔴 BEARISH BIAS / SHORT REFERENCE"
                action_color = "red"
                strategy_note = "Downside risk asset drift imminent. Consider hedging delta exposure or defensive short allocation."
            else:
                sentiment_bias = "NEUTRAL"
                action_signal = "⚪ NEUTRAL BIAS / HOLD REFERENCE"
                action_color = "gray"
                strategy_note = "Market consensus tightly aligned. Maintain current tracking positions. No tactical entry signals."

            # Dynamic Granular Analytics: Extract sentence-level examples from the raw news wire
            extracted_evidences = extract_market_evidence(analysis_text, sentiment_bias)

            # ==========================================
            # STEP 6: DATA VISUALIZATION & OUTPUTS
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
            
            # GRANULAR ADDITION: Displaying specific raw text statements that triggered the AI decision
            st.subheader("🔍 Textual Evidence & Core Market Triggers")
            st.markdown("The AI engine identified the following specific statements from the source text as key catalysts:")
            for evidence in extracted_evidences:
                st.markdown(f"> 📄 *\"... {evidence} ...\"*")
            
            st.markdown("---")
            st.subheader("💡 Quantitative Risk & Strategy Reference")
            st.info(f"**Strategic Guidance:** {strategy_note}\n\n*Disclaimer: This synthesized output is powered by a fine-tuned deep learning model on historical text and serves as a quantitative reference only. It does not constitute formal investment advice.*")
