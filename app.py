import streamlit as st
import torch
import urllib.request
import re
from bs4 import BeautifulSoup
from transformers import pipeline

# ==========================================
# STEP 1: WEBSITE FAVICON & PAGE CONFIG
# ==========================================
# Setting professional financial trending icon and expansive layout
st.set_page_config(
    page_title="Institutional Trading Decision Support System", 
    page_icon="📈", 
    layout="wide"
)

# ==========================================
# STEP 2: CACHED MODEL INITIALIZATION
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

with st.spinner("Synchronizing Institutional Model Matrices from HF Hub..."):
    sentiment_engine, topic_engine = initialize_pipelines()

# ==========================================
# STEP 3: HIGH-PRECISION LEXICAL ENGINES
# ==========================================
BULLISH_TRIGGERS = ["surged", "beat", "growth", "jumped", "positive", "highest", "record", "demand", "upgrade", "gained", "climb", "bullish"]
BEARISH_TRIGGERS = ["dropped", "missed", "fell", "slumped", "decline", "negative", "loss", "risk", "downgrade", "warned", "plunged", "deficit", "bearish", "constraint", "slowdown"]

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
        body_text = " ".join([p.get_text() for p in paragraphs[:4]])
        
        combined_text = f"{title}. {body_text}"
        cleaned_text = re.sub(r'\s+', ' ', combined_text).strip()
        return cleaned_text if len(cleaned_text) > 20 else None
    except Exception:
        return None

def extract_granular_evidence(text, primary_bias):
    """
    Advanced Risk Feature: Splits text into sentences to extract both supporting evidence 
    FOR the primary bias, and isolating OPPOSING forces to act as a micro-risk reference.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    primary_evidence = []
    opposing_evidence = []
    
    # Identify target tokens based on signal
    if primary_bias == "BULLISH":
        primary_tokens, opposing_tokens = BULLISH_TRIGGERS, BEARISH_TRIGGERS
    elif primary_bias == "BEARISH":
        primary_tokens, opposing_tokens = BEARISH_TRIGGERS, BULLISH_TRIGGERS
    else:
        primary_tokens, opposing_tokens = ["expected", "unchanged", "remained", "flat", "consensus"], []

    for sentence in sentences:
        sentence_str = sentence.strip()
        if len(sentence_str) < 20:
            continue
            
        # Extract supporting evidence
        if any(t in sentence_str.lower() for t in primary_tokens) and len(primary_evidence) < 2:
            if sentence_str not in primary_evidence:
                primary_evidence.append(sentence_str)
                
        # Extract counter-risk statements (Opposing Force)
        if opposing_tokens and any(t in sentence_str.lower() for t in opposing_tokens) and len(opposing_evidence) < 1:
            if sentence_str not in primary_evidence and sentence_str not in opposing_evidence:
                opposing_evidence.append(sentence_str)
                
    if not primary_evidence and sentences:
        primary_evidence.append(sentences[0])
        
    return primary_evidence, opposing_evidence

# ==========================================
# STEP 4: USER INTERFACE & INPUT LAYER
# ==========================================
st.title("📊 AI-Driven Financial News Router & Decision Support System")
st.markdown("---")

st.subheader("Financial News & Intelligence Input Portal")
st.markdown("*System auto-detects input format. Supports raw news copy, market tweets, or live news URLs (e.g., Yahoo Finance).*")

default_text = "https://finance.yahoo.com/news/nvidia-earnings-revenue-q3-2025-212015632.html"
user_input = st.text_area("Input Terminal Gateway (Text/URL):", value=default_text, height=90)

# ==========================================
# STEP 5: QUANTITATIVE INFERENCE LAYER
# ==========================================
if st.button("Generate Trading Intelligence Reference", type="primary"):
    if user_input.strip() == "":
        st.warning("Input buffer empty.")
    else:
        with st.spinner("Executing real-time multi-pipeline analytical sequence..."):
            
            is_url = user_input.strip().startswith(("http://", "https://"))
            analysis_text = user_input
            
            if is_url:
                st.info(f"🌐 URL Gateway active. Parsing text from remote node...")
                scraped_text = extract_text_from_url(user_input.strip())
                if scraped_text:
                    analysis_text = scraped_text
                    with st.expander("See Scraped News Context"):
                        st.write(analysis_text)
                else:
                    st.error("URL Extraction restricted by firewall. Advancing via raw string fallback mapping.")

            # Pipeline 1 Inference (Fine-tuned Engine)
            senti_out = sentiment_engine(analysis_text)[0]
            pred_sentiment = senti_out['label'].upper()
            senti_score = senti_out['score']
            
            # Pipeline 2 Inference (Zero-shot Router)
            TOPIC_LABELS = [
                "analyst rating upgrade downgrade or recommendation", "Federal Reserve or central bank monetary policy",
                "company news or product launch announcement", "corporate bonds treasury or debt market",
                "dividend payment or distribution announcement", "earnings report quarterly results or revenue",
                "energy sector oil gas or petroleum", "banking financial services or insurance",
                "currency exchange rate or forex", "general financial news or market opinion",
                "gold silver metals or raw materials", "initial public offering IPO or stock listing",
                "legal case lawsuit or financial regulation", "merger acquisition investment or deal",
                "macroeconomic data GDP inflation or interest rate", "stock market index performance or trading",
                "politics government policy or election", "executive hire fire or leadership change",
                "stock analysis investor opinion or commentary", "stock price movement rise fall or volatility"
            ]
            topic_out = topic_engine(analysis_text, candidate_labels=TOPIC_LABELS, truncation=True, max_length=512)
            top_topic = topic_out['labels'][0]
            topic_score = topic_out['scores'][0]
            
            # Execute Trading Decision Synthesis
            if "POS" in pred_sentiment:
                sentiment_bias = "BULLISH"
                action_signal = "🟢 BULLISH BIAS / LONG REFERENCE"
                action_color = "green"
                strategy_note = "High probability upward momentum. Favorable window for programmatic asset accumulation."
            elif "NEG" in pred_sentiment:
                sentiment_bias = "BEARISH"
                action_signal = "🔴 BEARISH BIAS / SHORT REFERENCE"
                action_color = "red"
                strategy_note = "Downside risk asset drift active. Tactical hedging or protective overlay execution advised."
            else:
                sentiment_bias = "NEUTRAL"
                action_signal = "⚪ NEUTRAL BIAS / HOLD REFERENCE"
                action_color = "gray"
                strategy_note = "Consensus balanced. Asset pricing normalized. No active tactical entry alpha detected."

            # Extract Primary Catalysts AND Counter-Sentiment Risks
            primary_catalysts, hidden_risks = extract_granular_evidence(analysis_text, sentiment_bias)

            # ==========================================
            # STEP 6: ADVANCED DATA VISUALIZATION
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
            
            # UI GRID FOR GRANULAR INTELLIGENCE
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("🔍 Core Supporting Market Triggers")
                st.markdown("Statements driving the primary AI market sentiment output:")
                for catalyst in primary_catalysts:
                    st.markdown(f"> ✅ *\"... {catalyst} ...\"*")
            
            with col_right:
                st.subheader("⚠️ Dual-Force Risk Audit")
                if hidden_risks:
                    st.markdown("Warning: The system isolated the following **opposing risk factors** within the wire context:")
                    for risk in hidden_risks:
                        st.markdown(f"> ❌ *\"... {risk} ...\"*")
                else:
                    st.success("No meaningful counter-sentiment lexical anomalies or opposing structural risk statements detected.")
            
            st.markdown("---")
            st.subheader("💡 Quantitative Risk & Strategy Reference")
            st.info(f"**Strategic Guidance:** {strategy_note}\n\n*Disclaimer: This synthesized output is powered by a fine-tuned deep learning model on historical text and serves as a quantitative reference only. It does not constitute formal investment advice.*")
