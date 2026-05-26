import streamlit as st
import torch
import urllib.request
import re
from bs4 import BeautifulSoup
from transformers import pipeline

# =====================================================================
# STEP 1: WEBSITE CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND)
# =====================================================================
# Setting high-density layout and an institutional trending chart favicon
st.set_page_config(
    page_title="Institutional Trading Decision Support System", 
    page_icon="📈", 
    layout="wide"
)

# =====================================================================
# STEP 2: CACHED MODEL INITIALIZATION MATRIX
# =====================================================================
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

# Show explicit deployment progress to users
with st.spinner("Synchronizing Institutional Model Matrices from HF Hub..."):
    sentiment_engine, topic_engine = initialize_pipelines()

# =====================================================================
# STEP 3: HIGH-PRECISION LEXICAL & SCRAPING ENGINES
# =====================================================================
BULLISH_TRIGGERS = ["surged", "beat", "growth", "jumped", "positive", "highest", "record", "demand", "upgrade", "gained", "climb", "bullish"]
BEARISH_TRIGGERS = ["dropped", "missed", "fell", "slumped", "decline", "negative", "loss", "risk", "downgrade", "warned", "plunged", "deficit", "bearish", "constraint", "slowdown"]

def extract_text_from_url(url):
    """
    Safely scrapes and extracts headlines/paragraphs from financial news wires 
    to bypass manual copy-pasting and handle firewalls natively.
    """
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
        body_text = " ".join([p.get_text() for p in paragraphs[:4]]) # Scrape top 4 relevant paragraphs
        
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
    
    # Map token spaces based on model primary inferences
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
            
        # Extract evidence supporting the primary direction
        if any(t in sentence_str.lower() for t in primary_tokens) and len(primary_evidence) < 2:
            if sentence_str not in primary_evidence:
                primary_evidence.append(sentence_str)
                
        # Extract counter-sentiment statements (Hidden Risks / Bearish Cues in a Bullish article)
        if opposing_tokens and any(t in sentence_str.lower() for t in opposing_tokens) and len(opposing_evidence) < 1:
            if sentence_str not in primary_evidence and sentence_str not in opposing_evidence:
                opposing_evidence.append(sentence_str)
                
    # Fallback to headline if sentence extraction yields empty matrices
    if not primary_evidence and sentences:
        primary_evidence.append(sentences[0])
        
    return primary_evidence, opposing_evidence

# =====================================================================
# STEP 4: USER INTERFACE & INPUT GATEWAY LAYER
# =====================================================================
st.title("📊 AI-Driven Financial News Router & Decision Support System")
st.markdown("---")

st.subheader("Financial News & Intelligence Input Portal")
st.markdown("*System auto-detects input format. Supports raw news copy, market tweets, or live news URLs (e.g., Yahoo Finance).*")

# Keep the working Yahoo Finance URL as the premium default benchmark example
default_text = "
