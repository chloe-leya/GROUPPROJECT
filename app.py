import streamlit as st
import torch
import urllib.request
import re
from bs4 import BeautifulSoup
from transformers import pipeline

# =====================================================================
# STEP 1: UI ARCHITECTURE & WORKSPACE SETUP
# =====================================================================
st.set_page_config(
    page_title="Institutional Trading Decision Support System", 
    page_icon="📈", 
    layout="wide"
)

# =====================================================================
# STEP 2: QUANTITATIVE TAXONOMIES & REVENUE ANCHORS
# =====================================================================
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

BULLISH_TRIGGERS = ["surged", "beat", "growth", "jumped", "positive", "highest", "record", "demand", "upgrade", "gained", "climb", "bullish", "rose", "soared", "soar", "strong", "rally"]
BEARISH_TRIGGERS = ["dropped", "missed", "fell", "slumped", "decline", "negative", "loss", "risk", "downgrade", "warned", "plunged", "deficit", "bearish", "constraint", "slowdown", "down", "weak", "disruption", "hamstrung", "shortage", "pressure", "bottleneck", "blockade"]

STRONG_BULLISH_KEYWORDS = ["high earnings", "all-time high profit", "above expectations", "surged higher", "strong demand", "revenue growth", "upswing", "frenzy", "rally"]
STRONG_BEARISH_KEYWORDS = ["plunged", "plunge", "dropped", "slumped", "crashed", "missed", "downgraded", "bearish", "fall", "decline", "weak", "hamstrung", "disruption"]

GARBAGE_PATTERNS = [
    "something went wrong", "cookie policy", "browser settings", "broker-dealer", 
    "investment adviser", "does not offer securities", "button links to", 
    "facilitate trading", "all rights reserved", "terms of service", "privacy policy", 
    "yahoo finance is not", "discover more", "further reading", "before you go", 
    "scmp poll", "min read", "read full article", "sign in to", "sharing tools",
    "t&cs and copyright", "breach offt.com", "licensing@ft.com", "subscribers may share"
]

# =====================================================================
# STEP 3: ASYNCHRONOUS PIPELINE LIFECYCLE
# =====================================================================
@st.cache_resource
def initialize_pipelines():
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipe = pipeline(
        "text-classification", 
        model="chloeleya/finbert-fine-tuned-sentiment-model", 
        device=device,
        top_k=None
    )
    topic_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    return sentiment_pipe, topic_pipe

with st.spinner("Synchronizing Institutional Model Matrices from HF Hub..."):
    sentiment_engine, topic_engine = initialize_pipelines()

# =====================================================================
# STEP 4: EXTRACTION ENGINE & TELEMETRY CLEANING PIPELINES
# =====================================================================
def advanced_text_cleaner(text):
    if not text:
        return ""
    # Strip out live stock ticker percentages (+5.43%) to prevent feature pollution
    text = re.sub(r'[-+]\d+(?:\.\d+)?\s*\(\s*[-+]\d+(?:\.\d+)?%\s*\)', '', text)
    # Clear index metadata scrapers
    text = re.sub(r'RT Quote\s*\|\s*Exchange\s*\|\s*USD', '', text, flags=re.IGNORECASE)
    # Wipe out raw ticker data tables
    text = re.sub(r'\b\d{1,3}(?:,\d{3})+\.(?:\d+)\b', '', text)
    return text

def extract_text_from_url(url):
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [
            ('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
            ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'),
            ('Accept-Language', 'en-US,en;q=0.5')
        ]
        with opener.open(url, timeout=10) as response:
            html = response.read()
        
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('h1').get_text().strip() if soup.find('h1') else ""
        
        paragraphs = soup.find_all('p')
        body_text_list = []
        for p in paragraphs:
            text_block = p.get_text().strip()
            if len(text_block) > 35 and not any(g in text_block.lower() for g in GARBAGE_PATTERNS):
                body_text_list.append(text_block)
                
        return title, " ".join(body_text_list[:4])
    except Exception:
        return None, None

def extract_fallback_title(url):
    slug = url.split('/')[-1].replace('.html', '')
    words = re.split(r'[-_]', slug)
    return " ".join([w for w in words if not w.isdigit() and len(w) > 2]).title()

def extract_granular_evidence(text, primary_bias):
    """
    Extracts structural market triggers and tail-risk signals based on primary sentiment.
    Implements macro black-swan detection and balanced neutral mapping.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)[:40] 
    primary_evidence, opposing_evidence = [], []
    
    # Institutional Macro and Micro Tail-Risk Glossaries
    macro_risk_words = ["war", "conflict", "clashed", "strikes", "supply shock", "shuttered", "disruption", "shortage", "blockade", "sanctions", "stalemate"]
    macro_turnaround_words = ["truce", "peace efforts", "ceasefire", "reopening", "negotiations", "agreement"]

    for sentence in sentences:
        s_clean = sentence.strip()
        if len(s_clean) < 25 or any(g in s_clean.lower() for g in GARBAGE_PATTERNS):
            continue
        
        s_lower = s_clean.lower()
        
        # --- PATH A: BULLISH PRIMARY SENTIMENT ---
        if primary_bias == "BULLISH":
            if any(t in s_lower for t in BULLISH_TRIGGERS) and len(primary_evidence) < 2:
                if s_clean not in primary_evidence:
                    primary_evidence.append(s_clean)
            if (any(r in s_lower for r in macro_risk_words) or any(t in s_lower for t in BEARISH_TRIGGERS)) and len(opposing_evidence) < 1:
                if s_clean not in primary_evidence:
                    opposing_evidence.append(s_clean)

        # --- PATH B: BEARISH PRIMARY SENTIMENT ---
        elif primary_bias == "BEARISH":
            if any(t in s_lower for t in BEARISH_TRIGGERS) and len(primary_evidence) < 2:
                if s_clean not in primary_evidence:
                    primary_evidence.append(s_clean)
            if (any(r in s_lower for r in macro_turnaround_words) or any(t in s_lower for t in BULLISH_TRIGGERS)) and len(opposing_evidence) < 1:
                if s_clean not in primary_evidence:
                    opposing_evidence.append(s_clean)

        # --- PATH C: NEUTRAL / COMPRESSED VOLATILITY SENTIMENT ---
        else:
            has_bull_token = any(t in s_lower for t in BULLISH_TRIGGERS)
            has_bear_token = any(t in s_lower for t in BEARISH_TRIGGERS)
            
            if len(primary_evidence) == 0 and (has_bull_token or has_bear_token):
                primary_evidence.append(s_clean)
            elif len(primary_evidence) == 1:
                first_is_bull = any(t in primary_evidence[0].lower() for t in BULLISH_TRIGGERS)
                if first_is_bull and has_bear_token:
                    primary_evidence.append(s_clean)
                elif not first_is_bull and has_bull_token:
                    primary_evidence.append(s_clean)
            
            if any(r in s_lower for r in macro_risk_words) and len(opposing_evidence) < 1:
                if s_clean not in primary_evidence:
                    opposing_evidence.append(s_clean)

    # Robust Fallback Vector Processing
    if not primary_evidence and sentences:
        for s in sentences:
            if len(s.strip()) > 35 and not any(g in s.lower() for g in GARBAGE_PATTERNS):
                primary_evidence.append(s.strip())
                if len(primary_evidence) == 2: 
                    break

    return primary_evidence[:2], opposing_evidence[:1]

# =====================================================================
# STEP 5: INTERACTION GATEWAY & OPERATIONAL INSTRUCTIONS
# =====================================================================
st.title("📊 AI-Driven Financial News Router & Decision Support System")
st.markdown("---")

st.subheader("Financial Intelligence Input Gateway")

st.info("💡 **Operational Guidance:** Copy and paste raw macroeconomic or equity transcripts below. The architecture utilizes localized telemetry filtering to decouple market noise and ticker artifacts prior to neural inference.")

placeholder_msg = "Paste regulatory wires, corporate text copies, or market tweets here..."
default_context = "NVIDIA Reinforces Bullish Outlook as AI Accelerators Demand Surges. Data center revenue remained the key growth engine, powered by demand for Nvidia's GPUs and AI accelerators. Companies developing large language models (LLMs), generative AI applications, and advanced computing systems continue to rely heavily on Nvidia hardware."

user_input = st.text_area("Input Terminal Gateway (Text / Live URL):", value=default_context, placeholder=placeholder_msg, height=120)
run_analysis = st.button("Execute Quantitative Analysis Chain", type="primary")

# =====================================================================
# STEP 6: RISK-MITIGATED INFERENCE OVERLAYS (RESOLVED MATRIX LOGIC)
# =====================================================================
if run_analysis:
    if not user_input.strip():
        st.warning("Input operational buffer is currently empty.")
    else:
        with st.spinner("Executing real-time multi-pipeline analytical sequence..."):
            is_url = user_input.strip().startswith(("http://", "https://"))
            raw_analysis_text = user_input
            news_title = ""
            
            if is_url:
                scraped_title, scraped_body = extract_text_from_url(user_input.strip())
                if scraped_title:
                    news_title = scraped_title
                    raw_analysis_text = f"{scraped_title}. {scraped_body}"
                    with st.expander("Scraped Telemetry Payload"):
                        st.write(raw_analysis_text)
                else:
                    news_title = extract_fallback_title(user_input.strip())
                    raw_analysis_text = news_title
                    st.warning(f"⚠️ Remote Firewall Interception. Fallback parser extracted vector intent: '{news_title}'")
            else:
                lines = raw_analysis_text.split('\n')
                sanitized_lines = [l.strip() for l in lines if l.strip() and not any(g in l.lower() for g in GARBAGE_PATTERNS)]
                raw_analysis_text = " ".join(sanitized_lines)

            # Scrub incoming dataset
            raw_analysis_text = advanced_text
