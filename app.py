import streamlit as st
import torch
import urllib.request
import re
from bs4 import BeautifulSoup
from transformers import pipeline

# =====================================================================
# STEP 1: WEBSITE CONFIGURATION (MUST BE THE FIRST STREAMLIT COMMAND)
# =====================================================================
st.set_page_config(
    page_title="Institutional Trading Decision Support System", 
    page_icon="📈", 
    layout="wide"
)

# =====================================================================
# STEP 2: FIXED BUSINESS ANCHORS & LEXICONS (GLOBAL SCOPE)
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

BULLISH_TRIGGERS = ["surged", "beat", "growth", "jumped", "positive", "highest", "record", "demand", "upgrade", "gained", "climb", "bullish", "rose", "soared", "soar", "high", "above"]
BEARISH_TRIGGERS = ["dropped", "missed", "fell", "slumped", "decline", "negative", "loss", "risk", "downgrade", "warned", "plunged", "deficit", "bearish", "constraint", "slowdown", "down"]

# ==========================================
# STEP 3: CACHED MODEL INITIALIZATION MATRIX
# ==========================================
@st.cache_resource
def initialize_pipelines():
    device = 0 if torch.cuda.is_available() else -1
    
    sentiment_pipe = pipeline(
        "text-classification",
        model="chloeleya/finbert-fine-tuned-sentiment-model",
        device=device
    )
    
    topic_pipe = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )
    
    return sentiment_pipe, topic_pipe

with st.spinner("Synchronizing Institutional Model Matrices from HF Hub..."):
    sentiment_engine, topic_engine = initialize_pipelines()

# =====================================================================
# STEP 4: HELPER UTILITIES FOR PARSING & DUAL-AUDITING
# =====================================================================
def extract_text_from_url(url):
    try:
        # Create a premium spoofed browser environment to bypass anti-scraping walls
        opener = urllib.request.build_opener()
        opener.addheaders = [
            ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
            ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'),
            ('Accept-Language', 'en-US,en;q=0.5'),
            ('Connection', 'keep-alive'),
            ('Cache-Control', 'max-age=0')
        ]
        
        with opener.open(url, timeout=10) as response:
            html = response.read()
        
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('h1').get_text().strip() if soup.find('h1') else ""
        
        # Pull text from paragraph elements
        paragraphs = soup.find_all('p')
        body_text_list = []
        for p in paragraphs:
            text_block = p.get_text().strip()
            # Explicitly exclude common dynamic error notifications and cookie text strings
            if len(text_block) > 25 and not any(x in text_block.lower() for x in ["something went wrong", "button links", "cookie policy", "browser settings"]):
                body_text_list.append(text_block)
                
        body_text = " ".join(body_text_list[:5])
        return title, body_text
    except Exception:
        return None, None

def extract_granular_evidence(text, primary_bias):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    primary_evidence = []
    opposing_evidence = []
    
    if primary_bias == "BULLISH":
        primary_tokens, opposing_tokens = BULLISH_TRIGGERS, BEARISH_TRIGGERS
    elif primary_bias == "BEARISH":
        primary_tokens, opposing_tokens = BEARISH_TRIGGERS, BULLISH_TRIGGERS
    else:
        primary_tokens, opposing_tokens = ["expected", "unchanged", "remained", "flat", "consensus"], []

    for sentence in sentences:
        sentence_str = sentence.strip()
        # Filter boilerplate navigation frames out of tracking lists
        if len(sentence_str) < 30 or any(x in sentence_str.lower() for x in ["yahoo finance", "sign in", "click here", "oops"]):
            continue
            
        if any(t in sentence_str.lower() for t in primary_tokens) and len(primary_evidence) < 2:
            if sentence_str not in primary_evidence:
                primary_evidence.append(sentence_str)
                
        if opposing_tokens and any(t in sentence_str.lower() for t in opposing_tokens) and len(opposing_evidence) < 1:
            if sentence_str not in primary_evidence and sentence_str not in opposing_evidence:
                opposing_evidence.append(sentence_str)
                
    # Fallback system if token matching yield is dry
    if not primary_evidence:
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 40 and not any(x in s_clean.lower() for x in ["yahoo finance", "oops", "something went wrong"]):
                primary_evidence.append(s_clean)
                if len(primary_evidence) >= 1:
                    break
        if not primary_evidence and sentences:
            primary_evidence.append(sentences[0])
        
    return primary_evidence, opposing_evidence

# =====================================================================
# STEP 5: USER INTERFACE LAYER (INPUT & INTERACTION)
# =====================================================================
st.title("📊 AI-Driven Financial News Router & Decision Support System")
st.markdown("---")

st.subheader("Financial News & Intelligence Input Portal")
st.markdown("*System auto-detects input format. Supports raw news copy, market tweets, or live news URLs (e.g., Yahoo Finance).*")

default_text = "https://finance.yahoo.com/markets/stocks/articles/dell-stock-soared-over-130-091719505.html"
user_input = st.text_area("Input Terminal Gateway (Text/URL):", value=default_text, height=90)

run_analysis = st.button("Generate Trading Intelligence Reference", type="primary")

# =====================================================================
# STEP 6: QUANTITATIVE INFERENCE & RENDERING SEQUENCE
# =====================================================================
if run_analysis:
    if user_input.strip() == "":
        st.warning("Input buffer empty. Please provide a valid URL or text wire.")
    else:
        with st.spinner("Executing real-time multi-pipeline analytical sequence..."):
            
            is_url = user_input.strip().startswith(("http://", "https://"))
            raw_analysis_text = user_input
            news_title = ""
            
            if is_url:
                st.info("🌐 URL Gateway active. Parsing text from remote node...")
                scraped_title, scraped_body = extract_text_from_url(user_input.strip())
                if scraped_title or scraped_body:
                    news_title = scraped_title
                    raw_analysis_text = f"{scraped_title}. {scraped_body}"
                    with st.expander("See Scraped News Context"):
                        st.write(raw_analysis_text)
                else:
                    st.error("URL Extraction restricted by target firewall. Advancing via raw string fallback mapping.")

            # =========================================================
            # 🔥 DATA CLEANING & TITLE BOOSTING MATRIX
            # =========================================================
            clean_analysis_text = re.sub(r'\d+(,\d+)*', '[NUM]', raw_analysis_text)
            clean_title = re.sub(r'\d+(,\d+)*', '[NUM]', news_title) if news_title else ""
            
            if is_url and clean_title:
                title_out = sentiment_engine(clean_title)[0]
                full_out = sentiment_engine(clean_analysis_text)[0]
                
                t_label = title_out['label'].upper().strip()
                f_label = full_out['label'].upper().strip()
                
                if ("POS" in t_label or t_label == "POSITIVE") and title_out['score'] > 0.80:
                    pred_sentiment = "POSITIVE"
                    senti_score = title_out['score']
                else:
                    if "POS" in f_label or f_label == "POSITIVE":
                        pred_sentiment = "POSITIVE"
                    elif "NEG" in f_label or f_label == "NEGATIVE":
                        pred_sentiment = "NEGATIVE"
                    else:
                        pred_sentiment = "NEUTRAL"
                    senti_score = full_out['score']
            else:
                senti_out = sentiment_engine(clean_analysis_text)[0]
                raw_label = senti_out['label'].upper().strip()
                pred_sentiment = "POSITIVE" if "POS" in raw_label else ("NEGATIVE" if "NEG" in raw_label else "NEUTRAL")
                senti_score = senti_out['score']
            
            # Pipeline 2 Inference: Zero-shot Topic Space Router
            topic_out = topic_engine(clean_analysis_text, candidate_labels=TOPIC_LABELS, truncation=True, max_length=512)
            top_topic = topic_out['labels'][0]
            topic_score = topic_out['scores'][0]
            
            # Execute Business Decision Translation Logic
            if pred_sentiment == "POSITIVE":
                sentiment_bias = "BULLISH"
                action_signal = "🟢 BULLISH BIAS / LONG REFERENCE"
                action_color = "green"
                strategy_note = "High probability upward momentum. Favorable window for programmatic asset accumulation or call placement."
            elif pred_sentiment == "NEGATIVE":
                sentiment_bias = "BEARISH"
                action_signal = "🔴 BEARISH BIAS / SHORT REFERENCE"
                action_color = "red"
                strategy_note = "Downside risk asset drift active. Tactical hedging overlays or short equity allocation advised."
            else:
                sentiment_bias = "NEUTRAL"
                action_signal = "⚪ NEUTRAL BIAS / HOLD REFERENCE"
                action_color = "gray"
                strategy_note = "Consensus balanced. Volatility compressed. Asset pricing normalized; alpha entry signals absent."

            primary_catalysts, hidden_risks = extract_granular_evidence(raw_analysis_text, sentiment_bias)

            # =====================================================================
            # STEP 7: ADVANCED OUTPUT VISUALIZATION
            # =====================================================================
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
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("🔍 Core Supporting Market Triggers")
                st.markdown("Specific statements driving the primary AI market sentiment output:")
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
