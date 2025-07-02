import os

import pandas as pd
from typing import List, Dict, Optional, Any, Union
import requests
import streamlit as st

# ---- Sentiment Analysis (VADER for speed, still quite robust for EN/HI) ----
def get_sentiment_analyzer():
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

def analyze_sentiment(
    df: pd.DataFrame,
    text_col: str = "message",
    out_col_prefix: str = "sent_"
) -> pd.DataFrame:
    """
    Analyze sentiment using VADER on uncleaned/original message text.
    """
    sia = get_sentiment_analyzer()
    sentiments = df[text_col].astype(str).apply(sia.polarity_scores)
    sent_df = pd.DataFrame(list(sentiments))
    df = df.copy()
    df[f"{out_col_prefix}neg"] = sent_df["neg"]
    df[f"{out_col_prefix}neu"] = sent_df["neu"]
    df[f"{out_col_prefix}pos"] = sent_df["pos"]
    df[f"{out_col_prefix}compound"] = sent_df["compound"]
    df["sentiment"] = df[f"{out_col_prefix}compound"].apply(
        lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
    )
    return df

# ---- Multilingual Emotion Detection using Hugging Face Inference API ----
def _get_hf_token() -> Optional[str]:
    return (
        st.secrets.get("HF_TOKEN", None)
        if hasattr(st, "secrets")
        else os.environ.get("HF_TOKEN")
    ) or os.environ.get("HF_TOKEN")

@st.cache_resource(show_spinner="Loading emotion model (local if possible, else API)...")
def load_local_emotion_pipeline() -> Optional[Any]:
    try:
        import torch  # noqa: F401
        from transformers import pipeline
        pipe = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
            truncation=True,
        )
        return pipe
    except Exception as e:
        st.warning(
            f"Local emotion model unavailable: {e}. Will use Hugging Face Inference API fallback if possible."
        )
        return None

def detect_emotion_local(texts: List[str], pipe: Any) -> List[str]:
    """
    Run emotion detection locally using the Hugging Face pipeline.
    Returns a list of dominant emotion labels (one per text).
    Always returns a list of the same length as texts.
    """
    if pipe is None:
        return []
    try:
        results = pipe(texts, truncation=True)
        labels = []
        for res in results:
            if isinstance(res, list) and res:
                top = max(res, key=lambda x: x['score'])
                labels.append(top['label'].lower())
            elif isinstance(res, dict) and 'label' in res:
                labels.append(res['label'].lower())
            else:
                labels.append("neutral")
        # Defensive: if output is wrong length, treat as failure
        if len(labels) != len(texts):
            st.warning("Local emotion model returned unexpected output length. Falling back to API.")
            return []
        return labels
    except Exception as e:
        st.warning(f"Error running local emotion model: {e}. Falling back to API.")
        return []

def detect_emotion_api(texts: List[str], hf_token: Optional[str] = None) -> List[str]:
    """
    Use Hugging Face Inference API via huggingface_hub.InferenceClient.
    Returns a list of dominant emotion labels (one per text).
    Always returns a list of the same length as texts.
    """
    labels = []
    try:
        from huggingface_hub import InferenceClient
        hf_token = hf_token or _get_hf_token()
        if not hf_token:
            st.warning("HF_TOKEN not found for Hugging Face API. Emotion detection will default to 'neutral'.")
            return ["neutral"] * len(texts)
        client = InferenceClient(token=hf_token)
        for text in texts:
            try:
                result = client.text_classification(
                    text,
                    model="SamLowe/roberta-base-go_emotions"
                )
                if isinstance(result, list) and result:
                    top = max(result, key=lambda x: x['score'])
                    labels.append(top['label'].lower())
                else:
                    labels.append("neutral")
            except Exception as e:
                st.warning(f"Error querying Hugging Face API: {e}")
                labels.append("neutral")
        st.info("Using Hugging Face Inference API for emotion detection.")
        return labels
    except ImportError:
        st.warning("huggingface_hub not installed. Install or add to requirements.txt for remote API fallback.")
        return ["neutral"] * len(texts)
    except Exception as e:
        st.warning(f"Error setting up Hugging Face Inference API: {e}")
        return ["neutral"] * len(texts)

def detect_emotion(
    texts: Union[str, List[str]],
    use_api: bool = True,
    pipe: Optional[Any] = None
) -> List[str]:
    """
    Detects emotion for each message in texts (original, uncleaned).
    Tries local model first, then API fallback, then "neutral" default.
    Always returns a list of the same length as input texts.
    """
    if isinstance(texts, str):
        texts = [texts]
    # Stage 1: Try local model
    if pipe is None:
        pipe = load_local_emotion_pipeline()
    labels = detect_emotion_local(texts, pipe)
    if labels and len(labels) == len(texts):
        return labels
    # Stage 2: Fallback to API
    if use_api:
        labels = detect_emotion_api(texts)
        if labels and len(labels) == len(texts):
            return labels
    # Final fallback: all neutral
    st.warning("Could not detect emotion (model and API unavailable). Defaulting to 'neutral'.")
    return ["neutral"] * len(texts)

def analyze_emotion(
    df: pd.DataFrame,
    text_col: str = "message",
    use_api: bool = True
) -> pd.DataFrame:
    """
    Adds an 'emotion' column using local model or API fallback.
    Uses original, uncleaned messages.
    Returns a copy of the DataFrame with 'emotion' column.
    """
    df = df.copy()
    messages = df[text_col].fillna("").astype(str).tolist()
    labels = detect_emotion(messages, use_api=use_api)
    df["emotion"] = labels
    return df