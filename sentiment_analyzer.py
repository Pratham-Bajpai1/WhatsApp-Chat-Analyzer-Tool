import pandas as pd
from typing import List, Dict, Optional, Any
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# ---- Sentiment Analysis ----
try:
    import nltk
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    import nltk
    nltk.download('vader_lexicon')

def get_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

def analyze_sentiment(df: pd.DataFrame, text_col: str = "message") -> pd.DataFrame:
    """
    Adds sentiment scores and label (pos/neg/neu) for each message.
    """
    sia = get_sentiment_analyzer()
    sentiments = df[text_col].astype(str).apply(sia.polarity_scores)
    sent_df = pd.DataFrame(list(sentiments))
    df = df.copy()
    df["sent_neg"] = sent_df["neg"]
    df["sent_neu"] = sent_df["neu"]
    df["sent_pos"] = sent_df["pos"]
    df["sent_compound"] = sent_df["compound"]
    df["sentiment"] = df["sent_compound"].apply(lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral"))
    return df

# ---- Emotion Detection with Hugging Face ----
def get_emotion_pipeline(model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
    try:
        from transformers import pipeline
        # Cache model on first load
        if not hasattr(get_emotion_pipeline, "_pipe"):
            get_emotion_pipeline._pipe = pipeline("text-classification", model=model_name, top_k=None, truncation=True)
        return get_emotion_pipeline._pipe
    except Exception as e:
        print("Could not load transformers emotion model:", str(e))
        return None

def analyze_emotion(
    df: pd.DataFrame,
    text_col: str = "message",
    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
) -> pd.DataFrame:
    """
    Adds 'emotion' column to DataFrame using a Hugging Face emotion model.
    Falls back to rule-based if model is not available.
    """
    pipe = get_emotion_pipeline(model_name)
    df = df.copy()
    if pipe:
        # Batch processing for efficiency
        messages = df[text_col].fillna("").astype(str).tolist()
        # The pipeline expects a list of texts
        try:
            results = pipe(messages, truncation=True)
            # Pick the highest score label for each message
            detected = []
            for out in results:
                if isinstance(out, list):
                    best = max(out, key=lambda x: x['score'])
                    label = best['label'].lower()
                elif isinstance(out, dict):
                    label = out.get('label', 'other').lower()
                else:
                    label = "other"
                detected.append(label)
            df["emotion"] = detected
        except Exception as e:
            print("Emotion pipeline failed, falling back to rule-based:", str(e))
            df["emotion"] = df[text_col].apply(detect_emotion)
    else:
        df["emotion"] = df[text_col].apply(detect_emotion)
    return df

# ---- Rule-based fallback ----
EMOTION_MAP = {
    "joy": [r"\bhappy\b", r"\bjoy\b", r"\bglad\b", r"\b😊|😁|😃|😄|🙂\b"],
    "sadness": [r"\bsad\b", r"\bunhappy\b", r"\b😭|😢|😔|😞\b"],
    "anger": [r"\bangry\b", r"\bmad\b", r"\b😡|😠\b"],
    "fear": [r"\bscared\b", r"\bfear\b", r"\banxious\b", r"\b😨|😱\b"],
    "surprise": [r"\bsurprise\b", r"\bshocked\b", r"\b😲|😮|😯\b"],
    "love": [r"\blove\b", r"\b❤️|😍|😘|💖\b"],
}
def detect_emotion(text: str, emotion_map: Optional[Dict[str, List[str]]] = None) -> str:
    emotion_map = emotion_map or EMOTION_MAP
    text = str(text).lower()
    for emotion, patterns in emotion_map.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return emotion
    return "other"