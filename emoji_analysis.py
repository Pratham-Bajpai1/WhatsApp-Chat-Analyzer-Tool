import pandas as pd
from typing import Dict
from collections import Counter
import emoji

def extract_emojis(text: str) -> list:
    """
    Extract all emojis from a given text.
    Args:
        text: The input string.
    Returns:
        List of emojis found in the string.
    """
    return [char for char in text if emoji.is_emoji(char)]

def emoji_stats(df: pd.DataFrame, selected_user: str = "Overall") -> pd.DataFrame:
    """
    Returns a DataFrame with emoji usage and frequencies for the chat or specific user.
    Args:
        df: Preprocessed chat DataFrame.
        selected_user: User to filter by, or "Overall".
    Returns:
        DataFrame with columns ['emoji', 'count']
    """
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    all_emojis = []
    for msg in df['message']:
        all_emojis.extend(extract_emojis(str(msg)))
    emoji_counts = Counter(all_emojis)
    emoji_df = pd.DataFrame(emoji_counts.items(), columns=['emoji', 'count']).sort_values('count', ascending=False).reset_index(drop=True)
    return emoji_df