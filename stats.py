import pandas as pd
from typing import Tuple, List, Set, Dict
from urlextract import URLExtract

def fetch_stats(
    df: pd.DataFrame,
    selected_user: str = "Overall",
    media_tokens: Set[str] = None
) -> Dict[str, int]:
    """
    Calculate total messages, words, media files, and links for the dataset or a user.
    Args:
        df: Preprocessed chat DataFrame.
        selected_user: User to filter by, or "Overall".
        media_tokens: Set of strings that indicate a media message (multilingual support).
    Returns:
        Dictionary with keys: total_messages, total_words, total_media_files, total_links
    """
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    # Total messages
    total_messages = df.shape[0]

    # Total words
    total_words = sum(len(str(msg).split()) for msg in df['message'])

    # Media files (generalized)
    media_tokens = media_tokens or {"<Media omitted>", "image omitted", "video omitted", "audio omitted"}
    media_mask = df['message'].str.lower().str.strip().isin(
        {token.lower() for token in media_tokens}
    )
    total_media_files = media_mask.sum()

    # Links
    extractor = URLExtract()
    total_links = sum(len(extractor.find_urls(str(msg))) for msg in df['message'])

    return dict(
        total_messages=total_messages,
        total_words=total_words,
        total_media_files=total_media_files,
        total_links=total_links
    )