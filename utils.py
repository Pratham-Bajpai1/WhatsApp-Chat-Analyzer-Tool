import pandas as pd
from typing import Set

def filter_user(df: pd.DataFrame, selected_user: str) -> pd.DataFrame:
    """
    Filter DataFrame for the selected user or return full DataFrame if 'Overall'.
    """
    if selected_user == "Overall":
        return df
    return df[df['user'] == selected_user]

def is_media_message(message: str, media_tokens: Set[str]) -> bool:
    """
    Check if a message is a media message (generalized, supports multilingual).
    """
    if not isinstance(message, str):
        return False
    return message.lower().strip() in {token.lower() for token in media_tokens}

def is_group_notification(user: str) -> bool:
    """
    Check if a user is a group/system notification.
    """
    return user == "group_notification"