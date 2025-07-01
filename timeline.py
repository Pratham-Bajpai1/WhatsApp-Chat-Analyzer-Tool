import pandas as pd
from typing import Literal, Optional

def timeline(
    df: pd.DataFrame,
    selected_user: str = "Overall",
    freq: Literal['D', 'ME', 'W'] = 'ME'
) -> pd.DataFrame:
    """
    Returns a timeline DataFrame (counts per period).
    Args:
        df: Preprocessed chat DataFrame.
        selected_user: User to filter by, or "Overall".
        freq: Frequency; 'D' = daily, 'ME' = month end, 'W' = weekly.
    Returns:
        DataFrame with columns ['period', 'message_count']
    """
    if selected_user != "Overall":
        df = df[df['user'] == selected_user]
    period_col = 'period'
    grouped = df.set_index('date').resample(freq)['message'].count().reset_index()
    grouped.rename(columns={'date': period_col, 'message': 'message_count'}, inplace=True)
    return grouped