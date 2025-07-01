import pandas as pd
from typing import Tuple

def get_busy_users(df: pd.DataFrame, top_n: int = 5) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Returns the top_n busiest users and their contribution percentage.
    Args:
        df: Preprocessed chat DataFrame.
        top_n: How many top users to return.
    Returns:
        Tuple of (top_users_counts, percent_df)
    """
    user_counts = df['user'].value_counts().head(top_n)
    percent_df = (df['user'].value_counts(normalize=True) * 100).round(2).reset_index()
    percent_df.columns = ['user', 'percent']
    return user_counts, percent_df