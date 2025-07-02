import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Union, List, Dict, Any
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_timeline(
    timeline_df: pd.DataFrame,
    title: str = "Chat Timeline",
    period_col: str = "period",
    count_col: str = "message_count"
) -> go.Figure:
    """
    Plot a timeline (daily, weekly, monthly) of message counts.
    Args:
        timeline_df: DataFrame with period and message_count columns.
        title: Chart title.
        period_col: Name of the period column.
        count_col: Name of the message count column.
    Returns:
        Plotly Figure.
    """
    fig = px.line(
        timeline_df,
        x=period_col,
        y=count_col,
        title=title,
        markers=True,
        labels = {period_col: "Time", count_col: "Number of Messages"}
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Time",
        yaxis_title="Messages",
        hovermode="x unified",
        legend_title_text="",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig

def plot_user_activity(
    df: pd.DataFrame,
    activity_type: str = "hourly"
) -> go.Figure:
    """
    Plot user activity distribution (hourly, daily of week).
    Args:
        df: DataFrame with 'hour' and 'day' columns.
        activity_type: 'hourly' or 'daily'
    Returns:
        Plotly Figure.
    """
    if activity_type == "hourly":
        activity = df['hour'].value_counts().sort_index()
        fig = px.bar(
            x=activity.index,
            y=activity.values,
            labels={'x': 'Hour of Day', 'y': 'Messages'},
            title="Activity by Hour of Day"
        )
        fig.update_traces(hovertemplate="Hour %{x}: %{y} messages")
        fig.update_layout(template="plotly_white")
        return fig
    elif activity_type == "daily":
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # If 'day_name' not precomputed in df, create it
        if 'day_name' not in df.columns:
            df['day_name'] = pd.to_datetime(df['date']).dt.day_name()
        activity = df['day_name'].value_counts().reindex(day_order)
        fig = px.bar(
            x=activity.index,
            y=activity.values,
            labels={'x': 'Day of Week', 'y': 'Messages'},
            title="Activity by Day of Week"
        )
        fig.update_traces(hovertemplate="%{x}: %{y} messages")
        fig.update_layout(template="plotly_white")
        return fig
    else:
        raise ValueError("activity_type must be 'hourly' or 'daily'")

def plot_busy_users(
    user_counts: pd.Series,
    percent_df: pd.DataFrame,
    top_n: int = 5
) -> go.Figure:
    """
    Plot busy users as a bar chart.
    Args:
        user_counts: Series of top user message counts.
        percent_df: DataFrame with 'user' and 'percent' columns.
    Returns:
        Plotly Figure.
    """
    top_counts = user_counts.head(top_n)
    fig = px.bar(
        x=top_counts.index,
        y=top_counts.values,
        labels={'x': 'User', 'y': 'Messages'},
        title=f"Top {top_n} Active Users",
        color=top_counts.values,
        color_continuous_scale="Viridis",
    )
    fig.update_traces(hovertemplate="User %{x}: %{y} messages")
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig

def plot_wordcloud(
    freq_dict: Dict[str, int],
    width: int = 800,
    height: int = 400,
    max_words: int = 200,
    background_color: str = "white",
    font_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate a wordcloud from a frequency dictionary.
    Args:
        freq_dict: Dictionary of word frequencies.
        width: Width of the wordcloud image.
        height: Height of the wordcloud image.
        max_words: Maximum number of words.
        background_color: Background color.
    Returns:
        Matplotlib Figure.
    """
    if font_path is None:
        # Try to use a Devanagari-friendly font; fallback to default if not found.
        try:
            font_path = "assets/NotoSansDevanagari-Regular.ttf"
        except Exception:
            font_path = None
    wc = WordCloud(
        width=width, height=height, max_words=max_words, background_color=background_color,
        font_path=font_path, regexp=r"[\w']+|[\u0900-\u097F]+"
    ).generate_from_frequencies(freq_dict)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout()
    return fig

def plot_common_words(
    common_words_df: pd.DataFrame,
    top_n: int = 25
) -> go.Figure:
    """
    Plot most common words as a horizontal bar chart.
    Args:
        common_words_df: DataFrame with 'words' and 'frequency' columns.
    Returns:
        Plotly Figure.
    """
    df = common_words_df.sort_values('frequency', ascending=True).head(top_n)
    fig = px.bar(
        df,
        x='frequency',
        y='words',
        orientation='h',
        title=f'Most Common {top_n} Words',
        labels={'frequency': 'Count', 'words': 'Word'},
        color='frequency',
        color_continuous_scale="Blues"
    )
    fig.update_traces(hovertemplate="Word %{y}: %{x} uses")
    fig.update_layout(template="plotly_white", yaxis={'categoryorder': 'total ascending'})
    return fig

def plot_emoji_pie(
    emoji_df: pd.DataFrame,
    top_n: int = 5
) -> go.Figure:
    """
    Plot a pie chart of the top emojis used.
    Args:
        emoji_df: DataFrame with 'emoji' and 'count' columns.
        top_n: Number of top emojis to display.
    Returns:
        Plotly Figure.
    """
    top_emojis = emoji_df.head(top_n)
    fig = px.pie(
        top_emojis,
        values='count',
        names='emoji',
        title=f'Top {top_n} Emojis Used',
        hole=0.3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template="plotly_white")
    return fig

def plot_emoji_bar(
    emoji_df: pd.DataFrame,
    top_n: int = 10
) -> go.Figure:
    """
    Plot a bar chart of the most used emojis.
    Args:
        emoji_df: DataFrame with 'emoji' and 'count' columns.
        top_n: Number of top emojis to display.
    Returns:
        Plotly Figure.
    """
    top_emojis = emoji_df.head(top_n)
    fig = px.bar(
        top_emojis,
        x='emoji',
        y='count',
        title=f'Most Used {top_n} Emojis',
        labels={'emoji': 'Emoji', 'count': 'Usage Count'},
        color='count',
        color_continuous_scale="OrRd"
    )
    fig.update_traces(hovertemplate="Emoji %{x}: %{y} uses")
    fig.update_layout(template="plotly_white")
    return fig

def plot_links_timeline(
    df: pd.DataFrame,
    freq: str = 'ME'
) -> go.Figure:
    """
    Plot a timeline of links shared in the chat.
    Args:
        df: Preprocessed chat DataFrame with 'date' and 'message' columns.
        freq: Frequency: 'D' = daily, 'M' = monthly, 'W' = weekly.
    Returns:
        Plotly Figure.
    """
    import re
    from urlextract import URLExtract
    extractor = URLExtract()
    df['has_link'] = df['message'].apply(lambda x: len(extractor.find_urls(str(x))) > 0)
    timeline = df[df['has_link']].set_index('date').resample(freq)['message'].count().reset_index()
    timeline.rename(columns={'date': 'period', 'message': 'link_count'}, inplace=True)
    fig = px.line(
        timeline,
        x='period',
        y='link_count',
        title='Links Shared Over Time',
        markers=True,
        labels = {'period': 'Time', 'link_count': 'Links Shared'}
    )
    fig.update_layout(template="plotly_white", xaxis_title="Time", yaxis_title="Links Shared")
    return fig

def plot_sentiment_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
    if df.empty or "sentiment" not in df:
        return None
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]
    fig = px.pie(
        sentiment_counts,
        values="count",
        names="sentiment",
        title="Sentiment Distribution",
        color="sentiment",
        color_discrete_map={"positive": "green", "negative": "red", "neutral": "gray"},
        hole=0.35,
    )
    return fig

def plot_sentiment_timeline(df: pd.DataFrame, freq: str = "D") -> Optional[go.Figure]:
    """
    Robust timeline plot for sentiment.
    If filtered to a single sentiment, shows only that sentiment's trend.
    If not enough data, returns None.
    """
    if df.empty or "sentiment" not in df:
        return None
    # Group by date and sentiment
    timeline = (
        df.set_index("date")
        .groupby([pd.Grouper(freq=freq), "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Only columns with actual sentiment
    y_cols = [c for c in ["positive", "neutral", "negative"] if c in timeline.columns]
    if len(timeline) < 2 or not y_cols:
        return None
    fig = px.line(
        timeline,
        x="date",
        y=y_cols,
        title="Sentiment Over Time",
        labels={"value": "Message Count", "date": "Date"},
    )
    fig.update_layout(legend_title_text="Sentiment")
    return fig

def plot_emotion_distribution(df: pd.DataFrame, emotions: Optional[List[str]] = None) -> Optional[go.Figure]:
    if df.empty or "emotion" not in df:
        return None
    counts = df["emotion"].value_counts().reset_index()
    counts.columns = ["emotion", "count"]
    if emotions:
        counts = counts[counts["emotion"].isin(emotions)]
    if counts.empty:
        return None
    fig = px.pie(
        counts,
        values="count",
        names="emotion",
        title="Emotion Distribution",
        hole=0.35,
    )
    return fig

def plot_emotion_timeline(df: pd.DataFrame, freq: str = "D", emotion_labels: Optional[List[str]] = None) -> Optional[go.Figure]:
    if df.empty or "emotion" not in df:
        return None
    timeline = (
        df.set_index("date")
        .groupby([pd.Grouper(freq=freq), "emotion"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    y_cols = emotion_labels if emotion_labels else [col for col in timeline.columns if col != "date"]
    y_cols = [c for c in y_cols if c in timeline.columns]
    if len(timeline) < 2 or not y_cols:
        return None
    fig = px.line(
        timeline,
        x="date",
        y=y_cols,
        title="Emotion Trends Over Time",
        labels={"value": "Message Count", "date": "Date"},
    )
    fig.update_layout(legend_title_text="Emotion")
    return fig

# You can add more visualizations as needed, e.g., media timeline, user comparison, etc.