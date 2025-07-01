import pandas as pd
import re
from typing import List, Dict, Any, Optional
from urlextract import URLExtract

MEDIA_PATTERNS = {
    "image": re.compile(r"(image omitted|photo omitted|image omitted|media omitted)", re.IGNORECASE),
    "video": re.compile(r"(video omitted|video omitted)", re.IGNORECASE),
    "document": re.compile(r"(document omitted|document omitted)", re.IGNORECASE),
    "audio": re.compile(r"(audio omitted|audio omitted)", re.IGNORECASE),
    "contact": re.compile(r"(contact card omitted|contact card omitted)", re.IGNORECASE),
}

DOCUMENT_EXTENSIONS = re.compile(r"\b([\w\-\.]+\.(pdf|apk|docx?|xlsx?|pptx?|zip|rar|mp3|mp4|jpg|jpeg|png|csv|txt))\b", re.IGNORECASE)
LOCATION_URL = re.compile(r"https://maps\.google\.com/\?q=([\-0-9\.]+),([\-0-9\.]+)")

def extract_links(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all shared URLs, with sender and timestamp.
    Returns a DataFrame: ['user', 'date', 'url']
    """
    extractor = URLExtract()
    rows = []
    for _, row in df.iterrows():
        urls = extractor.find_urls(str(row['message']))
        for url in urls:
            rows.append({"user": row['user'], "date": row['date'], "url": url})
    return pd.DataFrame(rows)

def group_links_by_user(links_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group extracted links by user and count.
    """
    return links_df.groupby("user")["url"].count().reset_index().rename(columns={"url": "link_count"}).sort_values("link_count", ascending=False)

def extract_media_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count and list media mentions (image, video, document, audio, contact).
    Returns summary table: ['type', 'count']
    """
    media_counts = {}
    for media_type, pat in MEDIA_PATTERNS.items():
        mask = df['message'].apply(lambda x: bool(pat.fullmatch(str(x).strip())))
        media_counts[media_type] = mask.sum()
    return pd.DataFrame(list(media_counts.items()), columns=["type", "count"]).sort_values("count", ascending=False)

def extract_document_mentions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract document filenames mentioned in messages, with sender and timestamp.
    Returns DataFrame: ['user', 'date', 'filename', 'message']
    """
    rows = []
    for _, row in df.iterrows():
        matches = DOCUMENT_EXTENSIONS.findall(str(row['message']))
        for match in matches:
            filename = match[0]
            rows.append({
                "user": row['user'],
                "date": row['date'],
                "filename": filename,
                "message": row['message']
            })
    return pd.DataFrame(rows)

def extract_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Google Maps location URLs, with coordinates, sender, and timestamp.
    Returns: ['user', 'date', 'latitude', 'longitude', 'url']
    """
    rows = []
    for _, row in df.iterrows():
        for match in LOCATION_URL.finditer(str(row['message'])):
            lat, lon = match.groups()
            url = match.group(0)
            rows.append({
                "user": row['user'],
                "date": row['date'],
                "latitude": lat,
                "longitude": lon,
                "url": url
            })
    return pd.DataFrame(rows)