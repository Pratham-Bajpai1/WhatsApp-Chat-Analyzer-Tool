import re
import pandas as pd
from typing import List, Optional, Tuple, Union, IO, Set
import zipfile
import io
from datetime import datetime


# Configurable media tokens (add more as needed)
DEFAULT_MEDIA_TOKENS = {
    "<Media omitted>",
    "‎image omitted",  # iPhone, invisible leading char
    "image omitted",
    "‎document omitted",
    "document omitted",
    "‎video omitted",
    "video omitted",
    "‎audio omitted",
    "audio omitted"
}

# Supported date/time regex patterns
DATE_TIME_PATTERNS = [
    # e.g., 12/5/21, 9:32 pm -
    r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?:\s?[apAP][mM])?)\s-\s',
    # e.g., [12/5/21, 21:32:00] - (some locales use brackets and 24h time)
    r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?::\d{2})?)\]\s-\s',
    # e.g., 2021-12-05, 21:32 - (ISO style)
    r'(\d{4}-\d{2}-\d{2}),\s(\d{1,2}:\d{2}(?:\s?[apAP][mM])?)\s-\s',
]

DATETIME_FORMATS = [
    "%d/%m/%y, %I:%M %p",
    "%d/%m/%Y, %I:%M %p",

    "%d/%m/%Y, %H:%M",
    "%d/%m/%y, %H:%M",
    "%Y-%m-%d, %H:%M",
    "%d/%m/%y, %H:%M:%S",
    "%d/%m/%Y, %H:%M:%S",
    "%d/%m/%y, %I:%M:%S %p",
    "%d/%m/%Y, %I:%M:%S %p",
    "%Y-%m-%d, %H:%M:%S",
]

def extract_txt_from_zip(zip_bytes: bytes) -> Optional[str]:
    """
    Extract the first WhatsApp .txt file from a zip archive.
    Returns the decoded text, or None if not found.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                if name.lower().endswith('.txt'):
                    with zf.open(name) as f:
                        return f.read().decode('utf-8', errors='replace')
    except Exception as e:
        print(f"Failed to extract txt from zip: {e}")
    return None

def try_parse_datetime(date_str: str, fmts: List[str]) -> Optional[datetime]:
    """
    Try various known WhatsApp date formats. Returns a datetime or None.
    """
    for fmt in fmts:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except Exception:
            continue
    return None

def detect_format_and_pattern(data: str) -> Tuple[str, Optional[re.Pattern]]:
    """
    Detects WhatsApp export format: 'android' or 'iphone'.
    Returns (format_name, re.Pattern for split).
    """
    # Android: 02/01/2025, 13:27 - Name: Message
    pat_android = r'^\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s[APMapm]{2})?\s-\s'
    # iPhone: [01/01/25, 8:31:54 AM] Name: Message or [01/01/25, 8:31 AM] Name: Message
    pat_iphone = r'^\[.*?\]\s'
    # Try Android
    if re.search(pat_android, data, re.MULTILINE):
        return "android", re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s?[APMapm]{2})?)\s-\s')
    # Try iPhone
    if re.search(pat_iphone, data, re.MULTILINE):
        return "iphone", re.compile(r'\[(.*?)\]\s')
    return "unknown", None

def preprocess(
    source: Union[str, bytes, IO[bytes]],
    media_tokens: Optional[Set[str]] = None
) -> pd.DataFrame:
    """
    Preprocess WhatsApp chat data (txt or zip) into a structured DataFrame.
    Supports Android and iPhone export styles, including multiline messages.
    Args:
        source: chat txt data as str, bytes, or file-like object
        media_tokens: set of strings considered media; can be customized
    Returns:
        pd.DataFrame with columns: date, user, message, year, month, day, hour, minute
    """
    # Read data from zip, bytes, or file-like
    if isinstance(source, bytes):
        txt = extract_txt_from_zip(source)
        if not txt:
            txt = source.decode("utf-8", errors="replace")
    elif hasattr(source, "read"):
        pos = source.tell()
        content = source.read()
        if isinstance(content, bytes):
            txt = extract_txt_from_zip(content)
            if not txt:
                txt = content.decode("utf-8", errors="replace")
        else:
            txt = content
        source.seek(pos)
    else:
        txt = source

    txt = txt.replace('\r\n', '\n').replace('\r', '\n')
    fmt, pattern = detect_format_and_pattern(txt)
    if not pattern:
        raise ValueError("Unsupported WhatsApp export format or unreadable file.")

    # Split messages and extract dates
    if fmt == "android":
        date_fmts = [
            "%d/%m/%Y, %H:%M",
            "%d/%m/%Y, %I:%M %p",
            "%d/%m/%y, %H:%M",
            "%d/%m/%y, %I:%M %p"
        ]
        split_regex = re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s?[APMapm]{2})?)\s-\s')
        splits = [m for m in split_regex.finditer(txt)]
        if not splits:
            raise ValueError("No valid Android-style messages found.")
        messages = []
        dates = []
        for idx, m in enumerate(splits):
            start = m.end()
            end = splits[idx + 1].start() if idx + 1 < len(splits) else len(txt)
            chunk = txt[start:end].strip()
            date_str = m.group(1)
            date_obj = try_parse_datetime(date_str, date_fmts)
            dates.append(date_obj if date_obj else date_str)
            messages.append(chunk)
    elif fmt == "iphone":
        # e.g., [01/01/25, 8:31:54 AM] Name: Message
        date_fmts = [
            "%d/%m/%y, %I:%M:%S %p",
            "%d/%m/%y, %I:%M %p",
            "%d/%m/%Y, %I:%M:%S %p",
            "%d/%m/%Y, %I:%M %p",
            "%d/%m/%y, %H:%M:%S",
            "%d/%m/%Y, %H:%M:%S",
            "%d/%m/%y, %H:%M",
            "%d/%m/%Y, %H:%M",
        ]
        # Use non-greedy split to handle multiline
        split_regex = re.compile(r'\[(.*?)\]\s')
        splits = [m for m in split_regex.finditer(txt)]
        if not splits:
            raise ValueError("No valid iPhone-style messages found.")
        messages = []
        dates = []
        for idx, m in enumerate(splits):
            start = m.end()
            end = splits[idx + 1].start() if idx + 1 < len(splits) else len(txt)
            chunk = txt[start:end].strip()
            date_str = m.group(1)
            date_obj = try_parse_datetime(date_str, date_fmts)
            dates.append(date_obj if date_obj else date_str)
            messages.append(chunk)
    else:
        raise ValueError("Unsupported WhatsApp export format or unreadable file.")

    # Parse users and messages (support multiline and group notifications)
    users = []
    msg_texts = []
    # For iPhone, sometimes system messages have no colon
    user_regex = re.compile(r'^([\w\W]+?):\s')
    for msg in messages:
        m = user_regex.match(msg)
        if m:
            users.append(m.group(1))
            msg_texts.append(msg[m.end():])
        else:
            users.append("group_notification")
            msg_texts.append(msg)

    df = pd.DataFrame({'date': dates, 'user': users, 'message': msg_texts})
    df = df.dropna(subset=['date', 'message'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    return df