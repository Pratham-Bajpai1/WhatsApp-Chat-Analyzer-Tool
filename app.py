import streamlit as st
import pandas as pd
import preprocessing, stopwords, stats, emoji_analysis, busy_users, timeline, visualization
import content_extractor as content_extractor
import sentiment_analyzer as sentiment_analyzer
from collections import Counter
import os

# ==============================
# Mobile Detection and Setup
# ==============================
MOBILE_WIDTH_THRESHOLD = 700

# Initialize mobile detection in session state
if "is_mobile" not in st.session_state:
    st.session_state["is_mobile"] = False

# JavaScript to detect mobile and post message to Streamlit
st.markdown(f"""
    <script>
    const isMobile = window.innerWidth < {MOBILE_WIDTH_THRESHOLD};
    window.parent.postMessage({{isMobile}}, "*");
    </script>
""", unsafe_allow_html=True)

# For development/testing - can be removed in production
mobile_override = st.sidebar.selectbox(
    "Simulate mobile view?",
    ["Detect automatically", "Yes (force mobile)", "No (force desktop)"],
    index=0,
    help="For development/testing only"
)

if mobile_override == "Yes (force mobile)":
    is_mobile = True
elif mobile_override == "No (force desktop)":
    is_mobile = False
else:
    is_mobile = st.session_state.get("is_mobile", False)

# ==============================
# Responsive CSS
# ==============================
st.markdown("""
    <style>
    /* Base styles that work for both mobile and desktop */
    .main { background: #f9f9f9; }
    .stTabs [role="tablist"] { justify-content: center; }
    .stTooltip { font-size: 0.9em; color: #888 !important; }

    /* Mobile-specific adjustments */
    @media screen and (max-width: %dpx) {
        .block-container { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
        .sidebar-content, .css-ew7tme { padding: 0.5rem 0.5rem !important; }
        .stRadio [role='radiogroup'] > label { font-size: 1.10rem; padding: 0.15rem 0.3rem; }
        .stSelectbox label, .stMultiSelect label { font-size: 1.07rem; }
        .stDataFrame { max-width: 100vw !important; }
        .stDataFrame, .dataframe, .stTable { font-size: 0.98rem; }
        .js-plotly-plot { width: 100%% !important; }
        .stExpanderHeader { font-size: 1.08rem; }
        .streamlit-expanderContent { padding: 0.5rem 0.25rem; }
        .metric-label { font-size: 0.93em; }
        .stMarkdown h1 { font-size: 1.5rem !important; }
        .stMarkdown h2 { font-size: 1.3rem !important; }
        .stMarkdown h3 { font-size: 1.1rem !important; }
    }
    </style>
""" % MOBILE_WIDTH_THRESHOLD, unsafe_allow_html=True)

# ==============================
# Sidebar Configuration
# ==============================
st.sidebar.title("📊 WhatsApp Chat Analyzer")
sidebar_img = """
    <div style="text-align: center;">
        <img src="https://i.gifer.com/75ez.gif" width="%s" height="%s">
    </div>
""" % ("220" if is_mobile else "300", "90" if is_mobile else "150")
st.sidebar.markdown(sidebar_img, unsafe_allow_html=True)
st.sidebar.markdown("Analyze your exported WhatsApp chat (.txt or .zip).")

uploaded_file = st.sidebar.file_uploader("Upload WhatsApp export (.txt or .zip)", type=["txt", "zip"])
date_filter = st.sidebar.date_input("Optional: Filter by date", [], help="Show messages only within this date range")

# ==============================
# Main Content
# ==============================
st.title("📲 WhatsApp Chat Analyzer")
st.markdown("Upload a WhatsApp chat export to explore detailed statistics, emoji usage, timelines, and more!")

if uploaded_file is None:
    st.info("Please upload a WhatsApp .txt or .zip file to get started.")
    st.stop()


# ==============================
# Data Loading
# ==============================
@st.cache_data(show_spinner=False, max_entries=5)
def load_chat_data(file) -> pd.DataFrame:
    if file is None:
        return None
    try:
        content = file.read()
        df = preprocessing.preprocess(content)
        return df
    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
        return None


with st.spinner("Processing chat data..."):
    df = load_chat_data(uploaded_file)
    if df is None or df.empty:
        st.error("❌ No data could be extracted from the file.")
        st.stop()

if len(date_filter) == 2:
    start, end = date_filter
    df = df[(df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))]

user_list = df['user'].dropna().unique().tolist()
if "group_notification" in user_list:
    user_list.remove("group_notification")
user_list.sort()
user_list.insert(0, "Overall")
selected_user = st.sidebar.selectbox("Analyze messages by user", user_list)

STOPFILE_PATH = "assets/stop_hinglish.txt"
sw = stopwords.Stopwords(stopword_file=STOPFILE_PATH)
stopword_set = sw.load()

# ==============================
# Navigation Setup
# ==============================
PAGES = [
    "Chat Overview", "User Stats", "Emoji Analysis", "Timelines",
    "WordCloud", "Links & Media", "Shared Content", "Sentiment Analysis"
]

if is_mobile:
    # Vertical radio buttons for mobile
    selected_page = st.radio(
        "Navigate",
        PAGES,
        horizontal=False,
        label_visibility="collapsed",
        key="mobile_nav"
    )
else:
    # Tabs for desktop - we'll use st.tabs and track the selected tab
    tabs = st.tabs(PAGES)
    for i, tab in enumerate(tabs):
        if tab.button(PAGES[i]):
            st.session_state.current_tab = i
    current_tab = st.session_state.get("current_tab", 0)
    selected_page = PAGES[current_tab]


# ==============================
# Helper Functions for Responsive Layouts
# ==============================
def responsive_metrics(stat_dict):
    if is_mobile:
        col1, col2 = st.columns(2)
        col1.metric("Messages", stat_dict['total_messages'], help="Total messages sent")
        col2.metric("Words", stat_dict['total_words'], help="Total words used")
        col1, col2 = st.columns(2)
        col1.metric("Media Files", stat_dict['total_media_files'], help="Total media attachments shared")
        col2.metric("Links", stat_dict['total_links'], help="Total links shared")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Messages", stat_dict['total_messages'], help="Total messages sent")
        col2.metric("Words", stat_dict['total_words'], help="Total words used")
        col3.metric("Media Files", stat_dict['total_media_files'], help="Total media attachments shared")
        col4.metric("Links", stat_dict['total_links'], help="Total links shared")


def responsive_dataframe(df, height_mobile=220, height_desktop=500):
    st.caption("Scroll horizontally →" if is_mobile else "")
    st.dataframe(df, use_container_width=True, hide_index=True, height=height_mobile if is_mobile else height_desktop)


def responsive_plotly_chart(fig, height_mobile=260, height_desktop=400):
    fig.update_layout(
        height=height_mobile if is_mobile else height_desktop,
        margin=dict(t=40, b=24, l=8, r=8)
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================
# Page Content - Only render the selected page
# ==============================
if selected_page == "Chat Overview":
    st.header("Overview")
    st.markdown("**Basic statistics for the selected user or group.**")
    st.divider()
    stat_dict = stats.fetch_stats(df, selected_user)
    responsive_metrics(stat_dict)
    st.write("")
    st.info("Below is the complete chat log. Use the scrollbars to explore the data.")
    responsive_dataframe(df)

elif selected_page == "User Stats":
    st.header("Active Users")
    st.markdown("**Who are the most active participants in the chat?**")
    st.divider()

    unique_users = [u for u in df['user'].unique() if u != "group_notification"]
    n_users = len(unique_users)

    if n_users <= 2:
        min_n = 1
        max_n = n_users
        step = 1
        help_msg = "Individual chat detected. Only 1 or 2 users."
    else:
        min_n = 2
        max_n = n_users
        step = 1
        help_msg = f"Group chat detected. {n_users} participants."

    top_n = st.slider(
        "How many top users to show?",
        min_value=min_n, max_value=max_n, value=min(5, max_n), step=step,
        help=help_msg
    )

    if selected_user == "Overall":
        user_counts, percent_df = busy_users.get_busy_users(df)
        responsive_plotly_chart(
            visualization.plot_busy_users(user_counts, percent_df, top_n=top_n)
        )
        st.caption(f"Showing top {top_n} users by message count.")
        responsive_dataframe(percent_df.head(top_n), 300, 300)
    else:
        st.info("User statistics are only shown for group (Overall) selection.")

    st.divider()
    st.subheader("Activity Patterns")
    st.caption("See when the chat is most active during the day and week.")

    if is_mobile:
        responsive_plotly_chart(visualization.plot_user_activity(df, activity_type="hourly"))
        responsive_plotly_chart(visualization.plot_user_activity(df, activity_type="daily"))
    else:
        col1, col2 = st.columns(2)
        with col1:
            responsive_plotly_chart(visualization.plot_user_activity(df, activity_type="hourly"))
        with col2:
            responsive_plotly_chart(visualization.plot_user_activity(df, activity_type="daily"))

elif selected_page == "Emoji Analysis":
    st.header("Emoji Analysis")
    st.markdown("**Which emojis are used the most?**")
    st.divider()

    top_n_emoji = st.slider("Show top N emojis:", min_value=2, max_value=20, value=10, step=1)
    emoji_df = emoji_analysis.emoji_stats(df, selected_user)

    if not emoji_df.empty:
        if is_mobile:
            responsive_plotly_chart(visualization.plot_emoji_bar(emoji_df, top_n=top_n_emoji))
            with st.expander("Emoji Data Table"):
                responsive_dataframe(emoji_df.head(top_n_emoji), 350, 350)
        else:
            col1, col2 = st.columns([1, 2])
            with col1:
                responsive_dataframe(emoji_df.head(top_n_emoji), 350, 350)
            with col2:
                responsive_plotly_chart(visualization.plot_emoji_bar(emoji_df, top_n=top_n_emoji))

        st.divider()
        st.subheader("Top Emojis Pie Chart")
        responsive_plotly_chart(visualization.plot_emoji_pie(emoji_df, top_n=top_n_emoji))
        st.caption(f"Showing top {top_n_emoji} emojis by usage.")
    else:
        st.info("No emojis found for the selected user.")

elif selected_page == "Timelines":
    st.header("Timeline Analysis")
    st.markdown("**How does chat activity change over time?**")
    st.divider()

    if is_mobile:
        responsive_plotly_chart(
            visualization.plot_timeline(timeline.timeline(df, selected_user, freq='ME'), title="Monthly Timeline"))
        responsive_plotly_chart(
            visualization.plot_timeline(timeline.timeline(df, selected_user, freq='W'), title="Weekly Timeline"))
        responsive_plotly_chart(
            visualization.plot_timeline(timeline.timeline(df, selected_user, freq='D'), title="Daily Timeline"))
    else:
        col1, col2, col3 = st.columns(3)
        for freq, label, col in zip(['ME', 'W', 'D'], ['Monthly', 'Weekly', 'Daily'], [col1, col2, col3]):
            time_df = timeline.timeline(df, selected_user, freq=freq)
            col.plotly_chart(visualization.plot_timeline(time_df, title=f"{label} Timeline"), use_container_width=True)
            col.caption(f"{label} message activity.")

elif selected_page == "WordCloud":
    st.header("WordCloud & Common Words")
    st.markdown("**Visualize the most used words (stopwords removed).**")
    st.divider()

    font_path = None
    if os.path.exists("assets/NotoSansDevanagari-Regular.ttf"):
        font_path = "assets/NotoSansDevanagari-Regular.ttf"

    st.caption(
        "WordCloud supports Hindi and English. For best results in Hindi, ensure NotoSansDevanagari is present in the assets folder.")
    temp_df = df if selected_user == "Overall" else df[df['user'] == selected_user]
    filtered_words = []

    for msg in temp_df['message']:
        for word in str(msg).lower().split():
            word = word.strip().strip('.,!?-_()[]{}')
            if word and word not in stopword_set and not word.isnumeric():
                filtered_words.append(word)

    freq_dict = dict(Counter(filtered_words).most_common(100))

    if freq_dict:
        st.pyplot(visualization.plot_wordcloud(freq_dict, font_path=font_path), use_container_width=True)
        st.divider()

        top_n_words = st.slider("Show top N common words:", min_value=5, max_value=50, value=25, step=5)
        common_words_df = pd.DataFrame(freq_dict.items(), columns=["words", "frequency"]).sort_values("frequency",
                                                                                                      ascending=False)

        responsive_plotly_chart(visualization.plot_common_words(common_words_df, top_n=top_n_words))
        responsive_dataframe(common_words_df.head(top_n_words), 350, 350)
    else:
        st.info("Not enough text data to generate a wordcloud.")

elif selected_page == "Links & Media":
    st.header("Links & Media Over Time")
    st.markdown("**How are links and media shared throughout the chat history?**")
    st.divider()

    responsive_plotly_chart(visualization.plot_links_timeline(df, freq='ME'))
    st.caption("Shows the volume of links shared each month.")

elif selected_page == "Shared Content":
    st.header("Shared Content")
    st.markdown("🔗 **Links, Media Mentions, Documents, and Locations shared in the chat**")
    st.divider()

    # Optional user/date filter
    if is_mobile:
        filter_user = st.selectbox("Filter by user (optional)",
                                   ["All"] + [u for u in df['user'].unique() if u != "group_notification"], index=0)
        filter_date = st.date_input("Filter by date (optional)", [])
    else:
        sc1, sc2 = st.columns(2)
        filter_user = sc1.selectbox("Filter by user (optional)",
                                    ["All"] + [u for u in df['user'].unique() if u != "group_notification"], index=0)
        filter_date = sc2.date_input("Filter by date (optional)", [])

    filtered_df = df.copy()
    if filter_user != "All":
        filtered_df = filtered_df[filtered_df['user'] == filter_user]
    if len(filter_date) == 2:
        start, end = filter_date
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.to_datetime(start)) & (filtered_df['date'] <= pd.to_datetime(end))]
    st.write("")

    # Links Section
    with st.expander("🔗 Shared Links", expanded=not is_mobile):
        links_df = content_extractor.extract_links(filtered_df)
        if not links_df.empty:
            group_links = content_extractor.group_links_by_user(links_df)
            responsive_plotly_chart(
                visualization.px.bar(
                    group_links, x="user", y="link_count", title="Links Shared by User",
                    labels={"user": "User", "link_count": "Links"}
                )
            )
            st.caption("All shared URLs (clickable):")
            links_df_display = links_df.copy()
            links_df_display["url"] = links_df_display["url"].apply(lambda x: f"[{x}]({x})")
            responsive_dataframe(links_df_display[["date", "user", "url"]], 250, 250)
        else:
            st.info("No links found in this chat.")

    # Media Mentions Section
    with st.expander("🖼️ Media Mentions (images, videos, documents, audio, contacts)", expanded=False):
        media_df = content_extractor.extract_media_mentions(filtered_df)
        if media_df["count"].sum() > 0:
            responsive_plotly_chart(
                visualization.px.pie(
                    media_df, names="type", values="count", title="Media Mentions"
                )
            )
            responsive_dataframe(media_df, 150, 150)
        else:
            st.info("No media mentions found.")

    # Document Mentions Section
    with st.expander("📄 Shared Documents", expanded=False):
        docs_df = content_extractor.extract_document_mentions(filtered_df)
        if not docs_df.empty:
            responsive_dataframe(docs_df[["date", "user", "filename", "message"]], 250, 250)
        else:
            st.info("No documents found in this chat.")

    # Locations Section
    with st.expander("📍 Shared Locations", expanded=False):
        loc_df = content_extractor.extract_locations(filtered_df)
        if not loc_df.empty:
            loc_df_display = loc_df.copy()
            loc_df_display["url"] = loc_df_display["url"].apply(lambda x: f"[Google Maps]({x})")
            responsive_dataframe(loc_df_display[["date", "user", "latitude", "longitude", "url"]], 200, 200)
        else:
            st.info("No location links found.")

elif selected_page == "Sentiment Analysis":
    st.header("Sentiment & Emotion Analysis")
    st.markdown(
        "See the distribution, trends, and timeline of positive, negative, and neutral sentiments — and basic emotions — in the chat.")
    st.divider()

    # Optional filters
    if is_mobile:
        filter_user = st.selectbox("Filter by user (optional)",
                                   ["All"] + [u for u in df['user'].unique() if u != "group_notification"], index=0,
                                   key="sentiment_user_filter")
        filter_sentiment = st.selectbox("Filter by sentiment (optional)", ["All", "positive", "neutral", "negative"],
                                        index=0)
    else:
        sc1, sc2 = st.columns(2)
        filter_user = sc1.selectbox("Filter by user (optional)",
                                    ["All"] + [u for u in df['user'].unique() if u != "group_notification"], index=0,
                                    key="sentiment_user_filter")
        filter_sentiment = sc2.selectbox("Filter by sentiment (optional)", ["All", "positive", "neutral", "negative"],
                                         index=0)

    filtered_df = df.copy()
    if filter_user != "All":
        filtered_df = filtered_df[filtered_df['user'] == filter_user]
    filtered_df = sentiment_analyzer.analyze_sentiment(filtered_df, text_col="message")
    if filter_sentiment != "All":
        filtered_df = filtered_df[filtered_df["sentiment"] == filter_sentiment]

    st.subheader("Sentiment Distribution")
    fig_s_dist = visualization.plot_sentiment_distribution(filtered_df)
    if fig_s_dist:
        responsive_plotly_chart(fig_s_dist)
    else:
        st.info("Not enough data to display sentiment distribution.")

    st.subheader("Sentiment Timeline")
    fig_s_time = visualization.plot_sentiment_timeline(filtered_df, freq="D")
    if fig_s_time:
        responsive_plotly_chart(fig_s_time)
    else:
        st.info("Not enough data to display sentiment timeline.")

    responsive_dataframe(filtered_df[["date", "user", "message", "sentiment", "sent_compound"]], 300, 300)
    st.divider()

    # Emotion analysis
    st.subheader("Emotion Trends")
    emo_df = sentiment_analyzer.analyze_emotion(df, text_col="message", use_api=True)

    if emo_df.empty or emo_df["emotion"].nunique() == 1 and emo_df["emotion"].iloc[0] == "neutral":
        st.info("No emotions detected; try reloading or check your Hugging Face API token.")
    else:
        available_emotions = sorted(emo_df["emotion"].unique())
        filter_emotions = st.multiselect("Filter emotions to show", options=available_emotions,
                                         default=available_emotions)
        emo_df_filtered = emo_df[emo_df["emotion"].isin(filter_emotions)]
        fig_e_dist = visualization.plot_emotion_distribution(emo_df_filtered, emotions=filter_emotions)

        if fig_e_dist:
            responsive_plotly_chart(fig_e_dist)
        else:
            st.info("Not enough data to display emotion distribution.")

        fig_e_time = visualization.plot_emotion_timeline(emo_df_filtered, freq="D", emotion_labels=filter_emotions)
        if fig_e_time:
            responsive_plotly_chart(fig_e_time)
        else:
            st.info("Not enough data to display emotion timeline.")

        responsive_dataframe(emo_df_filtered[["date", "user", "message", "emotion"]], 300, 300)

# ==============================
# Footer
# ==============================
if not is_mobile:
    st.markdown(
        '<hr><div style="text-align:center; color: #888;">'
        '© 2024 Pratham Bajpai & Contributors &nbsp;|&nbsp; '
        'Powered by Streamlit &nbsp;|&nbsp; '
        '<a href="https://github.com/Pratham-Bajpai1" target="_blank">GitHub</a>'
        '</div>',
        unsafe_allow_html=True
    )