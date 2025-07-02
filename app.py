import streamlit as st
import pandas as pd
import preprocessing, stopwords, stats, emoji_analysis, busy_users, timeline, visualization
import content_extractor as content_extractor
import sentiment_analyzer as sentiment_analyzer
from collections import Counter
import os
from datetime import datetime  # Import datetime for timestamp

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.markdown(
    """
    <style>
    .main { background: #f9f9f9; }
    .stTabs [role="tablist"] { justify-content: center; }
    .stTooltip { font-size: 0.9em; color: #888 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("📊 WhatsApp Chat Analyzer")
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <img src="https://i.gifer.com/75ez.gif" width="300" height="150">
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("Analyze your exported WhatsApp chat (.txt or .zip).")

uploaded_file = st.sidebar.file_uploader("Upload WhatsApp export (.txt or .zip)", type=["txt", "zip"])
date_filter = st.sidebar.date_input("Optional: Filter by date", [], help="Show messages only within this date range")

# --- Sidebar Navigation for Pages ---
page = st.sidebar.radio(
    "Navigation",
    ["Chat Analyzer", "Feedback"],
    index=0  # Default to Chat Analyzer
)


# --- MAIN LOGIC ---
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


# --- Feedback File Path ---
FEEDBACK_FILE = "feedback.csv"


# --- Helper function to load feedback ---
def load_feedback() -> pd.DataFrame:
    if os.path.exists(FEEDBACK_FILE):
        try:
            df_feedback = pd.read_csv(FEEDBACK_FILE)
            # Ensure timestamp is datetime for sorting
            df_feedback['timestamp'] = pd.to_datetime(df_feedback['timestamp'])
            return df_feedback
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=['name', 'rating', 'comment', 'timestamp'])
    return pd.DataFrame(columns=['name', 'rating', 'comment', 'timestamp'])


# --- Helper function to save feedback ---
def save_feedback(df_feedback: pd.DataFrame):
    df_feedback.to_csv(FEEDBACK_FILE, index=False)


# --- Chat Analyzer Page ---
if page == "Chat Analyzer":
    st.title("📲 WhatsApp Chat Analyzer")
    st.markdown("Upload a WhatsApp chat export to explore detailed statistics, emoji usage, timelines, and more!")

    # --- NEW WALKTHROUGH SECTION ---
    with st.expander("📘 How to Export Your WhatsApp Chat (Android & iPhone) in English", expanded=False):
        st.markdown("""
        To analyze your WhatsApp chat, you first need to export it from your phone. Follow these simple steps:
      
        ### 📱 For Android Users:
        1.  **Open WhatsApp** on your phone.
        2.  **Go to the chat** (individual or group) you want to export.
        3.  Tap the **three dots** (⋮) in the top right corner.
        4.  Select **"More"** ➡️ **"Export chat"**.
        5.  You'll be asked: **"Include media?"**
            * **Without Media (Recommended for Analyzer):** Choose this option. This typically creates a `.txt` file directly, or a `.zip` file containing only the `.txt` chat log. You can either extract the `.txt` file from the zip or upload the zip file directly to this tool. This method is faster and often sufficient for analysis.
            * **Include Media (Not Recommended due to large size):** This will create a `.zip` file containing the `.txt` chat log and all media (images, videos). This file will be significantly larger and might take longer to upload and process.
        6.  **Choose how to share** the chat (e.g., Email, Google Drive, Save to device). Select a method to get the `.txt` or `.zip` file to your computer or phone. This tool supports uploading both `.txt` and `.zip` files.
        
        ### 🍎 For iPhone Users:
        1.  **Open WhatsApp** on your iPhone.
        2.  **Go to the chat** (individual or group) you want to export.
        3.  Tap the **contact's name or group name** at the top of the screen.
        4.  Scroll down and select **"Export Chat"**.
        5.  You'll be asked: **"Attach Media?"**
            * **Without Media (Recommended for Analyzer):** Choose this option. This typically creates a `.txt` file directly, or a `.zip` file containing only the `.txt` chat log. You can either extract the `.txt` file from the zip or upload the zip file directly to this tool. This method is faster and often sufficient for analysis.
            * **Attach Media (Not Recommended due to large size):** This will create a `.zip` file containing the `.txt` chat log and all media (images, videos). This file will be significantly larger and might take longer to upload and process.
        6.  **Choose how to share** the chat (e.g., Mail, Save to Files, AirDrop). Select a method to get the `.txt` or `.zip` file to your computer or phone. This tool supports uploading both `.txt` and `.zip` files.
        
        ---
        **💡 Pro-Tip:** For the best and fastest analysis experience with this tool, **always choose "Without Media"** when exporting your chat. This typically generates a smaller `.txt` file or a compact `.zip` containing just the `.txt` chat log, making it quicker to upload and process.
        """)

    with st.expander("📘 How to Export Your WhatsApp Chat (Android & iPhone) in Hindi", expanded=False):
        st.markdown("""
        व्हाट्सएप चैट का विश्लेषण करने के लिए, आपको पहले इसे अपने फोन से निर्यात करना होगा। इन सरल चरणों का पालन करें:

        ### 📱 Android उपयोगकर्ताओं के लिए:
        1.  अपने फोन पर **व्हाट्सएप खोलें**।
        2.  जिस चैट (व्यक्तिगत या समूह) को आप निर्यात करना चाहते हैं, उस पर **जाएं**।
        3.  ऊपर दाएं कोने में **तीन बिंदुओं** (⋮) पर टैप करें।
        4.  **"अधिक"** ➡️ **"चैट निर्यात करें"** चुनें।
        5.  आपसे पूछा जाएगा: **"मीडिया शामिल करें?"**
            * **मीडिया के बिना (विश्लेषक के लिए अनुशंसित):** यह विकल्प चुनें। यह आमतौर पर सीधे एक `.txt` फ़ाइल बनाता है, या केवल `.txt` चैट लॉग वाली एक `.zip` फ़ाइल। आप ज़िप फ़ाइल से `.txt` फ़ाइल निकाल सकते हैं या सीधे इस टूल पर ज़िप फ़ाइल अपलोड कर सकते हैं। यह विधि तेज़ है और अक्सर विश्लेषण के लिए पर्याप्त होती है।
            * **मीडिया सहित (बड़े आकार के कारण अनुशंसित नहीं):** यह एक `.zip` फ़ाइल बनाएगा जिसमें `.txt` चैट लॉग और सभी मीडिया (छवियां, वीडियो) शामिल होंगे। यह फ़ाइल काफी बड़ी होगी और इसे अपलोड करने और प्रोसेस करने में अधिक समय लग सकता है।
        6.  चैट को **साझा करने का तरीका चुनें** (उदाहरण के लिए, ईमेल, गूगल ड्राइव, डिवाइस पर सहेजें)। अपने कंप्यूटर या फोन पर `.txt` या `.zip` फ़ाइल प्राप्त करने के लिए एक विधि चुनें। यह टूल `.txt` और `.zip` दोनों फ़ाइलों को अपलोड करने का समर्थन करता है।
    
        ### 🍎 iPhone उपयोगकर्ताओं के लिए:
        1.  अपने iPhone पर **व्हाट्सएप खोलें**।
        2.  जिस चैट (व्यक्तिगत या समूह) को आप निर्यात करना चाहते हैं, उस पर **जाएं**।
        3.  स्क्रीन के शीर्ष पर **संपर्क के नाम या समूह के नाम** पर टैप करें।
        4.  नीचे स्क्रॉल करें और **"चैट निर्यात करें"** चुनें।
        5.  आपसे पूछा जाएगा: **"मीडिया संलग्न करें?"**
            * **मीडिया के बिना (विश्लेषक के लिए अनुशंसित):** यह विकल्प चुनें। यह आमतौर पर सीधे एक `.txt` फ़ाइल बनाता है, या केवल `.txt` चैट लॉग वाली एक `.zip` फ़ाइल। आप ज़िप फ़ाइल से `.txt` फ़ाइल निकाल सकते हैं या सीधे इस टूल पर ज़िप फ़ाइल अपलोड कर सकते हैं। यह विधि तेज़ है और अक्सर विश्लेषण के लिए पर्याप्त होती है।
            * **मीडिया संलग्न करें (बड़े आकार के कारण अनुशंसित नहीं):** यह एक `.zip` फ़ाइल बनाएगा जिसमें `.txt` चैट लॉग और सभी मीडिया (छवियां, वीडियो) शामिल होंगे। यह फ़ाइल काफी बड़ी होगी और इसे अपलोड करने और प्रोसेस करने में अधिक समय लग सकता है।
        6.  चैट को **साझा करने का तरीका चुनें** (उदाहरण के लिए, मेल, फ़ाइलों में सहेजें, एयरड्रॉप)। अपने कंप्यूटर या फोन पर `.txt` या `.zip` फ़ाइल प्राप्त करने के लिए एक विधि चुनें। यह टूल `.txt` और `.zip` दोनों फ़ाइलों को अपलोड करने का समर्थन करता है।
    
        ---
        **💡 प्रो-टिप:** इस टूल के साथ सर्वश्रेष्ठ और सबसे तेज़ विश्लेषण अनुभव के लिए, अपनी चैट को निर्यात करते समय **हमेशा "मीडिया के बिना" चुनें**। यह आमतौर पर एक छोटी `.txt` फ़ाइल या केवल `.txt` चैट लॉग वाली एक कॉम्पैक्ट `.zip` फ़ाइल उत्पन्न करता है, जिससे यह अपलोड और प्रोसेस करने में तेज़ होता है।
        """)

    # --- END NEW WALKTHROUGH SECTION ---

    if uploaded_file is None:
        st.info("Please upload a WhatsApp .txt or .zip file to get started.")
        st.stop()

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

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Chat Overview", "User Stats", "Emoji Analysis", "Timelines", "WordCloud", "Links & Media", "Shared Content",
        "Sentiment Analysis"
    ])

    # --- TAB 1: CHAT OVERVIEW ---
    with tab1:
        st.header("Overview")
        st.markdown("**Basic statistics for the selected user or group.**")
        st.divider()
        stat_dict = stats.fetch_stats(df, selected_user)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Messages", stat_dict['total_messages'], help="Total messages sent")
        col2.metric("Words", stat_dict['total_words'], help="Total words used")
        col3.metric("Media Files", stat_dict['total_media_files'], help="Total media attachments shared")
        col4.metric("Links", stat_dict['total_links'], help="Total links shared")
        st.write("")
        st.info("Below is the complete chat log. Use the scrollbars to explore the data.")
        st.dataframe(df, use_container_width=True, hide_index=True, height=500)

    # --- TAB 2: USER STATS ---
    with tab2:
        st.header("Active Users")
        st.markdown("**Who are the most active participants in the chat?**")
        st.divider()
        # Compute unique users (excluding group_notification)
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
            st.plotly_chart(
                visualization.plot_busy_users(user_counts, percent_df, top_n=top_n),
                use_container_width=True
            )
            st.caption(f"Showing top {top_n} users by message count.")
            st.dataframe(
                percent_df.head(top_n),
                use_container_width=True,
                hide_index=True,
                height=300
            )
        else:
            st.info("User statistics are only shown for group (Overall) selection.")

        st.divider()
        st.subheader("Activity Patterns")
        st.caption("See when the chat is most active during the day and week.")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(visualization.plot_user_activity(df, activity_type="hourly"), use_container_width=True)
        with col2:
            st.plotly_chart(visualization.plot_user_activity(df, activity_type="daily"), use_container_width=True)

    # --- TAB 3: EMOJI ANALYSIS ---
    with tab3:
        st.header("Emoji Analysis")
        st.markdown("**Which emojis are used the most?**")
        st.divider()
        top_n_emoji = st.slider("Show top N emojis:", min_value=2, max_value=20, value=10, step=1)
        emoji_df = emoji_analysis.emoji_stats(df, selected_user)
        if not emoji_df.empty:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(emoji_df.head(top_n_emoji), use_container_width=True, hide_index=True, height=350)
            with col2:
                st.plotly_chart(visualization.plot_emoji_bar(emoji_df, top_n=top_n_emoji), use_container_width=True)
            st.divider()
            st.subheader("Top Emojis Pie Chart")
            st.plotly_chart(visualization.plot_emoji_pie(emoji_df, top_n=top_n_emoji), use_container_width=True)
            st.caption(f"Showing top {top_n_emoji} emojis by usage.")
        else:
            st.info("No emojis found for the selected user.")

    # --- TAB 4: TIMELINES ---
    with tab4:
        st.header("Timeline Analysis")
        st.markdown("**How does chat activity change over time?**")
        st.divider()
        col1, col2, col3 = st.columns(3)
        for freq, label, col in zip(['ME', 'W', 'D'], ['Monthly', 'Weekly', 'Daily'], [col1, col2, col3]):
            time_df = timeline.timeline(df, selected_user, freq=freq)
            col.plotly_chart(visualization.plot_timeline(time_df, title=f"{label} Timeline"), use_container_width=True)
            col.caption(f"{label} message activity.")

    # --- TAB 5: WORDCLOUD & COMMON WORDS ---
    with tab5:
        st.header("WordCloud & Common Words")
        st.markdown("**Visualize the most used words in your chat (stopwords removed).**")
        st.divider()
        font_path = None
        if os.path.exists("assets/NotoSansDevanagari-Regular.ttf"):
            font_path = "assets/NotoSansDevanagari-Regular.ttf"
        st.caption(
            "WordCloud supports Hindi and English.")
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
            st.plotly_chart(visualization.plot_common_words(common_words_df, top_n=top_n_words),
                            use_container_width=True)
            st.dataframe(common_words_df.head(top_n_words), use_container_width=True, hide_index=True, height=350)
        else:
            st.info("Not enough text data to generate a wordcloud.")

    # --- TAB 6: LINKS & MEDIA TIMELINE ---
    with tab6:
        st.header("Links & Media Over Time")
        st.markdown("**How are links and media shared throughout the chat history?**")
        st.divider()
        st.plotly_chart(visualization.plot_links_timeline(df, freq='ME'), use_container_width=True)
        st.caption("Shows the volume of links shared each month.")
        # Optionally, add media timeline or more visualizations here

    with tab7:
        st.header("Shared Content")
        st.markdown("🔗 **Links, Media Mentions, Documents, and Locations shared in the chat**")
        st.divider()
        # Optional user/date filter
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
        with st.expander("🔗 Shared Links", expanded=True):
            links_df = content_extractor.extract_links(filtered_df)
            if not links_df.empty:
                # Show grouped by user
                group_links = content_extractor.group_links_by_user(links_df)
                st.plotly_chart(
                    visualization.px.bar(
                        group_links, x="user", y="link_count", title="Links Shared by User",
                        labels={"user": "User", "link_count": "Links"}
                    ),
                    use_container_width=True
                )
                st.caption("All shared URLs (clickable):")
                # Make links clickable
                links_df_display = links_df.copy()
                links_df_display["url"] = links_df_display["url"].apply(lambda x: f"[{x}]({x})")
                st.dataframe(links_df_display[["date", "user", "url"]], use_container_width=True, hide_index=True,
                             height=250)
            else:
                st.info("No links found in this chat.")

        # Media Mentions Section
        with st.expander("🖼️ Media Mentions (images, videos, documents, audio, contacts)", expanded=False):
            media_df = content_extractor.extract_media_mentions(filtered_df)
            if media_df["count"].sum() > 0:
                st.plotly_chart(
                    visualization.px.pie(
                        media_df, names="type", values="count", title="Media Mentions"
                    ),
                    use_container_width=True
                )
                st.dataframe(media_df, use_container_width=True, hide_index=True, height=150)
            else:
                st.info("No media mentions found.")

        # Document Mentions Section
        with st.expander("📄 Shared Documents", expanded=False):
            docs_df = content_extractor.extract_document_mentions(filtered_df)
            if not docs_df.empty:
                st.dataframe(docs_df[["date", "user", "filename", "message"]], use_container_width=True,
                             hide_index=True, height=250)
            else:
                st.info("No documents found in this chat.")

        # Locations Section
        with st.expander("📍 Shared Locations", expanded=False):
            loc_df = content_extractor.extract_locations(filtered_df)
            if not loc_df.empty:
                # Make links clickable
                loc_df_display = loc_df.copy()
                loc_df_display["url"] = loc_df_display["url"].apply(lambda x: f"[Google Maps]({x})")
                st.dataframe(loc_df_display[["date", "user", "latitude", "longitude", "url"]], use_container_width=True,
                             hide_index=True, height=200)
            else:
                st.info("No location links found.")

    with tab8:
        st.header("Sentiment & Emotion Analysis")
        st.markdown(
            "See the distribution, trends, and timeline of positive, negative, and neutral sentiments — and basic emotions — in the chat.")
        st.divider()

        # Optional filters
        sc1, sc2 = st.columns(2)
        filter_user = sc1.selectbox("Filter by user (optional)",
                                    ["All"] + [u for u in df['user'].unique() if u != "group_notification"], index=0,
                                    key="sentiment_user_filter")
        filter_sentiment = sc2.selectbox("Filter by sentiment (optional)", ["All", "positive", "neutral", "negative"],
                                         index=0)
        filtered_df = df.copy()
        if filter_user != "All":
            filtered_df = filtered_df[filtered_df['user'] == filter_user]
        # Use uncleaned/original text for sentiment/emotion
        filtered_df = sentiment_analyzer.analyze_sentiment(filtered_df, text_col="message")
        if filter_sentiment != "All":
            filtered_df = filtered_df[filtered_df["sentiment"] == filter_sentiment]
        st.subheader("Sentiment Distribution")
        fig_s_dist = visualization.plot_sentiment_distribution(filtered_df)
        if fig_s_dist:
            st.plotly_chart(fig_s_dist, use_container_width=True)
        else:
            st.info("Not enough data to display sentiment distribution.")
        st.subheader("Sentiment Timeline")
        fig_s_time = visualization.plot_sentiment_timeline(filtered_df, freq="D")
        if fig_s_time:
            st.plotly_chart(fig_s_time, use_container_width=True)
        else:
            st.info("Not enough data to display sentiment timeline.")
        st.dataframe(filtered_df[["date", "user", "message", "sentiment", "sent_compound"]], use_container_width=True,
                     height=300)
        st.divider()

        # Emotion analysis (dynamic emotion filter)
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
                st.plotly_chart(fig_e_dist, use_container_width=True)
            else:
                st.info("Not enough data to display emotion distribution.")
            fig_e_time = visualization.plot_emotion_timeline(emo_df_filtered, freq="D", emotion_labels=filter_emotions)
            if fig_e_time:
                st.plotly_chart(fig_e_time, use_container_width=True)
            else:
                st.info("Not enough data to display emotion timeline.")
            st.dataframe(emo_df_filtered[["date", "user", "message", "emotion"]], use_container_width=True, height=300)

# --- Feedback Page ---
# --- Feedback Page ---
elif page == "Feedback":
    st.title("⭐ User Feedback")
    st.markdown("We'd love to hear your thoughts on the WhatsApp Chat Analyzer!")

    st.header("Submit Your Feedback")

    # Initialize session state for feedback inputs if not already present
    if 'feedback_name_input' not in st.session_state:
        st.session_state.feedback_name_input = ""
    if 'feedback_comment_input' not in st.session_state:
        st.session_state.feedback_comment_input = ""
    if 'feedback_rating_input' not in st.session_state:
        st.session_state.feedback_rating_input = 5  # Default rating


    # Define a callback function to handle submission and clear form
    def submit_feedback_callback():
        if not st.session_state.feedback_comment_input:
            st.warning("Please enter a comment before submitting feedback.")
            return  # Don't proceed if comment is empty

        current_time = datetime.now()  # Use datetime from the import at the top
        new_feedback = pd.DataFrame([{
            'name': st.session_state.feedback_name_input if st.session_state.feedback_name_input else "Anonymous",
            'rating': st.session_state.feedback_rating_input,
            'comment': st.session_state.feedback_comment_input,
            'timestamp': current_time
        }])

        df_feedback = load_feedback()
        df_feedback = pd.concat([df_feedback, new_feedback], ignore_index=True)
        save_feedback(df_feedback)

        st.success("Thank you for your feedback! 🙏")

        # Reset the input fields *after* processing, before the next rerun
        st.session_state.feedback_comment_input = ""
        st.session_state.feedback_name_input = ""
        st.session_state.feedback_rating_input = 5
        # No need for st.experimental_rerun() here as state changes will trigger rerun


    with st.form("feedback_form", clear_on_submit=False):
        # Now, the 'value' parameter for each widget directly reads from and writes to session_state
        name = st.text_input("Your Name (Optional)", key="feedback_name_input",
                             value=st.session_state.feedback_name_input)
        comment = st.text_area("Your Comment or Review", key="feedback_comment_input",
                               value=st.session_state.feedback_comment_input)
        rating = st.select_slider(
            "Star Rating",
            options=[1, 2, 3, 4, 5],
            value=st.session_state.feedback_rating_input,  # This is the crucial change
            format_func=lambda x: f"{x} Star{'s' if x > 1 else ''}",
            help="Rate your experience from 1 (Poor) to 5 (Excellent)",
            key="feedback_rating_input"
        )

        st.form_submit_button("Submit Feedback", on_click=submit_feedback_callback)

    st.header("View All Feedback")
    df_feedback = load_feedback()

    if df_feedback.empty:
        st.info("No feedback submitted yet. Be the first to share your thoughts!")
    else:
        col1, col2, col3 = st.columns([1, 1, 2])

        # Filter by Star Rating
        filter_rating = col1.selectbox(
            "Filter by Star Rating",
            options=["All", 1, 2, 3, 4, 5],
            index=0
        )

        # Sort by Date
        sort_order = col2.selectbox(
            "Sort By",
            options=["Newest First", "Oldest First"],
            index=0
        )

        # Optional "Top Feedback"
        top_feedback_only = col3.checkbox("Show only 4-5 Star Feedback")

        filtered_df = df_feedback.copy()

        if filter_rating != "All":
            filtered_df = filtered_df[filtered_df['rating'] == filter_rating]

        if top_feedback_only:
            filtered_df = filtered_df[filtered_df['rating'] >= 4]

        if sort_order == "Newest First":
            filtered_df = filtered_df.sort_values(by='timestamp', ascending=False)
        else:
            filtered_df = filtered_df.sort_values(by='timestamp', ascending=True)

        if filtered_df.empty:
            st.warning("No feedback matches your current filter criteria.")
        else:
            # Display feedback using st.dataframe
            st.dataframe(
                filtered_df[['timestamp', 'name', 'rating', 'comment']].style.format({
                    'timestamp': lambda dt: dt.strftime("%Y-%m-%d %H:%M"),
                    'rating': lambda r: '⭐' * r
                }),
                use_container_width=True,
                hide_index=True,
                height=400
            )
            st.caption(f"Displaying {len(filtered_df)} out of {len(df_feedback)} total feedback entries.")

st.markdown(
    '<hr><div style="text-align:center; color: #888;">'
    '© 2025 Pratham Bajpai & Contributors &nbsp;|&nbsp; '
    'Powered by Streamlit &nbsp;|&nbsp; '
    '<a href="https://github.com/Pratham-Bajpai1" target="_blank">GitHub</a>'
    '</div>',
    unsafe_allow_html=True
)