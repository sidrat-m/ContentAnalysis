import json
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from collections import Counter
from nltk.corpus import stopwords
from textblob import TextBlob


# ==========================
# Load Data
# ==========================
with open("news_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

articles = data["articles"]
df = pd.DataFrame(articles)

st.set_page_config(page_title="News Visualization Dashboard", layout="wide")
st.title("📰 News Analysis Dashboard")

# ==========================
# Dropdown Selection
# ==========================
selected_title = st.selectbox(
    "Select a News Article",
    df["title"].tolist()
)

selected_article = df[df["title"] == selected_title].iloc[0]

st.subheader("📌 Selected News")
st.markdown(f"**Source:** {selected_article['source']}")
st.markdown(f"**Published:** {selected_article['published']}")
st.markdown(f"**Language:** {selected_article['language']}")
st.markdown(f"[🔗 Read Full News]({selected_article['link']})")

# ==========================
# Text Cleaning
# ==========================
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^A-Za-zঅ-হ ]+", " ", text)
    text = text.lower()
    return text


cleaned_text = clean_text(selected_article["title"] + " " + selected_article["summary"])

words = cleaned_text.split()

# English stopwords only (Bangla will stay untouched)
stop_words = set(stopwords.words("english"))
filtered_words = [w for w in words if w not in stop_words and len(w) > 2]

word_freq = Counter(filtered_words).most_common(15)

# ==========================
# Word Frequency Chart
# ==========================
freq_df = pd.DataFrame(word_freq, columns=["Word", "Frequency"])

fig_words = px.bar(
    freq_df,
    x="Word",
    y="Frequency",
    title="Top 15 Most Used Words",
    text="Frequency"
)

# ==========================
# Sentiment Analysis
# ==========================
sentiment_score = TextBlob(cleaned_text).sentiment.polarity

if sentiment_score > 0.1:
    sentiment_label = "Positive"
elif sentiment_score < -0.1:
    sentiment_label = "Negative"
else:
    sentiment_label = "Neutral"

# Sentiment Gauge
fig_sentiment = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        title={"text": f"Sentiment: {sentiment_label}"},
        gauge={
            "axis": {"range": [-1, 1]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [-1, -0.1], "color": "red"},
                {"range": [-0.1, 0.1], "color": "gray"},
                {"range": [0.1, 1], "color": "green"},
            ],
        },
    )
)

# ==========================
# Layout
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_words, use_container_width=True)

with col2:
    st.plotly_chart(fig_sentiment, use_container_width=True)

# ==========================
# Overall News Statistics
# ==========================
st.subheader("📊 Overall News Statistics")

language_counts = df["language"].value_counts().reset_index()
language_counts.columns = ["Language", "Count"]

fig_lang = px.pie(
    language_counts,
    names="Language",
    values="Count",
    title="News Language Distribution"
)

st.plotly_chart(fig_lang, use_container_width=True)
