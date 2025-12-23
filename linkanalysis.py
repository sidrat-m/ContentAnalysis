import re
import requests
from bs4 import BeautifulSoup
from collections import Counter
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import pandas as pd
from dotenv import load_dotenv
import os
# =========================
# Load Bangla BERT Sentiment Model
# =========================
model_name = "ahs95/banglabert-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1 )
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
# =========================
# Load Bangla Stopwords
# =========================
with open("bangla_stopwords.txt", encoding="utf-8") as f:
    bangla_stopwords = set(f.read().split())

# =========================
# Utility Functions
# =========================
def get_news_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join(p.get_text() for p in paragraphs)
    except:
        return ""

def get_news_metadata(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")

        title = soup.title.get_text().strip() if soup.title else "No title found"

        images = [
            img.get("src")
            for img in soup.find_all("img")
            if img.get("src") and img.get("src").startswith("http")
        ]

        videos = [
            video.get("src")
            for video in soup.find_all("video")
            if video.get("src")
        ]

        return title, images[:5], videos[:3]
    except:
        return "No title", [], []

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s\u0980-\u09FF]", " ", text)
    return text.lower()

def text_statistics(text):
    tokens = clean_text(text).split()
    return len(tokens), len(set(tokens))

def top_words_figure(text):
    tokens = [
        w for w in clean_text(text).split()
        if len(w) > 2 and w not in bangla_stopwords
    ]

    freq = Counter(tokens).most_common(15)

    if not freq:
        return px.bar(title="No meaningful words found")

    words, counts = zip(*freq)
    fig = px.bar(
        x=words,
        y=counts,
        text=counts,
        title="Top 15 Meaningful Words (Stopwords Removed)"
    )
    fig.update_traces(textposition="outside")
    return fig

def sentiment_figure(text):
    if not text.strip():
        return px.pie(title="No text"), "No sentiment detected"

    result = nlp(text[:512])[0]
    label = result["label"]
    score = round(result["score"] * 100, 2)

    fig = px.pie(
        names=[label],
        values=[score],
        title=f"Sentiment: {label} ({score}%)"
    )

    confidence_text = f"📊 Sentiment Confidence Score: {score}%"
    return fig, confidence_text
def paragraph_sentiment_analysis(url):
    """
    Extract paragraphs and compute sentiment for each
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")

        paragraphs = [
            p.get_text().strip()
            for p in soup.find_all("p")
            if len(p.get_text().strip()) > 50  # ignore tiny lines
        ]

        results = []
        for para in paragraphs:
            cleaned = clean_text(para)
            if cleaned.strip():
                result = nlp(cleaned[:512])[0]
                results.append({
                    "text": para,
                    "label": result["label"],
                    "score": round(result["score"] * 100, 2)
                })

        return results

    except:
        return []
def sentiment_to_numeric(label, confidence):
    """
    Map sentiment labels to numeric range
    """
    label = label.lower()

    if label == "very negative":
        return -2 * confidence / 100
    elif label == "negative":
        return -1 * confidence / 100
    elif label == "neutral":
        return 0
    elif label == "positive":
        return 1 * confidence / 100
    elif label == "very positive":
        return 2 * confidence / 100
    else:
        return 0
def paragraph_sentiment_lineplot(paragraph_sentiments):
    if not paragraph_sentiments:
        return px.line(title="No paragraph sentiment data")

    x = []
    y = []
    labels = []

    for i, ps in enumerate(paragraph_sentiments):
        numeric_score = sentiment_to_numeric(ps["label"], ps["score"])
        x.append(f"P{i+1}")
        y.append(numeric_score)
        labels.append(f"{ps['label']} ({ps['score']}%)")

    fig = px.line(
        x=x,
        y=y,
        markers=True,
        title="📈 Paragraph-wise Sentiment Transition",
        labels={"x": "Paragraph", "y": "Sentiment Intensity"}
    )

    fig.update_traces(
        text=labels,
        textposition="top center"
    )

    fig.update_layout(
        yaxis=dict(
            range=[-2.2, 2.2],
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray'
        )
    )

    return fig
def extract_video_id(url):
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path[1:]
    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [None])[0]
    return None
def get_youtube_video_info(video_id):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    response = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()

    item = response["items"][0]
    snippet = item["snippet"]
    stats = item["statistics"]

    return {
        "title": snippet["title"],
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "comments": int(stats.get("commentCount", 0))
    }
def get_youtube_comments(video_id, max_comments=100):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_comments,
        textFormat="plainText"
    )

    response = request.execute()
    for item in response["items"]:
        text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(text)

    return comments
def analyze_comments(comments):
    """
    Returns:
    - sentiment_counts (dict)
    - comment_results (list)
    """

    sentiment_map = {
        "very negative": "negative",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
        "very positive": "positive"
    }

    counts = {
        "positive": 0,
        "neutral": 0,
        "negative": 0
    }

    comment_results = []

    for comment in comments:
        if len(comment.strip()) < 5:
            continue

        result = nlp(comment[:512])[0]

        raw_label = result["label"].lower()
        score = round(result["score"] * 100, 2)

        # 🔁 Normalize label
        label = sentiment_map.get(raw_label, "neutral")

        counts[label] += 1

        comment_results.append({
            "text": comment,
            "length": len(comment.split()),
            "sentiment": label,
            "raw_sentiment": raw_label,
            "confidence": score
        })

    return counts, comment_results


def comment_sentiment_bar(counter):
    return px.bar(
        x=list(counter.keys()),
        y=list(counter.values()),
        title="Comment Sentiment Distribution",
        labels={"x": "Sentiment", "y": "Count"}
    )
def comment_tone_pie(counter):
    return px.pie(
        names=list(counter.keys()),
        values=list(counter.values()),
        title="Comment Tone Analysis"
    )
def comment_length_vs_sentiment(comment_results):
    if not comment_results:
        return px.scatter(title="No comment data")

    df = pd.DataFrame(comment_results)

    sentiment_map = {
        "negative": -1,
        "neutral": 0,
        "positive": 1
    }

    df["sentiment_value"] = df["sentiment"].map(sentiment_map)

    fig = px.scatter(
        df,
        x="length",
        y="sentiment_value",
        color="sentiment",
        size="confidence",
        title="Comment Length vs Sentiment",
        labels={
            "length": "Comment Length (words)",
            "sentiment_value": "Sentiment Scale"
        }
    )

    fig.update_yaxes(
        tickvals=[-1, 0, 1],
        ticktext=["Negative", "Neutral", "Positive"]
    )

    return fig
# =========================
# Dash App
# =========================
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Content Analysis Dashboard", style={
                'textAlign': 'center',
                'color': "#0D092CFD",   # Light gray title
                'marginBottom': '30px'
            }),

    html.Div([
        html.Label("Select Link Type:"),
        dcc.Dropdown(
            id="link-type",
            options=[
                {"label": "📰 News Link", "value": "news"},
                {"label": "📘 Facebook Post", "value": "facebook"},
                {"label": "▶️ YouTube Link", "value": "youtube"},
            ],
            value="news",
            style={
                    'width': '100%',
                    'backgroundColor': "#F2EDFE44",  # light gray dropdown
                    'color': '#000000',
                    'marginTop': '10px'
                }
        ),
        html.Br(),
        html.Label("Enter Link:"),
        dcc.Input(id="input-link", type="text", style={
                    'width': '70%',
                    'padding': '5px',
                    'borderRadius': '5px',
                    'border': '1px solid #ccc',
                    'marginLeft': '10px',
                    'marginRight': '10px'
                }),
        
        html.Button("Process", id="process-btn", n_clicks=0,
                style={
                    'backgroundColor': '#0f3460',  # darker navy
                    'color': '#ffffff',
                    'border': 'none',
                    'padding': '10px 20px',
                    'borderRadius': '5px',
                    'cursor': 'pointer'
                })
    ]),

    html.Hr(style={'borderColor': '#e0e0e0'}),
    html.Div(id="dynamic-dashboard")
])

# =========================
# Callback
# =========================
@app.callback(
    Output("dynamic-dashboard", "children"),
    Input("process-btn", "n_clicks"),
    State("input-link", "value"),
    State("link-type", "value")
)

def process_link(n_clicks, link, link_type):
    if n_clicks == 0 or not link:
        return html.P("Please enter a link and click Process.")

    # ================= NEWS =================
    if link_type == "news":
        text = get_news_text(link)
        title, images, videos = get_news_metadata(link)
        total_words, unique_words = text_statistics(text)
        fig_sent, confidence = sentiment_figure(text)

        return html.Div([

            html.H2(title),

            html.Div([
                html.P(f"📏 News Length (characters): {len(text)}"),
                html.P(f"📝 Total Words: {total_words}"),
                html.P(f"🔑 Unique Words: {unique_words}")
            ], style={
                'border': '1px solid #ddd',
                'padding': '10px',
                'marginBottom': '20px',
                'backgroundColor': "#7C46FA0F",
            }),

            html.H4("Attachments"),
            html.Div([
                html.Img(src=img, style={'width': '200px', 'margin': '5px'})
                for img in images
            ]),

            html.Hr(),

            html.H4("Full News Content"),
            html.Div(
                text,
                style={
                    'whiteSpace': 'pre-wrap',
                    'maxHeight': '400px',
                    'overflowY': 'scroll',
                    'border': '1px solid #ccc',
                    'padding': '10px'
                }
            ),

            html.Button(
                "🔍 Show Paragraph-wise Analysis",
                id="para-btn",
                n_clicks=0,
                style={
                    'backgroundColor': '#6a0dad',
                    'color': '#ffffff',
                    'border': 'none',
                    'padding': '10px 15px',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'marginBottom': '20px'
                }
            ),

            html.Div(id="paragraph-analysis-container"),

            html.Hr(),

            dcc.Graph(figure=top_words_figure(text)),

            html.H4("Sentiment Analysis"),
            dcc.Graph(figure=fig_sent),
            html.P(confidence, style={'fontWeight': 'bold', 'color': 'green'})
        ])

    # ================= FACEBOOK =================
    elif link_type == "facebook":
        fig_sent, confidence = sentiment_figure(link)
        return html.Div([
            html.H3("Facebook Post Analysis"),
            dcc.Graph(figure=fig_sent),
            html.P(confidence)
        ])

    # ================= YOUTUBE =================
    elif link_type == "youtube":
        video_id = extract_video_id(link)
        if not video_id:
            return html.P("Invalid YouTube link")

        info = get_youtube_video_info(video_id)
        comments = get_youtube_comments(video_id)

        sentiment_counts, comment_results = analyze_comments(comments)

        fig_bar = comment_sentiment_bar(sentiment_counts)
        fig_pie = comment_tone_pie(sentiment_counts)
        fig_scatter = comment_length_vs_sentiment(comment_results)

        return html.Div([

            html.H2(info["title"]),

            html.Div([
                html.P(f"👀 Views: {info['views']}"),
                html.P(f"👍 Likes: {info['likes']}"),
                html.P("👎 Dislikes: Not Public"),
                html.P(f"💬 Total Comments: {info['comments']}")
            ], style={
                'border': '1px solid #ddd',
                'padding': '10px',
                'marginBottom': '20px',
                'backgroundColor': "#7C46FA0F",
            }),

            html.Hr(),

            html.H3("📊 Comment Sentiment Analysis"),
            dcc.Graph(figure=fig_bar),

            html.H3("🎭 Comment Tone Analysis"),
            dcc.Graph(figure=fig_pie),

            html.H3("📈 Comment Length vs Sentiment"),
            dcc.Graph(figure=fig_scatter)
        ])
@app.callback(
    Output("paragraph-analysis-container", "children"),
    Input("para-btn", "n_clicks"),
    State("input-link", "value"),
    prevent_initial_call=True
)
def load_paragraph_analysis(n_clicks, link):
    if not link:
        return html.P("No link provided.")

    paragraph_sentiments = paragraph_sentiment_analysis(link)

    if not paragraph_sentiments:
        return html.P("No paragraph sentiment data found.")

    return html.Div([
        html.H3("📑 Paragraph-wise Sentiment Analysis"),

        html.Div([
            html.Div([
                html.P(f"🧾 Paragraph {i+1}", style={'fontWeight': 'bold'}),
                html.P(ps["text"]),
                html.P(
                    f"Sentiment: {ps['label']} | {ps['score']}%",
                    style={'fontWeight': 'bold'}
                ),
                html.Hr()
            ], style={
                'padding': '10px',
                'border': '1px solid #ddd',
                'marginBottom': '10px'
            })
            for i, ps in enumerate(paragraph_sentiments)
        ]),

        dcc.Graph(
            figure=paragraph_sentiment_lineplot(paragraph_sentiments)
        )
    ])


# =========================
# Run
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)
