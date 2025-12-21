import json
import re
import requests
from bs4 import BeautifulSoup
from collections import Counter
import plotly.express as px
from dash import Dash, dcc, html
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# =========================
# Load scraped news
# =========================
with open("bangla_english_news.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# =========================
# Function to fetch full text from a link
# =========================
def get_article_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        return text
    except:
        return ""

# =========================
# Clean Text Function
# =========================
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s\u0980-\u09FF]", " ", text)
    return text.lower()

# =========================
# Fetch and clean all article texts
# =========================
all_texts = []
for article in data["articles"]:
    full_text = get_article_text(article["link"])
    if full_text:
        all_texts.append(clean_text(full_text))
    else:
        all_texts.append(clean_text(article["summary"]))

combined_text = " ".join(all_texts)
tokens = [w for w in combined_text.split() if len(w) > 2]

# =========================
# Top 15 Words
# =========================
word_freq = Counter(tokens)
top_words = word_freq.most_common(15)
words, counts = zip(*top_words)
fig_wc = px.bar(
    x=words, y=counts, text=counts, title="Top 15 Words in Full Articles"
)
fig_wc.update_traces(textposition="outside")

# =========================
# Bangla BERT Sentiment Analysis
# =========================
model_name = "ahs95/banglabert-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentiments = []
for text in all_texts:
    result = nlp(text[:512])[0]  # truncate to 512 tokens
    sentiments.append(result["label"])

sentiment_counts = Counter(sentiments)
fig_sent = px.pie(
    names=list(sentiment_counts.keys()),
    values=list(sentiment_counts.values()),
    title="Sentiment Distribution of Full Articles"
)

# =========================
# Dash Dashboard
# =========================
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Bangla News Analysis Dashboard", style={'textAlign': 'center'}),
    html.Div([
        html.H2("Top 15 Words"),
        dcc.Graph(figure=fig_wc)
    ]),
    html.Div([
        html.H2("Sentiment Distribution"),
        dcc.Graph(figure=fig_sent)
    ])
])

if __name__ == "__main__":
    app.run(debug=True)

