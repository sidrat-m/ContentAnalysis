import feedparser
import json
from datetime import datetime, timedelta
import urllib.parse
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download stopwords if not present
nltk.download("stopwords")

# =========================
# Load Bangla Stopwords
# =========================
with open("bangla_stopwords.txt", encoding="utf-8") as f:
    bangla_stopwords = set(f.read().split())

english_stopwords = set(stopwords.words("english"))
ALL_STOPWORDS = english_stopwords.union(bangla_stopwords)

# =========================
# Clean Text (Bangla + English)
# =========================
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s\u0980-\u09FF]", " ", text)
    text = text.lower()
    return text

# =========================
# News Scraper Function
# =========================
def scrape_news(
    keywords,
    max_results=20,
    days=10,
    output_file="bangla_english_news.json"
):
    """
    Scrape recent English + Bangla news and save to JSON
    """

    query = urllib.parse.quote(keywords)
    rss_url = (
        f"https://news.google.com/rss/search?"
        f"q={query}+when:{days}d"
        f"&hl=bn-BD&gl=BD&ceid=BD:bn"
    )

    feed = feedparser.parse(rss_url)
    articles = []

    for entry in feed.entries[:max_results]:
        text_content = entry.get("title", "") + " " + entry.get("summary", "")

        cleaned = clean_text(text_content)
        tokens = [
            w for w in cleaned.split()
            if w not in ALL_STOPWORDS and len(w) > 2
        ]

        articles.append({
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
            "source": entry.get("source", {}).get("title", "Unknown"),
            "summary": entry.get("summary", ""),
            "keywords_extracted": tokens[:20]
        })

    output_data = {
        "search_keyword": keywords,
        "language": "Bangla + English",
        "date_range_days": days,
        "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_articles": len(articles),
        "articles": articles
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(articles)} articles to '{output_file}'")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    keywords = "হুমাম কাদের"
    scrape_news(
        keywords=keywords,
        max_results=20,
        days=10,
        output_file="bangla_english_news.json"
    )
