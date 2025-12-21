import feedparser
import json
from datetime import datetime
import urllib.parse


def scrape_news(keywords, days=10, max_results=10, output_file="news_results.json"):

    # Add date range filter
    query_with_date = f"{keywords} when:{days}d"
    query = urllib.parse.quote(query_with_date)

    rss_urls = {
        "english": f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
        "bangla":  f"https://news.google.com/rss/search?q={query}&hl=bn&gl=BD&ceid=BD:bn"
    }

    articles = []
    seen_links = set()

    for lang, url in rss_urls.items():
        feed = feedparser.parse(url)

        for entry in feed.entries[:max_results]:
            link = entry.get("link", "")

            if link in seen_links:
                continue
            seen_links.add(link)

            article = {
                "language": lang,
                "title": entry.get("title", ""),
                "link": link,
                "published": entry.get("published", ""),
                "source": entry.get("source", {}).get("title", "Unknown"),
                "summary": entry.get("summary", "")
            }
            articles.append(article)

    output_data = {
        "keyword": keywords,
        "date_range_days": days,
        "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_articles": len(articles),
        "articles": articles
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(articles)} articles from last {days} days to {output_file}")




# =============================
# ▶ Run
# =============================
if __name__ == "__main__":
    keywords = "হুম্মাম কাদের"
    scrape_news(keywords, days=10, max_results=15)
