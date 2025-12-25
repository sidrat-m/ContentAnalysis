# Content Analysis Dashboard

A **Dash-based interactive dashboard** for analyzing content from **News articles, Facebook posts, and YouTube videos**.  
It supports text extraction, sentiment analysis, word frequency analysis, and visualization of comments and paragraph-level sentiment.

---

## Features

- **News Link Analysis**
  - Extracts full text, images, and videos from news URLs.
  - Computes total and unique word counts.
  - Visualizes top meaningful words (stopwords removed).
  - Performs sentiment analysis on full content and individual paragraphs.

- **Facebook Post Analysis**
  - Sentiment analysis of a given Facebook post or link.
  - Provides an interactive visualization of sentiment results.

- **YouTube Video Analysis**
  - Extracts video metadata: views, likes, comments.
  - Scrapes up to 100 comments per video.
  - Performs sentiment analysis on comments.
  - Visualizes comment sentiment distribution, tone, and length vs sentiment.

- Uses **Bangla BERT model** (`ahs95/banglabert-sentiment-analysis`) for sentiment detection on Bangla text.
- Clean, interactive **Dash web interface** for easy input and visualization.

---

## Installation

pip install -r requirements.txt
**Add your API keys in a .env file:**
YOUTUBE_API_KEY=YOUR_YOUTUBE_API_KEY


Run the app:
python linkanalysis.py

Open your browser at http://127.0.0.1:8050.

- **Steps on the dashboard:**
  -Select the type of link: News, Facebook, or YouTube.
  
  -Enter the URL in the input box.
  
  -Click Process to analyze content.
  
  -For news articles, optionally click Show Paragraph-wise Analysis to see paragraph sentiment transition.
