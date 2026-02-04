# 🔍 MiniSearch

A lightning-fast Wikipedia search engine built from scratch with Python, featuring BM25 ranking, AI-powered summaries, and support for phrase queries and boolean operators.

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.1-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

## ✨ Features

- **🚀 Fast BM25 Ranking**: Industry-standard probabilistic ranking algorithm for relevant search results
- **🤖 AI Summaries**: Get instant overviews of search results powered by Groq's Llama 3.3 70B
- **📝 Advanced Query Syntax**:
  - Simple multi-word queries: `neural networks`
  - Phrase search: `"machine learning"`
  - Boolean operators: `python AND (learning OR neural) NOT robotics`
- **📚 10,000 Wikipedia Articles**: Pre-indexed and ready to search
- **⚡ Inverted Index**: Sub-second query response times
- **🎨 Clean UI**: Modern, responsive interface with real-time results

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │  Static HTML/CSS/JS
│   (index.html)  │
└────────┬────────┘
         │
         │ HTTP API
         ▼
┌─────────────────┐
│   Flask App     │  Search endpoint + routing
│   (app.py)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Query Engine   │  BM25 scoring, query parsing
│(query_engine.py)│  AI summary generation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Inverted Index  │  Term → Doc mappings
│  (index.json)   │  TF-IDF, positions
└─────────────────┘
```

## 🛠️ Tech Stack

- **Backend**: Python 3.13, Flask, Gunicorn
- **AI**: Groq API (Llama 3.3 70B)
- **Search**: Custom BM25 implementation with Porter Stemmer
- **Data**: 10,000 Wikipedia articles (crawled via Wikipedia API)
- **Deployment**: Railway (or any cloud platform)

## 📦 Installation

### Prerequisites

- Python 3.13+
- pip
- (Optional) Groq API key for AI summaries

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/minisearch.git
   cd minisearch
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional, for AI summaries)
   ```bash
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```
   Get your free API key at [console.groq.com](https://console.groq.com)

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

## 🚀 Deployment

### Deploy to Railway

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin your-repo-url
   git push -u origin main
   ```

2. **Connect to Railway**
   - Go to [railway.app](https://railway.app)
   - Create new project → Deploy from GitHub
   - Select your repository
   - Add environment variable: `GROQ_API_KEY` (if using AI summaries)

3. **Done!** Railway will auto-deploy your app

### Deploy to Other Platforms

The app works on any platform that supports Python web apps:
- **Render**: Add `Procfile` and deploy
- **Heroku**: Standard Python deployment
- **AWS/GCP**: Use Elastic Beanstalk or App Engine

## 📖 Usage

### Simple Search
```
neural networks
```
Returns all documents containing both "neural" and "networks"

### Phrase Search
```
"machine learning"
```
Finds exact phrase matches where words appear consecutively

### Boolean Queries
```
python AND (learning OR neural) NOT robotics
```
Supports AND, OR, NOT operators with parentheses for grouping

### AI Summaries
AI-powered overviews are automatically generated for every search when `GROQ_API_KEY` is configured. Disable by adding `?summary=false` to the search URL.

## 🗂️ Project Structure

```
minisearch/
├── app.py                 # Flask application entry point
├── query_engine.py        # Search engine core (BM25, AI summaries)
├── indexer.py            # Inverted index builder + Porter stemmer
├── crawler.py            # Wikipedia data crawler
├── requirements.txt      # Python dependencies
├── Procfile             # Deployment configuration
├── static/
│   └── index.html       # Frontend UI
├── crawled_data.json    # Raw Wikipedia articles (10,000 pages)
└── index.json           # Inverted index (generated from crawler)
```

## 🔧 How It Works

### 1. Data Collection (`crawler.py`)
- Crawls Wikipedia API starting from seed topics
- Fetches article text and outbound links
- Saves 10,000 articles to `crawled_data.json`

### 2. Indexing (`indexer.py`)
- Tokenizes text (lowercase, remove stopwords, stem)
- Builds inverted index: `term → {doc_freq, postings}`
- Stores term frequencies and positions for phrase search
- Saves to `index.json`

### 3. Query Processing (`query_engine.py`)
- Parses user query (simple/phrase/boolean)
- Scores documents using BM25 algorithm
- Generates snippets with highlighted terms
- Optionally creates AI summary via Groq API

### 4. Serving Results (`app.py`)
- Flask endpoint `/search?q=...&summary=true`
- Returns JSON with ranked results + AI overview
- Frontend renders results in real-time

## 🧮 BM25 Algorithm

MiniSearch uses the BM25 ranking function:

```
score(D,Q) = Σ IDF(qᵢ) · (f(qᵢ,D) · (k₁ + 1)) / (f(qᵢ,D) + k₁ · (1 - b + b · |D| / avgdl))
```

Where:
- `IDF(qᵢ)` = Inverse document frequency of term qᵢ
- `f(qᵢ,D)` = Frequency of qᵢ in document D
- `|D|` = Length of document D
- `avgdl` = Average document length
- `k₁` = 1.5 (term frequency saturation)
- `b` = 0.75 (length normalization)

## 🤖 AI Summaries

Powered by **Groq's Llama 3.3 70B Versatile** model:
- Synthesizes top 3 search results
- Generates concise 150-word overviews
- Sub-second response times
- Falls back gracefully if API unavailable

## 📊 Performance

- **Index Size**: 10,000 documents, 45,000+ unique terms
- **Query Time**: ~50-200ms (excluding AI summary)
- **AI Summary Time**: ~500-1000ms (via Groq)
- **Memory Usage**: ~150MB (index loaded in RAM)

## 🔐 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | No | API key for AI summaries (get free at console.groq.com) |
| `PORT` | No | Server port (default: 5000, auto-assigned on Railway) |

## 🧪 Testing

Run the search engine REPL for interactive testing:

```bash
python query_engine.py
```

Example session:
```
🔎 Search: artificial intelligence
  #1  [12.34]  Artificial Intelligence
      https://en.wikipedia.org/wiki/Artificial_intelligence
      ...AI is the simulation of human **intelligence** processes...
```

## 🛣️ Roadmap

- [ ] Add autocomplete suggestions
- [ ] Implement query spell-checking
- [ ] Support for filters (date, category)
- [ ] User search history
- [ ] PDF export of results
- [ ] Multi-language support

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Wikipedia** for providing the data via their API
- **Groq** for fast AI inference
- **Flask** for the lightweight web framework
- **Porter Stemmer** algorithm for text normalization

