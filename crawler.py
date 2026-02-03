
import requests
from collections import deque
import json
import time
import re
import logging

# --- CONFIGURATION ---
API_URL = "https://en.wikipedia.org/w/api.php"
# Seeds: Using article titles instead of URLs for the API
SEED_TITLES = ["Artificial intelligence", "Machine learning", "Python (programming language)", "History of the world", "Science"]
MAX_PAGES = 10000 
MAX_CHARS_PER_PAGE = 5000  # Space-saving: Truncate long articles
OUTPUT_FILE = "crawled_data.json"
USER_AGENT = "MySearchEngineCrawler/1.0 (your_email@example.com)" # PLEASE ADD YOUR EMAIL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def fetch_wiki_data(title, session):
    """Fetches text extract and outgoing links via Wikipedia API."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts|links",
        "explaintext": True,      # Get plain text instead of HTML
        "exlimit": "max",
        "pllimit": "max",         # Get as many links as possible (up to 500)
        "redirects": 1            # Follow redirects automatically
    }
    
    try:
        response = session.get(API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        page_id = next(iter(pages))
        page_data = pages[page_id]
        
        if "missing" in page_data:
            return None

        # Extracting the text and links
        title = page_data.get("title", "")
        text = page_data.get("extract", "")[:MAX_CHARS_PER_PAGE]
        
        # Filter links to keep only main-namespace articles (ns: 0)
        links = [l["title"] for l in page_data.get("links", []) if l.get("ns") == 0]
        
        return {
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "text": text,
            "links": links
        }
    except Exception as e:
        logger.warning(f"Failed to fetch {title}: {e}")
        return None

def crawl(seed_titles, max_pages):
    visited = set()
    queue = deque(seed_titles)
    pages = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    logger.info(f"Starting API crawl | Target: {max_pages} pages")

    while queue and len(pages) < max_pages:
        title = queue.popleft()

        if title in visited:
            continue
        visited.add(title)

        logger.info(f"[{len(pages) + 1}/{max_pages}] Fetching: {title}")

        # API is fast, but let's keep a tiny buffer to be polite
        # For 200 req/s, you'd remove this, but 0.05 is safer for a local script
        time.sleep(0.05) 
        
        result = fetch_wiki_data(title, session)
        if not result or not result["text"]:
            continue

        pages.append({
            "id": len(pages),
            **result
        })

        # Add discovered article titles to queue
        for linked_title in result["links"]:
            if linked_title not in visited:
                queue.append(linked_title)

    return pages

def save_pages(pages, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(pages)} pages to {filepath}")

if __name__ == "__main__":
    start_time = time.time()
    crawled_data = crawl(SEED_TITLES, MAX_PAGES)
    save_pages(crawled_data, OUTPUT_FILE)
    
    duration = time.time() - start_time
    print(f"\nCrawl finished in {duration:.2f} seconds.")