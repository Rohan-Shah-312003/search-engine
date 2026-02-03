import requests
from collections import deque
import json
import time
import logging

# --- CONFIGURATION ---
API_URL = "https://en.wikipedia.org/w/api.php"
SEED_TITLES = [
    "Artificial intelligence",
    "Machine learning",
    "Python (programming_language)",
    "History of the world",
    "Science",
]
MAX_PAGES = 10000
BATCH_SIZE = 20  # Wikipedia API limit for multi-page queries
MAX_CHARS_PER_PAGE = 5000
OUTPUT_FILE = "crawled_data.json"
USER_AGENT = "MyFastSearchCrawler/1.0 (your_email@example.com)"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_batch_data(titles, session):
    """Fetches text and links for a batch. Capped at 20 for 'extracts' reliability."""
    params = {
        "action": "query",
        "format": "json",
        "titles": "|".join(titles),
        "prop": "extracts|links",
        "explaintext": True,
        "exintro": True,  # Gets the intro section (saves massive space!)
        "exlimit": "20",  # Maximum allowed for extracts
        "pllimit": "max",
        "redirects": 1,
    }

    try:
        response = session.get(API_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        results = []
        pages = data.get("query", {}).get("pages", {})

        for page_id, page_data in pages.items():
            # Ensure the page actually has content and isn't a "Missing" page
            if int(page_id) < 0 or "extract" not in page_data:
                continue

            title = page_data.get("title", "")
            results.append(
                {
                    "title": title,
                    "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    "text": page_data.get("extract", "")[:MAX_CHARS_PER_PAGE],
                    "links": [
                        l["title"]
                        for l in page_data.get("links", [])
                        if l.get("ns") == 0
                    ],
                }
            )
        return results
    except Exception as e:
        logger.warning(f"Batch fetch failed: {e}")
        return []


def crawl(seed_titles, max_pages):
    visited = set()
    queue = deque(seed_titles)
    all_pages = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    logger.info(f"🚀 Starting Batch Crawl | Target: {max_pages} pages")

    while queue and len(all_pages) < max_pages:
        # Get up to BATCH_SIZE unvisited titles from the queue
        current_batch = []
        while queue and len(current_batch) < BATCH_SIZE:
            t = queue.popleft()
            if t not in visited:
                visited.add(t)
                current_batch.append(t)

        if not current_batch:
            continue

        logger.info(
            f"Fetching batch of {len(current_batch)} | Total collected: {len(all_pages)}"
        )

        # Respectful delay between batch calls
        time.sleep(0.2)

        batch_results = fetch_batch_data(current_batch, session)

        for res in batch_results:
            if len(all_pages) >= max_pages:
                break

            res["id"] = len(all_pages)
            all_pages.append(res)

            # Add new links to the queue
            for linked_title in res["links"]:
                if linked_title not in visited:
                    queue.append(linked_title)

    return all_pages


if __name__ == "__main__":
    start_time = time.time()
    data = crawl(SEED_TITLES, MAX_PAGES)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    duration = time.time() - start_time
    logger.info(f"✅ Done! Collected {len(data)} pages in {duration:.2f}s")
