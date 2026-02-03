import requests
from bs4 import BeautifulSoup
from collections import deque
import json
import time
import re
import os
import logging

BASE_URL      = "https://en.wikipedia.org"
SEED_URLS     = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
]
MAX_PAGES     = 50          # Total pages to crawl
CRAWL_DELAY   = 1.0         # Seconds between requests (be polite!)
OUTPUT_FILE   = "crawled_data.json"
USER_AGENT    = "MySearchEngineCrawler/1.0 (Educational Project)"

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# CORE FUNCTIONS

def is_valid_wiki_url(url: str) -> bool:
    if not url.startswith("/wiki/"):
        return False

    # Reject anything that isn't a regular article
    excluded_prefixes = (
        "/wiki/Special:",
        "/wiki/Talk:",
        "/wiki/File:",
        "/wiki/Template:",
        "/wiki/Help:",
        "/wiki/Wikipedia:",
        "/wiki/Portal:",
        "/wiki/Category:",
        "/wiki/User:",
        "/wiki/User_talk:",
        "/wiki/Module:",
        "/wiki/Draft:",
    )
    return not url.startswith(excluded_prefixes)

# extract <p> content
def extract_text(soup: BeautifulSoup) -> str:
    content_div = soup.find("div", {"class": "mw-parser-output"})
    if not content_div:
        return ""

    # Grab only <p> tags — the actual prose of the article
    paragraphs = content_div.find_all("p")
    text = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)

    # Clean up whitespace artifacts
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_title(soup: BeautifulSoup) -> str:
    title_tag = soup.find("title")
    if title_tag:
        # Wikipedia titles look like "Topic - Wikipedia" — strip the suffix
        return title_tag.get_text().replace(" - Wikipedia", "").strip()
    return "Untitled"


def extract_links(soup: BeautifulSoup) -> list[str]:
    content_div = soup.find("div", {"class": "mw-parser-output"})
    if not content_div:
        return []

    links = []
    for a_tag in content_div.find_all("a", href=True):
        href = a_tag["href"]
        if is_valid_wiki_url(href):
            links.append(BASE_URL + href)

    return links


def fetch_page(url: str, session: requests.Session) -> BeautifulSoup | None:
    """Fetches a page and returns its parsed BeautifulSoup, or None on failure."""
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, "lxml")
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def crawl(seed_urls: list[str], max_pages: int) -> list[dict]: # using bfs
    visited   = set()
    queue     = deque(seed_urls)
    pages     = []

    session   = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    logger.info(f"Starting crawl | max_pages={max_pages} | seeds={len(seed_urls)}")

    while queue and len(pages) < max_pages:
        url = queue.popleft()

        # Skip if already visited or not a valid wiki page
        if url in visited:
            continue
        visited.add(url)

        logger.info(f"[{len(pages)+1}/{max_pages}] Crawling: {url}")

        soup = fetch_page(url, session)
        if soup is None:
            continue

        title = extract_title(soup)
        text  = extract_text(soup)
        links = extract_links(soup)

        # Skip pages with no meaningful content
        if not text:
            logger.warning(f"  Skipping (no text): {title}")
            continue

        pages.append({
            "id":    len(pages),
            "title": title,
            "url":   url,
            "text":  text,
            "links": links,
        })

        # Add discovered links to the queue
        for link in links:
            if link not in visited:
                queue.append(link)

        # Polite crawl delay
        time.sleep(CRAWL_DELAY)

    logger.info(f"Crawl complete. Collected {len(pages)} pages.")
    return pages

def save_pages(pages: list[dict], filepath: str):
    """Saves crawled data to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(pages)} pages → {filepath}")


if __name__ == "__main__":
    pages = crawl(SEED_URLS, MAX_PAGES)
    save_pages(pages, OUTPUT_FILE)

    # Quick preview of what was collected
    print("\n── Preview ──")
    for page in pages[:3]:
        print(f"  [{page['id']}] {page['title']}")
        print(f"       URL   : {page['url']}")
        print(f"       Text  : {page['text'][:120]}...")
        print(f"       Links : {len(page['links'])} outgoing")
        print()