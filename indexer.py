import json
import math
import re
import logging
from collections import defaultdict

# CONFIG
CRAWLED_DATA_FILE = "crawled_data.json"
INDEX_FILE = "index.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# STOPWORDS  (common English words to ignore)
STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "d",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "just",
    "ll",
    "m",
    "ma",
    "me",
    "mightn",
    "mightn't",
    "more",
    "most",
    "mustn",
    "mustn't",
    "my",
    "myself",
    "needn",
    "needn't",
    "no",
    "nor",
    "not",
    "now",
    "o",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "re",
    "s",
    "same",
    "shan",
    "shan't",
    "she",
    "she's",
    "should",
    "should've",
    "shouldn",
    "shouldn't",
    "so",
    "some",
    "such",
    "t",
    "than",
    "that",
    "that'll",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "ve",
    "very",
    "was",
    "wasn",
    "wasn't",
    "we",
    "were",
    "weren",
    "weren't",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
    "y",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
    # extras that show up a lot in Wikipedia
    "also",
    "one",
    "two",
    "new",
    "like",
    "many",
    "may",
    "would",
    "could",
    "use",
    "using",
    "used",
    "much",
    "well",
    "even",
    "still",
    "known",
    "often",
    "however",
    "though",
    "another",
    "every",
    "since",
    "first",
    "last",
    "around",
    "between",
    "between",
    "called",
    "often",
    "used",
    "based",
    "became",
    "according",
    "although",
    "including",
    "several",
    "various",
    "being",
    "been",
    "within",
}


# PORTER STEMMER (simplified)
# A hand-rolled implementation of the core Porter Stemmer rules.
# Handles the most common English suffixes — good enough for a search engine.


def _ends_with(word, suffix):
    return word.endswith(suffix) and len(word) > len(suffix)


def _measure(word):
    """
    Counts the number of VC (vowel-consonant) sequences in a word.
    e.g.  "tr" → 0, "ee" → 0, "tree" → 1, "oats" → 1, "trees" → 1
    """
    vowels = set("aeiouy")
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if prev_vowel and not is_vowel:
            count += 1
        prev_vowel = is_vowel
    return count


def _has_vowel(word):
    return any(c in "aeiouy" for c in word)


def stem(word: str) -> str:
    """Applies simplified Porter stemming rules to a single word."""
    if len(word) < 3:
        return word

    # Step 1: Plurals & past tenses
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass  # "caress" stays
    elif word.endswith("s") and not word.endswith("us") and not word.endswith("ss"):
        word = word[:-1]

    # ── Step 2: "-ed" / "-ing" ───────────────────────────
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]  # "agreed" → "agree"
    elif word.endswith("ed"):
        stem_part = word[:-2]
        if _has_vowel(stem_part):
            word = stem_part
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word += "e"
            elif len(word) >= 2 and word[-1] == word[-2] and word[-1] not in "lsz":
                word = word[:-1]
    elif word.endswith("ing"):
        stem_part = word[:-3]
        if _has_vowel(stem_part):
            word = stem_part
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word += "e"
            elif len(word) >= 2 and word[-1] == word[-2] and word[-1] not in "lsz":
                word = word[:-1]

    # Step 3: "-y" → "-i" when preceded by a vowel
    if word.endswith("y") and len(word) > 2 and _has_vowel(word[:-1]):
        word = word[:-1] + "i"

    # Step 4: Common suffixes
    suffixes_step4 = [
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("ator", "ate"),
        ("alli", "al"),
        ("ousli", "ous"),
        ("entli", "ent"),
        ("eli", "e"),
        ("fulness", "ful"),
        ("iveness", "ive"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ness", ""),
        ("ment", ""),
    ]
    for suffix, replacement in suffixes_step4:
        if word.endswith(suffix) and _measure(word[: -len(suffix)]) > 0:
            word = word[: -len(suffix)] + replacement
            break

    # Step 5: Final cleanup
    if word.endswith("e") and _measure(word[:-1]) > 1:
        word = word[:-1]
    if word.endswith("l") and word[-2:] == "ll" and _measure(word[:-1]) > 1:
        word = word[:-1]

    return word


# TEXT PROCESSING PIPELINE

def tokenize(text: str) -> list[str]:
    """
    Lowercase → strip punctuation → split into words → remove stopwords → stem.
    Returns a list of processed tokens.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)  # keep only alphanumeric runs
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [stem(t) for t in tokens]
    return tokens


# INVERTED INDEX BUILDER
# 
# Structure of the inverted index:
#
#   {
#     "metadata": {
#         "num_docs": <int>,
#         "avg_doc_length": <float>
#     },
#     "doc_lengths": { "<doc_id>": <int>, ... },      ← token count per doc
#     "index": {
#         "<term>": {
#             "doc_freq": <int>,                       ← how many docs contain this term
#             "postings": {
#                 "<doc_id>": {
#                     "term_freq": <int>,              ← how many times the term appears
#                     "positions": [<int>, ...]        ← character positions (for phrase search later)
#                 },
#                 ...
#             }
#         },
#         ...
#     }
#   }


def build_index(pages: list[dict]) -> dict:
    """
    Takes the crawled pages list and builds the full inverted index.
    """
    num_docs = len(pages)
    index = {}  # term → { doc_freq, postings }
    doc_lengths = {}  # doc_id → number of tokens

    for page in pages:
        doc_id = str(page["id"])
        tokens = tokenize(page["text"])
        doc_lengths[doc_id] = len(tokens)

        # Count term frequency and track positions within this doc
        term_positions = defaultdict(list)  # term → [pos0, pos1, ...]
        for pos, token in enumerate(tokens):
            term_positions[token].append(pos)

        # Merge into the global index
        for term, positions in term_positions.items():
            if term not in index:
                index[term] = {"doc_freq": 0, "postings": {}}

            index[term]["doc_freq"] += 1
            index[term]["postings"][doc_id] = {
                "term_freq": len(positions),
                "positions": positions,
            }

        logger.info(
            f"  Indexed doc {doc_id}: '{page['title']}' ({len(tokens)} tokens, {len(term_positions)} unique)"
        )

    avg_doc_length = sum(doc_lengths.values()) / num_docs if num_docs else 0

    full_index = {
        "metadata": {"num_docs": num_docs, "avg_doc_length": round(avg_doc_length, 2)},
        "doc_lengths": doc_lengths,
        "index": index,
    }

    logger.info(f"Index built: {num_docs} docs, {len(index)} unique terms")
    return full_index


# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────


def save_index(full_index: dict, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(full_index, f, ensure_ascii=False, indent=2)
    logger.info(f"Index saved → {filepath}")


def load_index(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"Loading crawled data from {CRAWLED_DATA_FILE}...")
    with open(CRAWLED_DATA_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

    if not pages:
        logger.warning("No pages found in crawled data. Run crawler.py first.")
    else:
        full_index = build_index(pages)
        save_index(full_index, INDEX_FILE)
