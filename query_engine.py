import json
import math
import re
import logging
from collections import defaultdict
from indexer import tokenize, load_index

# CONFIG
INDEX_FILE = "index.json"
CRAWLED_DATA_FILE = "crawled_data.json"
TOP_K = 5  # number of results to return by default
SNIPPET_LENGTH = 200  # max chars in the snippet shown per result

# BM25 tuning knobs
BM25_K1 = 1.5  # term-frequency saturation. Higher -> longer docs get more credit for repeated terms
BM25_B = 0.75  # length normalisation. 0 = ignore doc length, 1 = full normalisation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# DATA LOADING  (cached at module level)
_index_cache = None
_pages_cache = None  # list of crawled page dicts, keyed later by id
_pages_by_id = None  # { "0": {title, url, text, ...}, ... }


def _ensure_loaded():
    """Lazy-loads index + crawled pages exactly once."""
    global _index_cache, _pages_cache, _pages_by_id
    if _index_cache is not None:
        return
    _index_cache = load_index(INDEX_FILE)
    with open(CRAWLED_DATA_FILE, "r", encoding="utf-8") as f:
        _pages_cache = json.load(f)
    _pages_by_id = {str(p["id"]): p for p in _pages_cache}
    logger.info(
        "Loaded index (%d docs, %d terms) + crawled pages",
        _index_cache["metadata"]["num_docs"],
        len(_index_cache["index"]),
    )


# QUERY PARSER
# 
# Supports three syntaxes the user can type:
#
#   Plain multi-word   →  neural networks          (implicit AND)
#   Phrase (quotes)    →  "neural networks"        (exact adjacent-token match)
#   Boolean operators  →  python AND (learning OR neural) NOT robotics
#
# Parsing strategy:
#   1. If the query contains quotes  → phrase mode
#   2. If the query contains AND/OR/NOT → boolean mode
#   3. Otherwise                      → simple multi-term AND mode

# Shape of a query object
class Query:
    def __init__(self, mode, terms=None, phrase_tokens=None, boolean_ast=None, raw=""):
        self.mode = mode  # "simple" | "phrase" | "boolean"
        self.terms = terms or []  # list of stemmed tokens (simple mode)
        self.phrase_tokens = (
            phrase_tokens  # list of stemmed tokens that must be adjacent (phrase mode)
        )
        self.boolean_ast = boolean_ast  # nested dict AST (boolean mode)
        self.raw = raw  # original query string


def parse_query(raw: str) -> Query:
    raw = raw.strip()
    if not raw:
        return Query(mode="simple", raw=raw)

    # Phrase query: "..."
    phrase_match = re.match(r'^"(.+)"$', raw)
    if phrase_match:
        inner = phrase_match.group(1)
        tokens = tokenize(inner)
        logger.info("Parsed as PHRASE query: %s", tokens)
        return Query(mode="phrase", phrase_tokens=tokens, raw=raw)

    # Boolean query: contains AND / OR / NOT
    if re.search(r"\b(AND|OR|NOT)\b", raw):
        ast = _parse_boolean(raw)
        logger.info("Parsed as BOOLEAN query: %s", ast)
        return Query(mode="boolean", boolean_ast=ast, raw=raw)

    # Simple multi-term (implicit AND)
    tokens = tokenize(raw)
    logger.info("Parsed as SIMPLE query: %s", tokens)
    return Query(mode="simple", terms=tokens, raw=raw)


def _parse_boolean(raw: str) -> dict:
    """
    Minimal recursive-descent parser for boolean expressions.
    Grammar:
        expr   → factor (( AND | OR ) factor)*
        factor → NOT factor | ATOM
        ATOM   → word  |  "(" expr ")"
    Returns a nested dict AST, e.g.:
        {"op": "AND", "left": {"term": "python"}, "right": {"op": "NOT", "operand": {"term": "robot"}}}
    """
    tokens = raw.split()
    pos = [0]  # mutable pointer so nested calls can advance it

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def consume():
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def parse_expr():
        left = parse_factor()
        while peek() in ("AND", "OR"):
            op = consume()
            right = parse_factor()
            left = {"op": op, "left": left, "right": right}
        return left

    def parse_factor():
        if peek() == "NOT":
            consume()
            operand = parse_factor()
            return {"op": "NOT", "operand": operand}
        return parse_atom()

    def parse_atom():
        if peek() == "(":
            consume()  # eat '('
            node = parse_expr()
            if peek() == ")":
                consume()  # eat ')'
            return node
        # bare word → tokenize+stem it
        word = consume()
        stemmed = tokenize(word)
        return {"term": stemmed[0] if stemmed else word.lower()}

    return parse_expr()


# ─────────────────────────────────────────────
# BM25 SCORER
# ─────────────────────────────────────────────
# BM25 formula per term t in query, per document d:
#
#   IDF(t)  =  ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
#   TF_norm =  (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
#   score   =  Σ  IDF(t) * TF_norm
#
# N        = total docs in corpus
# df(t)    = number of docs containing term t
# tf(t,d)  = raw count of term t in doc d
# |d|      = length of doc d (in tokens)
# avgdl    = average doc length across corpus


def _bm25_idf(doc_freq: int, num_docs: int) -> float:
    """IDF component — penalises terms that appear in almost every document."""
    return math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)


def _bm25_tf(term_freq: int, doc_length: int, avg_doc_length: float) -> float:
    """Saturating TF component — diminishing returns for repeated terms."""
    return (term_freq * (BM25_K1 + 1)) / (
        term_freq + BM25_K1 * (1 - BM25_B + BM25_B * doc_length / avg_doc_length)
    )


def score_simple(terms: list[str]) -> list[tuple[str, float]]:
    """
    Scores ALL docs in the index against a list of query terms using BM25.
    Only docs that contain at least one query term are scored.
    Returns list of (doc_id, score) sorted descending.
    """
    _ensure_loaded()
    index = _index_cache["index"]
    num_docs = _index_cache["metadata"]["num_docs"]
    avg_dl = _index_cache["metadata"]["avg_doc_length"]
    doc_lengths = _index_cache["doc_lengths"]

    scores = defaultdict(float)  # doc_id → cumulative BM25 score

    for term in terms:
        if term not in index:
            continue  # unknown term — skip gracefully
        entry = index[term]
        idf = _bm25_idf(entry["doc_freq"], num_docs)

        for doc_id, posting in entry["postings"].items():
            tf_norm = _bm25_tf(posting["term_freq"], doc_lengths[doc_id], avg_dl)
            scores[doc_id] += idf * tf_norm

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────
# PHRASE SEARCH
# ─────────────────────────────────────────────
# Uses the position lists stored in the index.
# A phrase matches doc d if every consecutive pair of tokens
# appears at consecutive positions somewhere in d.


def score_phrase(phrase_tokens: list[str]) -> list[tuple[str, float]]:
    """
    Finds docs where phrase_tokens appear consecutively (in order).
    Score = BM25 of the first token (so results are still meaningfully ranked).
    """
    _ensure_loaded()
    if not phrase_tokens:
        return []

    index = _index_cache["index"]
    num_docs = _index_cache["metadata"]["num_docs"]
    avg_dl = _index_cache["metadata"]["avg_doc_length"]
    doc_lengths = _index_cache["doc_lengths"]

    # Start with candidate docs that contain the first token
    first = phrase_tokens[0]
    if first not in index:
        return []

    # For each candidate doc, check whether all tokens are consecutive
    matches = []
    for doc_id, posting in index[first]["postings"].items():
        start_positions = posting["positions"]

        for start in start_positions:
            found = True
            for offset, token in enumerate(phrase_tokens[1:], 1):
                if token not in index:
                    found = False
                    break
                postings_for_token = index[token]["postings"]
                if doc_id not in postings_for_token:
                    found = False
                    break
                if (start + offset) not in postings_for_token[doc_id]["positions"]:
                    found = False
                    break
            if found:
                # Phrase confirmed — score with BM25 of first token as tiebreaker
                idf = _bm25_idf(index[first]["doc_freq"], num_docs)
                tf_norm = _bm25_tf(posting["term_freq"], doc_lengths[doc_id], avg_dl)
                matches.append((doc_id, idf * tf_norm))
                break  # one match per doc is enough

    return sorted(matches, key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────
# BOOLEAN SEARCH
# ─────────────────────────────────────────────
# Walks the AST and returns a SET of matching doc_ids at each node.
# Final set is scored with BM25 on the leaf terms for ranking.


def _eval_boolean(node: dict) -> set[str]:
    """Recursively evaluates a boolean AST node → set of matching doc_ids."""
    _ensure_loaded()
    index = _index_cache["index"]

    if "term" in node:
        # Leaf: return all docs containing this term
        term = node["term"]
        if term in index:
            return set(index[term]["postings"].keys())
        return set()

    op = node["op"]

    if op == "NOT":
        all_docs = set(_index_cache["doc_lengths"].keys())
        return all_docs - _eval_boolean(node["operand"])

    left = _eval_boolean(node["left"])
    right = _eval_boolean(node["right"])

    if op == "AND":
        return left & right
    if op == "OR":
        return left | right

    return set()


def _collect_leaf_terms(node: dict) -> list[str]:
    """Walks the AST and collects every leaf term (for BM25 scoring)."""
    if "term" in node:
        return [node["term"]]
    terms = []
    if "operand" in node:
        terms += _collect_leaf_terms(node["operand"])
    if "left" in node:
        terms += _collect_leaf_terms(node["left"])
    if "right" in node:
        terms += _collect_leaf_terms(node["right"])
    return terms


def score_boolean(ast: dict) -> list[tuple[str, float]]:
    """
    Evaluates the boolean AST to get the matching doc set,
    then ranks those docs by BM25 on the positive leaf terms.
    """
    matching_ids = _eval_boolean(ast)
    if not matching_ids:
        return []

    # Score the full corpus with BM25, then filter to the boolean match set
    leaf_terms = _collect_leaf_terms(ast)
    all_scored = score_simple(leaf_terms)
    return [(did, score) for did, score in all_scored if did in matching_ids]


# ─────────────────────────────────────────────
# SNIPPET GENERATOR
# ─────────────────────────────────────────────
# Finds the best window of text around the first query-term hit
# and returns it as a short preview with the match bolded.


def generate_snippet(
    text: str, query_terms: list[str], max_len: int = SNIPPET_LENGTH
) -> str:
    """
    Finds the sentence / region containing the first raw (un-stemmed) term hit,
    returns a trimmed window of ≤ max_len chars with **bold** highlights.
    Falls back to the opening of the text if no term is found.
    """
    lower = text.lower()

    # Try to locate any query term (raw, before stemming) in the original text
    best_pos = len(text)  # fallback: start of doc
    for term in query_terms:
        idx = lower.find(term.lower())
        if idx != -1 and idx < best_pos:
            best_pos = idx

    # Centre a window around the hit
    half = max_len // 2
    start = max(0, best_pos - half)
    end = min(len(text), best_pos + half)
    window = text[start:end]

    # Trim to nearest word boundary on both sides
    if start > 0:
        space = window.find(" ")
        if space != -1:
            window = window[space + 1 :]
    if end < len(text):
        space = window.rfind(" ")
        if space != -1:
            window = window[:space]

    # Bold-highlight every occurrence of any query term in the snippet
    # Sort longest-first so "neural networks" highlights before "neural"
    for term in sorted(query_terms, key=len, reverse=True):
        if len(term) < 2:
            continue  # skip single-char noise like "a"
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        window = pattern.sub(lambda m: f"**{m.group()}**", window)

    # Add ellipsis if we trimmed
    if start > 0:
        window = "..." + window
    if end < len(text):
        window = window + "..."

    return window


# ─────────────────────────────────────────────
# MAIN SEARCH ENTRY POINT
# ─────────────────────────────────────────────


def search(raw_query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Public API.  Takes a raw query string, returns a list of result dicts:
        [
            {
                "rank":    1,
                "doc_id":  "2",
                "title":   "Python Programming Language",
                "url":     "https://...",
                "score":   3.42,
                "snippet": "...Python is a high-level **programming** language..."
            },
            ...
        ]
    """
    _ensure_loaded()
    query = parse_query(raw_query)

    # ── Route to the right scorer ─────────────────────────
    if query.mode == "phrase":
        scored = score_phrase(query.phrase_tokens)
        raw_terms = query.phrase_tokens  # for snippet highlighting
    elif query.mode == "boolean":
        scored = score_boolean(query.boolean_ast)
        raw_terms = _collect_leaf_terms(query.boolean_ast)
    else:
        scored = score_simple(query.terms)
        raw_terms = query.terms

    # ── Build result dicts ────────────────────────────────
    # Also keep the original (un-stemmed) words for snippet highlighting
    original_words = re.findall(r"[a-z0-9]+", raw_query.lower())

    results = []
    for rank, (doc_id, score) in enumerate(scored[:top_k], 1):
        page = _pages_by_id.get(doc_id, {})
        snippet = generate_snippet(page.get("text", ""), original_words)
        results.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "title": page.get("title", "Unknown"),
                "url": page.get("url", ""),
                "score": round(score, 4),
                "snippet": snippet,
            }
        )

    return results


def print_results(results: list[dict]):
    """Pretty-prints a result list to the terminal."""
    if not results:
        print("\n  No results found.")
        return
    print()
    for r in results:
        print(f"  #{r['rank']}  [{r['score']}]  {r['title']}")
        print(f"      {r['url']}")
        print(f"      {r['snippet']}")
        print()


# ─────────────────────────────────────────────
# INTERACTIVE REPL  (run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    _ensure_loaded()
    print("╔══════════════════════════════════════════════╗")
    print("║          🔍  Search Engine REPL              ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  plain words      →  neural networks        ║")
    print('║  exact phrase     →  "neural networks"      ║')
    print("║  boolean          →  python AND NOT robot   ║")
    print("║  quit             →  q                      ║")
    print("╚══════════════════════════════════════════════╝\n")

    while True:
        try:
            raw = input("  🔎 Search: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break
        if raw.lower() in ("q", "quit", "exit"):
            print("  Goodbye.")
            break
        if not raw:
            continue

        results = search(raw)
        print_results(results)
