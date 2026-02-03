import os
from flask import Flask, request, jsonify
from query_engine import search, _ensure_loaded, _index_cache

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Pre-warm the index on startup so the first query isn't slow
_ensure_loaded()


# ─────────────────────────────────────────────
# SERVE THE FRONTEND
# ─────────────────────────────────────────────


@app.route("/")
def index():
    return app.send_static_file("index.html")


# ─────────────────────────────────────────────
# SEARCH API
# ─────────────────────────────────────────────


@app.route("/search")
def search_endpoint():
    query = request.args.get("q", "").strip()
    top_k = request.args.get("top_k", 5, type=int)

    if not query:
        return jsonify({"query": "", "count": 0, "results": []})

    results = search(query, top_k=top_k)
    return jsonify(
        {
            "query": query,
            "count": len(results),
            "results": results,
        }
    )


# ─────────────────────────────────────────────
# INDEX STATS
# ─────────────────────────────────────────────


@app.route("/stats")
def stats():
    _ensure_loaded()
    return jsonify(
        {
            "num_docs": _index_cache["metadata"]["num_docs"],
            "num_terms": len(_index_cache["index"]),
        }
    )


# ─────────────────────────────────────────────
# MAIN ENTRY POINT (for local development only)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Starting Search Engine Server...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
