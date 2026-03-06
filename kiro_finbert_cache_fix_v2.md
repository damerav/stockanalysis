# Kiro Prompt: FinBERT Caching Architecture Fix (v2)

## Problem Statement

The daily pipeline is logging:

```
FinBERT: scoring 1294 uncached articles...
```

This means FinBERT — a heavy transformer model — is running inference on 1,294 articles in a single blocking batch on every pipeline run. This is slow, wasteful, and the root cause is a **broken caching architecture with three distinct gaps**.

---

## Root Cause Analysis

### Gap 1: `analyzer.py` Has No Cache Awareness (Most Critical)

`src/llm/analyzer.py` is the module called by the main pipeline (`daily_run.py`) for sentiment analysis. Its `analyze_sentiment()` method runs FinBERT on **every article passed to it, every time, unconditionally**. There is no check against the `finbert_cache` table before running inference. The articles passed to it come from the `news` table (the main PostgreSQL table), not from `raw_articles`, so even if `finbert_cache` were checked, the article IDs would not match.

### Gap 2: `geopolitical_features.py` Has the Cache But Reads from the Wrong Table

`src/data/geopolitical_features.py` has the correct `finbert_cache` table and the correct `LEFT JOIN` pattern to find uncached articles. However, it reads from `raw_articles` (the news.db / RSS pipeline), while `analyzer.py` reads from the `news` table (the Finnhub / Polygon pipeline). These are two separate article stores with no shared cache key.

### Gap 3: The `finbert_cache` Table Has No `scored_at` Timestamp

The `finbert_cache` table schema is:
```sql
CREATE TABLE IF NOT EXISTS finbert_cache (
    article_id INTEGER PRIMARY KEY,
    fb_positive REAL,
    fb_negative REAL,
    fb_neutral REAL
)
```
There is no `scored_at` column. This means there is no way to audit when an article was scored, detect stale scores, or monitor cache health from the dashboard.

### Gap 4: `NewsFeatureProcessor` Uses VADER Only, Not FinBERT

`src/data/news_features.py` reads all articles from `raw_articles` and computes sentiment using VADER only. It never checks the `finbert_cache`. This means the expanded sentiment that feeds into the final blended score is lower quality than it could be.

---

## Architecture After This Fix

```
news_fetcher.py (RSS)          news_fetcher.py (Finnhub)
       │                                │
       ▼                                ▼
 raw_articles table              news table (main DB)
       │                                │
       └──────────────┬─────────────────┘
                      ▼
              finbert_cache table
              (keyed by url_hash — works for BOTH tables)
                      │
                      ▼
   analyzer.py reads cache first
   news_features.py reads cache first
   geopolitical_features.py reads cache first
                      │
                      ▼
   daily_run.py step 4 is fast
   (cache hit rate > 95% after day 1)
```

---

## Part 1: Schema Changes

### File: `src/data/init_db.py`

**Step 1a — Add `scored_at` column to `finbert_cache` table.**

Find the `finbert_cache` table creation in `_migrate_schema()`. The table is currently created inline in `geopolitical_features.py` rather than in `init_db.py`. Add the following to the `SCHEMA` string in `init_db.py` so it is managed centrally:

```sql
CREATE TABLE IF NOT EXISTS finbert_cache (
    url_hash TEXT PRIMARY KEY,      -- SHA-256 of (headline + summary[:300]), hex-encoded
    article_id INTEGER,             -- optional back-reference to raw_articles.id
    fb_positive REAL NOT NULL,
    fb_negative REAL NOT NULL,
    fb_neutral REAL NOT NULL,
    fb_label TEXT,                  -- 'positive', 'negative', or 'neutral'
    fb_score REAL,                  -- fb_positive - fb_negative (precomputed)
    scored_at TEXT NOT NULL         -- ISO timestamp of when FinBERT was run
);
```

**Key design decision:** The cache key is `url_hash` (a SHA-256 hash of `headline + summary[:300]`), NOT `article_id`. This means the cache works for articles from **both** the `raw_articles` table and the `news` table, because both tables store the same article text even if they have different integer IDs.

**Step 1b — Add migration block in `_migrate_schema()`.**

Add the following migration alongside the existing `ALTER TABLE` blocks:

```python
# finbert_cache: drop old integer-keyed table and recreate with url_hash key
try:
    # Check if old schema (article_id PRIMARY KEY) exists
    old_schema = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='finbert_cache'"
    ).fetchone()
    if old_schema and "url_hash" not in (old_schema[0] or ""):
        # Old schema detected — rename and recreate
        conn.execute("ALTER TABLE finbert_cache RENAME TO finbert_cache_old")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS finbert_cache (
                url_hash TEXT PRIMARY KEY,
                article_id INTEGER,
                fb_positive REAL NOT NULL,
                fb_negative REAL NOT NULL,
                fb_neutral REAL NOT NULL,
                fb_label TEXT,
                fb_score REAL,
                scored_at TEXT NOT NULL
            )
        """)
        # Migrate old rows — compute url_hash from article text where possible
        # (old rows without text cannot be migrated; they will be re-scored on next run)
        logger.info("finbert_cache: migrated to url_hash-keyed schema")
except Exception as e:
    logger.warning(f"finbert_cache migration: {e}")
conn.commit()
```

For **PostgreSQL / TimescaleDB**, the migration block should use:
```python
try:
    conn.execute("ALTER TABLE finbert_cache ADD COLUMN url_hash TEXT")
    conn.execute("ALTER TABLE finbert_cache ADD COLUMN fb_label TEXT")
    conn.execute("ALTER TABLE finbert_cache ADD COLUMN fb_score REAL")
    conn.execute("ALTER TABLE finbert_cache ADD COLUMN scored_at TEXT")
except Exception:
    pass  # Columns already exist
```

---

## Part 2: New Shared FinBERT Cache Utility

### New File: `src/data/finbert_cache_utils.py`

Create this new file. It is the single source of truth for all FinBERT cache operations. Both `analyzer.py` and `geopolitical_features.py` will import from it.

```python
"""
finbert_cache_utils.py

Shared utility for FinBERT sentiment caching.
Provides cache-aware scoring that:
1. Computes a url_hash key from article text (works for both raw_articles and news tables)
2. Checks finbert_cache before running FinBERT inference
3. Writes new scores to finbert_cache immediately after inference
4. Returns cached scores for already-processed articles

This eliminates the 1,294-article uncached batch problem by ensuring
FinBERT only runs on articles it has never seen before.
"""

import hashlib
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def _make_url_hash(headline: str, summary: str) -> str:
    """Compute a stable cache key from article text.

    Uses SHA-256 of (headline + summary[:300]) to ensure the same article
    from different sources (raw_articles vs news table) maps to the same key.
    """
    text = (headline or "").strip() + " " + (summary or "")[:300].strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_cached_scores(router, url_hashes: list[str]) -> dict:
    """Fetch cached FinBERT scores for a list of url_hashes.

    Returns dict mapping url_hash -> {fb_positive, fb_negative, fb_neutral, fb_label, fb_score}.
    Missing hashes are not included in the result.
    """
    if not url_hashes:
        return {}
    placeholders = ",".join(["?" for _ in url_hashes])
    try:
        df = router.query(
            f"SELECT url_hash, fb_positive, fb_negative, fb_neutral, fb_label, fb_score "
            f"FROM finbert_cache WHERE url_hash IN ({placeholders})",
            tuple(url_hashes),
        )
        if df.empty:
            return {}
        return {
            row["url_hash"]: {
                "fb_positive": float(row["fb_positive"]),
                "fb_negative": float(row["fb_negative"]),
                "fb_neutral": float(row["fb_neutral"]),
                "fb_label": row.get("fb_label", "neutral"),
                "fb_score": float(row.get("fb_score") or 0.0),
            }
            for _, row in df.iterrows()
        }
    except Exception as e:
        logger.warning(f"finbert_cache read failed: {e}")
        return {}


def write_cached_scores(router, scores: list[dict]) -> int:
    """Write a batch of FinBERT scores to the cache.

    Each dict in scores must have keys:
        url_hash, article_id (optional), fb_positive, fb_negative, fb_neutral

    Returns count of rows written.
    """
    if not scores:
        return 0
    now = datetime.now().isoformat()
    written = 0
    for s in scores:
        try:
            fb_score = float(s.get("fb_positive", 0)) - float(s.get("fb_negative", 0))
            # Determine label from highest probability
            probs = {
                "positive": float(s.get("fb_positive", 0)),
                "negative": float(s.get("fb_negative", 0)),
                "neutral": float(s.get("fb_neutral", 1)),
            }
            fb_label = max(probs, key=probs.get)
            router.execute(
                "INSERT OR REPLACE INTO finbert_cache "
                "(url_hash, article_id, fb_positive, fb_negative, fb_neutral, "
                "fb_label, fb_score, scored_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    s["url_hash"],
                    s.get("article_id"),
                    float(s.get("fb_positive", 0)),
                    float(s.get("fb_negative", 0)),
                    float(s.get("fb_neutral", 1)),
                    fb_label,
                    round(fb_score, 4),
                    now,
                ),
            )
            written += 1
        except Exception as e:
            logger.warning(f"finbert_cache write failed for {s.get('url_hash', '?')}: {e}")
    return written


def score_articles_with_cache(
    router,
    articles: list[dict],
    finbert_pipeline,
    batch_size: int = 32,
) -> list[dict]:
    """Score a list of articles using FinBERT, with full cache support.

    Each article dict must have 'headline' and 'summary' keys.
    Optionally may have 'id' (maps to article_id in cache).

    Returns list of dicts (same order as input) with keys:
        url_hash, fb_positive, fb_negative, fb_neutral, fb_label, fb_score,
        from_cache (bool)

    Algorithm:
    1. Compute url_hash for every article.
    2. Batch-fetch all existing cache entries for those hashes.
    3. Only run FinBERT on articles NOT in cache.
    4. Write new scores to cache.
    5. Return merged results in original order.
    """
    if not articles:
        return []

    # Step 1: Compute url_hashes for all articles
    hashes = [_make_url_hash(a.get("headline", ""), a.get("summary", "")) for a in articles]

    # Step 2: Fetch cached scores
    cached = get_cached_scores(router, hashes)
    cache_hits = sum(1 for h in hashes if h in cached)
    cache_misses = len(hashes) - cache_hits

    if cache_misses > 0:
        logger.info(f"FinBERT cache: {cache_hits} hits, {cache_misses} misses — "
                    f"running inference on {cache_misses} new articles")
    else:
        logger.info(f"FinBERT cache: {cache_hits} hits, 0 misses — skipping inference entirely")

    # Step 3: Run FinBERT only on uncached articles
    uncached_indices = [i for i, h in enumerate(hashes) if h not in cached]
    if uncached_indices and finbert_pipeline is not None:
        uncached_texts = []
        for i in uncached_indices:
            a = articles[i]
            text = (a.get("headline", "") + " " + a.get("summary", "")[:300]).strip()
            uncached_texts.append(text or "neutral")

        # Run FinBERT in batches
        new_scores = []
        for batch_start in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[batch_start:batch_start + batch_size]
            batch = [t[:512] for t in batch]
            try:
                preds = finbert_pipeline(batch)
                for p in preds:
                    label = p["label"].lower()
                    score = float(p["score"])
                    new_scores.append({
                        "positive": score if label == "positive" else 0.0,
                        "negative": score if label == "negative" else 0.0,
                        "neutral": score if label == "neutral" else 0.0,
                    })
            except Exception as e:
                logger.warning(f"FinBERT batch inference failed: {e}")
                new_scores.extend([
                    {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
                    for _ in batch
                ])

        # Step 4: Write new scores to cache
        cache_rows = []
        for rank, orig_idx in enumerate(uncached_indices):
            if rank >= len(new_scores):
                break
            s = new_scores[rank]
            a = articles[orig_idx]
            cache_rows.append({
                "url_hash": hashes[orig_idx],
                "article_id": a.get("id"),
                "fb_positive": s["positive"],
                "fb_negative": s["negative"],
                "fb_neutral": s["neutral"],
            })
            # Also update the in-memory cached dict for Step 5
            cached[hashes[orig_idx]] = {
                "fb_positive": s["positive"],
                "fb_negative": s["negative"],
                "fb_neutral": s["neutral"],
                "fb_label": max(s, key=s.get),
                "fb_score": round(s["positive"] - s["negative"], 4),
            }

        written = write_cached_scores(router, cache_rows)
        logger.info(f"FinBERT cache: wrote {written} new scores")

    # Step 5: Assemble results in original order
    results = []
    for i, h in enumerate(hashes):
        c = cached.get(h, {"fb_positive": 0.0, "fb_negative": 0.0, "fb_neutral": 1.0,
                           "fb_label": "neutral", "fb_score": 0.0})
        results.append({
            "url_hash": h,
            "fb_positive": c["fb_positive"],
            "fb_negative": c["fb_negative"],
            "fb_neutral": c["fb_neutral"],
            "fb_label": c.get("fb_label", "neutral"),
            "fb_score": c.get("fb_score", 0.0),
            "from_cache": h in cached and i not in (uncached_indices if uncached_indices else []),
        })

    return results
```

---

## Part 3: Update `analyzer.py` to Use the Cache

### File: `src/llm/analyzer.py`

**Step 3a — Add router initialization to `__init__`.**

Find the `__init__` method:

```python
    def __init__(self, config: dict = None):
        ...
        # P1: FinBERT fast-path
        self.finbert = None
        self.finbert_available = False
        self._init_finbert()
```

Add router initialization immediately after `self._init_finbert()`:

```python
        # Cache-aware FinBERT: initialize router for cache reads/writes
        self._router = None
        try:
            from src.data.db_router import get_router
            self._router = get_router(config)
        except Exception:
            pass  # Router unavailable — will fall back to uncached scoring
```

**Step 3b — Replace `analyze_sentiment()` with a cache-aware version.**

Find the full `analyze_sentiment()` method (lines 271–316) and replace it entirely with:

```python
    def analyze_sentiment(self, articles: list[dict]) -> list[dict]:
        """Analyze sentiment using two-tier pipeline with FinBERT cache (P1 enhanced):
        1. Fast path: FinBERT on articles NOT in cache (cache-hit articles are free)
        2. Deep path: DeepSeek on top 5 highest-impact articles

        Falls back to LLM-only or FinBERT-only if either is unavailable.
        Cache key: SHA-256 of (headline + summary[:300]) — works across both
        raw_articles and news tables.
        """
        if not self.finbert_available and not self.llm_available:
            logger.info("No sentiment models available, returning neutral")
            return [{"score": 0.0, "confidence": 0, "topics": []} for _ in articles]

        # --- Fast path: FinBERT with cache ---
        fast_results = []
        if self.finbert_available:
            if self._router is not None:
                # Cache-aware path: only run inference on uncached articles
                from src.data.finbert_cache_utils import score_articles_with_cache
                fb_results = score_articles_with_cache(
                    router=self._router,
                    articles=articles,
                    finbert_pipeline=self.finbert,
                    batch_size=32,
                )
                for r in fb_results:
                    # Convert fb_score (-1 to +1) to the format expected downstream
                    # fb_score = fb_positive - fb_negative
                    raw_score = r["fb_score"]
                    # Scale to match original _finbert_score output convention
                    # Original: score_map * result["score"], range approx -1 to +1
                    conf = int(max(r["fb_positive"], r["fb_negative"], r["fb_neutral"]) * 100)
                    fast_results.append({
                        "score": round(raw_score, 4),
                        "confidence": conf,
                        "topics": [],
                    })
            else:
                # Fallback: no router available, run uncached (original behavior)
                logger.info(f"FinBERT fast-path (uncached): scoring {len(articles)} articles...")
                for a in articles:
                    text = f"{a.get('headline', '')} {a.get('summary', '')[:300]}"
                    fb = self._finbert_score(text)
                    fast_results.append({
                        "score": fb["score"],
                        "confidence": fb["confidence"],
                        "topics": [],
                    })
                logger.info("FinBERT fast-path complete")
        else:
            fast_results = [{"score": 0.0, "confidence": 0, "topics": []} for _ in articles]

        if not self.llm_available:
            return fast_results

        # --- Deep path: DeepSeek on top 5 by absolute FinBERT score ---
        abs_scores = [abs(r["score"]) for r in fast_results]
        top_indices = sorted(range(len(abs_scores)),
                             key=lambda i: abs_scores[i], reverse=True)[:5]
        top_articles = [articles[i] for i in top_indices]

        logger.info(f"DeepSeek deep-path: analysing top {len(top_articles)} articles...")
        deep_results = self._analyze_batch(top_articles)

        # Merge: replace fast scores with deep scores for top articles
        results = list(fast_results)
        for rank, orig_idx in enumerate(top_indices):
            if rank < len(deep_results):
                results[orig_idx] = deep_results[rank]

        return results
```

---

## Part 4: Update `geopolitical_features.py` to Use the Shared Cache

### File: `src/data/geopolitical_features.py`

**Step 4a — Replace the inline `finbert_cache` creation and the uncached query.**

Find the `compute_daily_finbert_features()` function (line 216 onwards). Replace the entire function body with:

```python
def compute_daily_finbert_features(config: dict = None, days_back: int = 7) -> pd.DataFrame:
    """Compute daily FinBERT sentiment features from recent news articles.

    Uses the shared finbert_cache_utils for cache-aware scoring.
    Cache key is url_hash (SHA-256 of headline+summary[:300]) — works for
    both raw_articles and news tables.

    Returns DataFrame with: date, finbert_positive, finbert_negative,
    finbert_neutral, finbert_score (positive - negative)
    """
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    from src.data.db_router import get_router
    from src.data.finbert_cache_utils import score_articles_with_cache
    router = get_router(config)

    # Fetch recent articles from raw_articles
    try:
        articles_df = router.query(
            "SELECT id, published_at, headline, summary "
            "FROM raw_articles "
            f"WHERE published_at >= '{cutoff}' "
            "ORDER BY published_at"
        )
    except Exception as e:
        logger.warning(f"raw_articles query failed: {e}")
        return pd.DataFrame()

    if articles_df.empty:
        return pd.DataFrame()

    # Convert to list of dicts for score_articles_with_cache
    articles = [
        {
            "id": row["id"],
            "headline": row["headline"] or "",
            "summary": row["summary"] or "",
            "published_at": row["published_at"],
        }
        for _, row in articles_df.iterrows()
    ]

    # Load FinBERT pipeline (reuse if already loaded, else load fresh)
    finbert_pipe = None
    try:
        from transformers import pipeline as hf_pipeline
        finbert_pipe = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # CPU; change to 0 for GPU
            truncation=True,
            max_length=512,
        )
    except Exception as e:
        logger.warning(f"FinBERT load failed in geopolitical_features: {e}")

    # Score with cache — only uncached articles will hit the model
    scored = score_articles_with_cache(
        router=router,
        articles=articles,
        finbert_pipeline=finbert_pipe,
        batch_size=32,
    )

    # Reconstruct DataFrame with dates
    records = []
    for article, score in zip(articles, scored):
        date_str = _parse_date_str(article["published_at"] or "")
        if not date_str:
            continue
        records.append({
            "date": date_str,
            "fb_positive": score["fb_positive"],
            "fb_negative": score["fb_negative"],
            "fb_neutral": score["fb_neutral"],
        })

    if not records:
        return pd.DataFrame()

    result_df = pd.DataFrame(records)
    daily = result_df.groupby("date").agg(
        finbert_positive=("fb_positive", "mean"),
        finbert_negative=("fb_negative", "mean"),
        finbert_neutral=("fb_neutral", "mean"),
    ).reset_index()
    daily["finbert_score"] = daily["finbert_positive"] - daily["finbert_negative"]
    return daily
```

---

## Part 5: Cache Pre-warming in `news_fetcher.py`

### File: `src/data/news_fetcher.py`

The goal is to score new articles with FinBERT **at fetch time**, so that by the time the daily pipeline runs, the cache is already warm and the pipeline's FinBERT step is a near-instant cache read.

**Step 5a — Add a `_score_new_articles_finbert()` method to `NewsFetcher`.**

Add the following method to the `NewsFetcher` class after the `__init__` method:

```python
    def _score_new_articles_finbert(self, limit: int = 200) -> int:
        """Score recently fetched, uncached articles with FinBERT.

        Called at the end of fetch_all() to pre-warm the cache.
        Only processes articles not yet in finbert_cache.
        Limits to `limit` articles per call to avoid blocking the fetch step.

        Returns count of articles scored.
        """
        from src.data.finbert_cache_utils import _make_url_hash, write_cached_scores

        # Find recently inserted articles not yet in finbert_cache
        try:
            uncached_df = self.router.query(
                "SELECT a.id, a.headline, a.summary "
                "FROM raw_articles a "
                "LEFT JOIN finbert_cache c "
                "  ON MD5(CONCAT(TRIM(a.headline), ' ', LEFT(TRIM(a.summary), 300))) = c.url_hash "
                f"WHERE c.url_hash IS NULL "
                "ORDER BY a.fetched_at DESC "
                f"LIMIT {limit}"
            )
        except Exception:
            # Fallback for SQLite (no MD5/CONCAT) — use Python-side hash check
            try:
                recent_df = self.router.query(
                    "SELECT id, headline, summary FROM raw_articles "
                    "ORDER BY fetched_at DESC LIMIT 500"
                )
                if recent_df.empty:
                    return 0
                hashes = [
                    _make_url_hash(row["headline"] or "", row["summary"] or "")
                    for _, row in recent_df.iterrows()
                ]
                from src.data.finbert_cache_utils import get_cached_scores
                cached_set = set(get_cached_scores(self.router, hashes).keys())
                uncached_rows = [
                    (row, h) for (_, row), h in zip(recent_df.iterrows(), hashes)
                    if h not in cached_set
                ][:limit]
                if not uncached_rows:
                    return 0
                uncached_df = recent_df.iloc[[
                    list(recent_df.index).index(row.name)
                    for row, _ in uncached_rows
                ]]
            except Exception as e:
                logger.warning(f"FinBERT pre-warm query failed: {e}")
                return 0

        if uncached_df.empty:
            return 0

        logger.info(f"FinBERT pre-warm: scoring {len(uncached_df)} new articles...")

        # Load FinBERT
        try:
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1,  # CPU
                truncation=True,
                max_length=512,
            )
        except Exception as e:
            logger.warning(f"FinBERT pre-warm: model load failed ({e}), skipping")
            return 0

        from src.data.finbert_cache_utils import score_articles_with_cache
        articles = [
            {"id": row["id"], "headline": row["headline"] or "", "summary": row["summary"] or ""}
            for _, row in uncached_df.iterrows()
        ]
        scored = score_articles_with_cache(
            router=self.router,
            articles=articles,
            finbert_pipeline=pipe,
            batch_size=32,
        )
        logger.info(f"FinBERT pre-warm: {sum(1 for s in scored if not s['from_cache'])} articles scored and cached")
        return len(scored)
```

**Step 5b — Call `_score_new_articles_finbert()` at the end of `fetch_all()`.**

Find the `fetch_all()` method (it calls `fetch_rss()`, `fetch_finnhub()`, etc.). Add the following at the very end of `fetch_all()`, before the return statement:

```python
        # Pre-warm FinBERT cache for newly fetched articles
        # This runs after all fetching is complete, scoring up to 200 new articles
        # so the daily pipeline's sentiment step is a fast cache read
        try:
            self._score_new_articles_finbert(limit=200)
        except Exception as e:
            logger.warning(f"FinBERT pre-warm failed (non-fatal): {e}")
```

---

## Part 6: Upgrade `NewsFeatureProcessor` to Use FinBERT Cache

### File: `src/data/news_features.py`

**Step 6a — Replace `process_articles()` with a cache-aware version.**

Find the `process_articles()` method (lines 91–126) and replace it entirely with:

```python
    def process_articles(self, articles: list[dict] = None) -> pd.DataFrame:
        """Process raw articles into feature DataFrame.

        Uses FinBERT scores from cache if available, falling back to VADER.

        Returns DataFrame with columns: date, ticker, headline, sentiment_compound,
        sentiment_positive, sentiment_negative, clean_text
        """
        if articles is None:
            df = self.router.query(
                "SELECT * FROM raw_articles ORDER BY published_at DESC"
            )
            if df.empty:
                articles = []
            else:
                articles = df.to_dict("records")

        if not articles:
            return pd.DataFrame()

        # Check FinBERT cache for all articles
        from src.data.finbert_cache_utils import _make_url_hash, get_cached_scores
        hashes = [_make_url_hash(a.get("headline", ""), a.get("summary", "")) for a in articles]
        cached_scores = get_cached_scores(self.router, hashes)

        records = []
        for i, a in enumerate(articles):
            text = (a.get("headline", "") + " " + a.get("summary", "")).strip()
            clean = self._clean_text(text)
            pub = a.get("published_at", "")
            date_str = pub[:10] if pub else datetime.now().strftime("%Y-%m-%d")

            # Use FinBERT score if cached, else VADER
            h = hashes[i]
            if h in cached_scores:
                fb = cached_scores[h]
                sent = {
                    "compound": fb["fb_score"],
                    "positive": fb["fb_positive"],
                    "negative": fb["fb_negative"],
                    "neutral": fb["fb_neutral"],
                }
            else:
                sent = self._get_sentiment(text)  # VADER fallback

            records.append({
                "date": date_str,
                "ticker": a.get("ticker", "MARKET"),
                "headline": a.get("headline", ""),
                "clean_text": clean,
                "sentiment_compound": sent["compound"],
                "sentiment_positive": sent["positive"],
                "sentiment_negative": sent["negative"],
                "source": a.get("source", "unknown"),
            })
        return pd.DataFrame(records)
```

---

## Part 7: Cache Health Monitoring in `system_management_app.py`

### File: `src/dashboard/system_management_app.py`

Add a **FinBERT Cache Health** section to the System Status tab. Find the section that renders system health metrics and add the following block:

```python
    # --- FinBERT Cache Health ---
    st.markdown("#### FinBERT Cache Health")
    try:
        router = get_router(load_config())
        cache_df = router.query(
            "SELECT "
            "  COUNT(*) as total_cached, "
            "  SUM(CASE WHEN scored_at >= datetime('now', '-1 day') THEN 1 ELSE 0 END) as scored_today, "
            "  SUM(CASE WHEN scored_at >= datetime('now', '-7 days') THEN 1 ELSE 0 END) as scored_week, "
            "  AVG(fb_score) as avg_score, "
            "  MIN(scored_at) as oldest, "
            "  MAX(scored_at) as newest "
            "FROM finbert_cache"
        )
        if not cache_df.empty:
            row = cache_df.iloc[0]
            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                st.metric("Total Cached Articles", f"{int(row['total_cached'] or 0):,}")
            with cc2:
                st.metric("Scored Today", f"{int(row['scored_today'] or 0):,}")
            with cc3:
                st.metric("Scored This Week", f"{int(row['scored_week'] or 0):,}")
            with cc4:
                avg = float(row['avg_score'] or 0)
                st.metric(
                    "Avg Sentiment Score",
                    f"{avg:+.3f}",
                    delta_color="normal" if avg > 0 else "inverse",
                )
            st.caption(
                f"Cache range: {row.get('oldest', 'N/A')} → {row.get('newest', 'N/A')}"
            )
        else:
            st.info("finbert_cache table is empty — will populate on next pipeline run")
    except Exception as e:
        st.warning(f"Could not read finbert_cache: {e}")
```

---

## Part 8: Validation Checklist

After implementation, verify the following:

| # | Check | Expected Result |
| :--- | :--- | :--- |
| 1 | `python -c "from src.data.finbert_cache_utils import _make_url_hash; print(_make_url_hash('Fed raises rates', 'The Federal Reserve raised interest rates by 25bps'))"` | Returns a 64-character hex string |
| 2 | `python -c "from src.data.finbert_cache_utils import _make_url_hash; h1 = _make_url_hash('Fed raises rates', 'The Federal Reserve raised interest rates by 25bps'); h2 = _make_url_hash('Fed raises rates', 'The Federal Reserve raised interest rates by 25bps'); print(h1 == h2)"` | Returns `True` — same text always produces same hash |
| 3 | Run the daily pipeline: `python -m src.pipeline.daily_run` | First run logs `FinBERT cache: N hits, M misses — running inference on M new articles`. Second run logs `FinBERT cache: N hits, 0 misses — skipping inference entirely` |
| 4 | Check the log after second run | No line reading `FinBERT: scoring 1294 uncached articles...` |
| 5 | Query the database: `SELECT COUNT(*) FROM finbert_cache` | Returns a non-zero count after first pipeline run |
| 6 | Query the database: `SELECT url_hash, fb_label, fb_score, scored_at FROM finbert_cache LIMIT 5` | Returns rows with valid url_hash (64 chars), fb_label in ('positive','negative','neutral'), fb_score in range [-1, 1], scored_at as ISO timestamp |
| 7 | Open System Management page | FinBERT Cache Health section shows total cached count, scored today count, and average sentiment score |
| 8 | Run `fetch_all()` manually after pipeline | Log shows `FinBERT pre-warm: N articles scored and cached` for newly fetched articles |
| 9 | Run pipeline again immediately after fetch | Log shows `FinBERT cache: N hits, 0 misses` — pre-warm worked |
| 10 | Verify `analyzer.py` uses cache | Add `print(r['from_cache'])` temporarily in `score_articles_with_cache` — after first run, all articles should show `from_cache=True` |
| 11 | Verify `news_features.py` uses cache | Add `print(h in cached_scores)` temporarily in `process_articles` — after first run, all articles should show `True` |

---

## Design Notes for Kiro

**Why `url_hash` instead of `article_id`:** The `news` table (used by `analyzer.py`) and the `raw_articles` table (used by `geopolitical_features.py`) are separate tables with independent integer ID sequences. The same article can appear in both tables with different IDs. Using a content-based hash (SHA-256 of headline + summary) ensures the cache works across both tables without any foreign key relationship.

**Why pre-warm at fetch time:** The daily pipeline runs at a fixed time (e.g., 6:00 PM). The news fetcher runs at multiple points during the day. By scoring new articles at fetch time, the cache is fully warm before the pipeline runs, making the pipeline's FinBERT step effectively free (all cache hits).

**Why `limit=200` in pre-warm:** FinBERT on CPU takes approximately 0.1–0.3 seconds per article. 200 articles = 20–60 seconds, which is acceptable during a background fetch. If the backlog ever exceeds 200 articles (e.g., after a restart), the pipeline run will handle the overflow and the pre-warm will catch up on the next fetch cycle.

**TimescaleDB compatibility:** The `finbert_cache` table is NOT a hypertable. It is a standard relational table keyed by `url_hash`. It does not need time-based partitioning because it is looked up by hash, not by time range. Do not convert it to a hypertable.
