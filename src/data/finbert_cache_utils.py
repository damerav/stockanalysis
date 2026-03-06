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
    result = {}
    # Process in chunks of 500 to avoid SQL parameter limits
    chunk_size = 500
    for start in range(0, len(url_hashes), chunk_size):
        chunk = url_hashes[start:start + chunk_size]
        placeholders = ",".join(["?" for _ in chunk])
        try:
            df = router.query(
                f"SELECT url_hash, fb_positive, fb_negative, fb_neutral, fb_label, fb_score "
                f"FROM finbert_cache WHERE url_hash IN ({placeholders})",
                tuple(chunk),
            )
            if df.empty:
                continue
            for _, row in df.iterrows():
                result[row["url_hash"]] = {
                    "fb_positive": float(row["fb_positive"]),
                    "fb_negative": float(row["fb_negative"]),
                    "fb_neutral": float(row["fb_neutral"]),
                    "fb_label": row.get("fb_label", "neutral"),
                    "fb_score": float(row.get("fb_score") or 0.0),
                }
        except Exception as e:
            logger.warning(f"finbert_cache read failed: {e}")
    return result


def write_cached_scores(router, scores: list[dict]) -> int:
    """Write a batch of FinBERT scores to the cache.

    Each dict in scores must have keys:
        url_hash, fb_positive, fb_negative, fb_neutral
    Optional: article_id

    Returns count of rows written.
    """
    if not scores:
        return 0
    now = datetime.now().isoformat()
    written = 0
    for s in scores:
        try:
            fb_pos = float(s.get("fb_positive", 0))
            fb_neg = float(s.get("fb_negative", 0))
            fb_score = fb_pos - fb_neg
            probs = {"positive": fb_pos, "negative": fb_neg,
                     "neutral": float(s.get("fb_neutral", 1))}
            fb_label = max(probs, key=probs.get)
            router.execute(
                "INSERT OR REPLACE INTO finbert_cache "
                "(url_hash, article_id, fb_positive, fb_negative, fb_neutral, "
                "fb_label, fb_score, scored_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    s["url_hash"],
                    s.get("article_id"),
                    fb_pos,
                    fb_neg,
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

    # Step 1: Compute url_hashes
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
            # Update in-memory cached dict for Step 5
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
    uncached_set = set(uncached_indices) if uncached_indices else set()
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
            "from_cache": i not in uncached_set,
        })

    return results
