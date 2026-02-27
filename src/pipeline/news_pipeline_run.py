"""News Pipeline Orchestrator — Fetch, process, train, predict.

Usage:
    python -m src.pipeline.news_pipeline_run
"""

import logging
import sqlite3
import sys
import os

import yaml
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.news_fetcher import NewsFetcher
from src.data.news_features import NewsFeatureProcessor
from src.model.news_predictor import NewsPredictor

logger = logging.getLogger(__name__)


def run_news_pipeline(config: dict = None):
    """Execute the full news pipeline: fetch → process → train → store."""
    if config is None:
        with open("config.yaml") as f:
            config = yaml.safe_load(f) or {}

    if not config.get("news_pipeline", {}).get("enabled", True):
        logger.info("News pipeline disabled in config")
        return

    logger.info("=== News Pipeline Start ===")

    # Step 1: Fetch articles
    fetcher = NewsFetcher(config)
    count = fetcher.fetch_all()
    logger.info(f"Step 1: Fetched {count} new articles")

    # Step 2: Process features
    processor = NewsFeatureProcessor(config)
    article_df = processor.process_articles()
    logger.info(f"Step 2: Processed {len(article_df)} articles")

    if article_df.empty:
        logger.warning("No articles to process — skipping training")
        fetcher.close()
        processor.close()
        return

    # Step 3: Build daily features and store
    daily_df = processor.build_daily_features()
    processor.store_features(daily_df)
    logger.info(f"Step 3: Built {len(daily_df)} daily feature rows")

    # Step 4: Train news predictor (if enough data)
    texts = (article_df["headline"] + " " + article_df.get("clean_text", "")).tolist()
    tfidf = processor.transform_tfidf(texts)
    processor.save_vectorizer()

    # Get price data for targets
    try:
        db_path = config.get("database", {}).get("path", "./data/spy.db")
        conn = sqlite3.connect(db_path)
        prices = pd.read_sql_query("SELECT date, close FROM prices ORDER BY date", conn)
        conn.close()
    except Exception as e:
        logger.warning(f"Could not load prices for target creation: {e}")
        prices = pd.DataFrame()

    if not prices.empty and len(article_df) >= 20:
        predictor = NewsPredictor(horizon_minutes=60)
        X = predictor.prepare_features(tfidf, article_df)
        y = predictor.create_targets(prices, article_df["date"].tolist())

        if len(y) > 0 and len(set(y)) > 1:
            metrics = predictor.train(X, y)
            predictor.save()
            logger.info(f"Step 4: Model trained — {metrics}")
        else:
            logger.warning("Step 4: Not enough target diversity for training")
    else:
        logger.info("Step 4: Skipped training (insufficient data)")

    fetcher.close()
    processor.close()
    logger.info("=== News Pipeline Complete ===")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    run_news_pipeline()
