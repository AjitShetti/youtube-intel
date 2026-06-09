import argparse
import sys
from pathlib import Path
from src.config import get_config
from src.scraper.youtube_scraper import fetch_comments
from src.preprocessing.clean_text import process_raw_json
from src.sentiment.sentiment_model import SentimentAnalyzer
from src.embeddings.embedder import Embedder
from src.topic_modeling.topic_model import TopicModeler
from src.utils.logger import setup_logger
logger = setup_logger(__name__)



def run_pipeline(url, steps):
    logger.info(f"Starting pipeline for URL: {url}")
    
    # Extract video ID at the start - needed for all file paths
    from src.scraper.youtube_scraper import get_video_id
    video_id = get_video_id(url)
    logger.info(f"Video ID: {video_id}")
    
    # Ensure necessary directories exist
    for dir_name in ["data/raw", "data/cleaned", "data/embeddings", "data/topics"]:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Scrape
    if "scrape" in steps or "all" in steps:
        logger.info("\n--- Step 1: Scraping Comments ---")
        raw_json_path = fetch_comments(url)
        if not raw_json_path:
            logger.info("Scraping failed.")
            return
        logger.info(f"Scraped data saved to: {raw_json_path}")
    else:
        raw_json_path = f"data/raw/{video_id}.json"
        if not Path(raw_json_path).exists():
             logger.info(f"Raw data not found at {raw_json_path}. Cannot proceed.")
             return

    # Step 2: Preprocess
    if "preprocess" in steps or "all" in steps:
        logger.info("\n--- Step 2: Preprocessing ---")
        cleaned_csv_path = process_raw_json(raw_json_path)
        if not cleaned_csv_path:
            logger.info("Preprocessing failed.")
            return
    else:
        cleaned_csv_path = f"data/cleaned/{video_id}.csv"
        if not Path(cleaned_csv_path).exists():
             logger.info(f"Cleaned data not found at {cleaned_csv_path}. Cannot proceed.")
             return

    # Step 3: Sentiment
    if "sentiment" in steps or "all" in steps:
        logger.info("\n--- Step 3: Sentiment Analysis ---")
        analyzer = SentimentAnalyzer()
        # Overwrite the cleaned CSV with sentiment data
        df = analyzer.process_csv(cleaned_csv_path, out_path=cleaned_csv_path)
        logger.info("Sentiment analysis complete.")

    # Step 4: Embeddings
    if "embeddings" in steps or "all" in steps:
        logger.info("\n--- Step 4: Generating Embeddings ---")
        embedder = Embedder()
        result = embedder.process_csv(cleaned_csv_path)
        if not result:
            logger.info("Embedding generation failed.")
            return
        emb_path, map_path = result
    else:
        emb_path = f"data/embeddings/{video_id}_embeddings.npy"
        if not Path(emb_path).exists():
             logger.info(f"Embeddings not found at {emb_path}. Cannot proceed.")
             return

    # Step 5: Topic Modeling
    if "topics" in steps or "all" in steps:
        logger.info("\n--- Step 5: Topic Modeling ---")
        modeler = TopicModeler()
        modeler.run(cleaned_csv_path, emb_path)
    else:
        topics_path = f"data/topics/{video_id}_topics.csv"
        if not Path(topics_path).exists():
             logger.info(f"Topics data not found at {topics_path}. Cannot proceed.")
             return None

    # Define output paths
    topics_path = f"data/topics/{video_id}_topics.csv"
    topic_info_path = f"data/topics/{video_id}_topic_info.csv"

    logger.info("\nPipeline completed successfully!")
    
    return {
        "topics_path": topics_path,
        "cleaned_csv_path": cleaned_csv_path,
        "topic_info_path": topic_info_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Intelligence Pipeline")
    parser.add_argument("--url", required=True, help="YouTube Video URL")
    parser.add_argument("--steps", nargs="+", default=["all"], 
                        choices=["scrape", "preprocess", "sentiment", "embeddings", "topics", "all"],
                        help="Pipeline steps to run")
    
    args = parser.parse_args()
    run_pipeline(args.url, args.steps)
