import re
import pandas as pd
import spacy
from pathlib import Path
import html
from src.utils.logger import setup_logger
logger = setup_logger(__name__)



try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    logger.warning("spaCy model 'en_core_web_sm' not found. Attempting to download...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
EMOJI_RE = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)


def clean_comment(text: str) -> str:
    """Clean and preprocess a single comment."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha]

    return " ".join(tokens)


def process_raw_json(json_path, out_path="data/cleaned"):
    """Load raw scraped JSON, clean it, save as CSV."""
    
    json_path = Path(json_path)
    
    if not json_path.exists():
        logger.error(f"File not found: {json_path}")
        return None
    
    try:
        df = pd.read_json(json_path)
    except ValueError as e:
        logger.error(f"Invalid JSON format: {e}")
        return None
    except Exception as e:
        logger.info(f"Error reading JSON: {e}")
        return None
    
    if df.empty:
        logger.error("JSON file contains no data")
        return None
    
    if "text_raw" not in df.columns:
        logger.error("'text_raw' column not found in JSON")
        return None
    
    logger.info("Cleaning comments...")
    df["text_clean"] = df["text_raw"].apply(clean_comment)
    
    original_count = len(df)
    df = df[df["text_clean"].str.len() > 2]
    filtered_count = len(df)
    
    logger.info(f"Filtered {original_count - filtered_count} comments (too short)")
    
    if df.empty:
        logger.warning("No comments remain after cleaning")
        return None
    
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    
    video_id = json_path.stem
    csv_path = out_path / f"{video_id}.csv"
    
    try:
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"Successfully saved {filtered_count} cleaned comments to: {csv_path}")
    except Exception as e:
        logger.info(f"Error saving CSV: {e}")
        return None
    
    return str(csv_path)


if __name__ == "__main__":
    path = input("Enter raw JSON path: ")
    result = process_raw_json(path)
    if result:
        logger.info(f"\nOutput: {result}")
    else:
        logger.info("\nProcessing failed.")