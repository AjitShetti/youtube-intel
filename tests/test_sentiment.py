import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# We patch the model loading so it doesn't download weights during tests
@patch("src.sentiment.sentiment_model.AutoModelForSequenceClassification.from_pretrained")
@patch("src.sentiment.sentiment_model.AutoTokenizer.from_pretrained")
def test_sentiment_analyzer_process_csv(mock_tokenizer, mock_model, tmp_path):
    from src.sentiment.sentiment_model import SentimentAnalyzer
    
    # Create dummy CSV
    csv_file = tmp_path / "cleaned.csv"
    df = pd.DataFrame({
        "comment_id": [1, 2],
        "text_clean": ["great video", "terrible content"]
    })
    df.to_csv(csv_file, index=False)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(device="cpu")
    
    # Mock predict method to avoid real inference
    analyzer.predict = MagicMock(side_effect=[("positive", 0.95), ("negative", 0.88)])
    
    out_file = tmp_path / "enriched.csv"
    
    # Run method
    result_df = analyzer.process_csv(str(csv_file), out_path=str(out_file))
    
    # Asserts
    assert "sentiment" in result_df.columns
    assert "sentiment_score" in result_df.columns
    assert result_df["sentiment"].iloc[0] == "positive"
    assert result_df["sentiment"].iloc[1] == "negative"
    
    # Check if file was saved
    assert out_file.exists()
    saved_df = pd.read_csv(out_file)
    assert "sentiment" in saved_df.columns
