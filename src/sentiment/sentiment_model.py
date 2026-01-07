import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from pathlib import Path
# from sentiment_model import SentimentAnalyzer  # Removed circular import


class SentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # RoBERTa has max_position_embeddings=514 (512 tokens + 2 special tokens)
        # Set to 510 to ensure final sequence with special tokens doesn't exceed 514
        self.tokenizer.model_max_length = 510
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model_name = model_name

    def predict(self, text: str):
        """Return sentiment label + score."""
        try:
            # Truncate to 510 tokens (final sequence will be 512 with special tokens)
            # This ensures we never exceed the model's 514 position embedding limit
            encoded = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=510,
                padding=False
            )
            
            # Additional safety check - should never trigger with correct settings
            if encoded['input_ids'].shape[1] > 514:
                encoded['input_ids'] = encoded['input_ids'][:, :514]
                encoded['attention_mask'] = encoded['attention_mask'][:, :514]
            
            encoded = encoded.to(self.device)
            
            with torch.no_grad():
                logits = self.model(**encoded).logits
            probs = F.softmax(logits, dim=1)[0]
        except Exception as e:
            print(f"[ERROR] Prediction failed for text: {str(text)[:50]}... Error: {e}")
            return "neutral", 0.0

        labels = ["negative", "neutral", "positive"]
        idx = torch.argmax(probs).item()
        return labels[idx], float(probs[idx])

    def process_csv(self, cleaned_csv_path, out_path=None):
        df = pd.read_csv(cleaned_csv_path)

        sentiments = []
        scores = []

        for text in df["text_clean"]:
            label, score = self.predict(str(text))
            sentiments.append(label)
            scores.append(score)

        df["sentiment"] = sentiments
        df["sentiment_score"] = scores

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"[OK] Saved sentiment-enriched CSV -> {out_path}")

        return df
