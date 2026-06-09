import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.config import get_config
from src.utils.logger import setup_logger
logger = setup_logger(__name__)




class Embedder:
    def __init__(self, model_name=None, device=None):
        """
        Load the sentence-transformers embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
            device: 'cpu' or 'cuda' (auto-detect if None)
        """
        if model_name is None:
            model_name = get_config("embeddings.model_name", "all-mpnet-base-v2")
            
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.model_name = model_name
            
            # Show which device is being used
            actual_device = self.model.device
            logger.info(f"Loaded model '{model_name}' on device: {actual_device}")
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise
    
    def embed_texts(self, texts, batch_size=32):
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            logger.warning("No texts provided for embedding")
            return np.array([])
        
        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            logger.warning("No valid texts after filtering")
            return np.array([])
        
        if len(valid_texts) < len(texts):
            logger.info(f"Filtered out {len(texts) - len(valid_texts)} empty texts")
        
        try:
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def process_csv(self, csv_path, out_dir="data/embeddings"):
        """
        Load cleaned CSV → extract clean text → embed → save npy + mapping.
        
        Args:
            csv_path: Path to the cleaned comments CSV
            out_dir: Directory to save embeddings and metadata
            
        Returns:
            Tuple of (embedding_path, mapping_path) or None on failure
        """
        csv_path = Path(csv_path)
        
        # Validate input file
        if not csv_path.exists():
            logger.error(f"File not found: {csv_path}")
            return None
        
        # Load CSV with error handling
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            return None
        
        # Check if DataFrame is empty
        if df.empty:
            logger.error("CSV file contains no data")
            return None
        
        # Validate required columns
        required_columns = ["text_clean", "comment_id"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Handle NaN values and convert to string
        df["text_clean"] = df["text_clean"].fillna("")
        texts = df["text_clean"].astype(str).tolist()
        
        logger.info(f"Loaded {len(texts)} comments from CSV")
        
        # Generate embeddings
        try:
            embs = self.embed_texts(texts)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
        
        if len(embs) == 0:
            logger.error("No embeddings generated")
            return None
        
        # Prepare output directory
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output file paths
        video_id = csv_path.stem
        emb_path = out_dir / f"{video_id}_embeddings.npy"
        map_path = out_dir / f"{video_id}_mapping.csv"
        meta_path = out_dir / f"{video_id}_meta.txt"
        
        # Save embeddings
        try:
            np.save(emb_path, embs)
            logger.info(f"Saved embeddings -> {emb_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            return None
        
        # Save mapping (only rows with valid embeddings)
        try:
            # If we filtered some texts, only save the valid ones
            if len(embs) < len(df):
                valid_df = df[df["text_clean"].str.strip() != ""]
                valid_df[["comment_id", "text_clean"]].to_csv(
                    map_path, index=False, encoding="utf-8"
                )
            else:
                df[["comment_id", "text_clean"]].to_csv(
                    map_path, index=False, encoding="utf-8"
                )
            logger.info(f"Saved mapping -> {map_path}")
        except Exception as e:
            logger.error(f"Failed to save mapping: {e}")
            return None
        
        # Save metadata
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(f"model={self.model_name}\n")
                f.write(f"num_embeddings={len(embs)}\n")
                f.write(f"embedding_dim={embs.shape[1]}\n")
                f.write(f"source_csv={csv_path.name}\n")
            logger.info(f"Saved metadata -> {meta_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return None
        
        return str(emb_path), str(map_path)


if __name__ == "__main__":
    # Example usage
    csv_path = input("Enter cleaned CSV path: ")
    
    try:
        embedder = Embedder()
        result = embedder.process_csv(csv_path)
        
        if result:
            emb_path, map_path = result
            logger.info(f"\n✓ Processing complete!")
            logger.info(f"  Embeddings: {emb_path}")
            logger.info(f"  Mapping: {map_path}")
        else:
            logger.info("\n✗ Processing failed.")
    except Exception as e:
        logger.info(f"\n✗ Error: {e}")