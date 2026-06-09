import numpy as np
import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from src.config import get_config
from src.utils.logger import setup_logger
logger = setup_logger(__name__)




class TopicModeler:
    def __init__(self, embed_model=None):
        """
        Initialize the topic modeler with a sentence transformer model.
        
        Args:
            embed_model: Name of the sentence transformer model
        """
        if embed_model is None:
            embed_model = get_config("embeddings.model_name", "all-mpnet-base-v2")
        self.embed_model_name = embed_model
        
        try:
            self.embedder = SentenceTransformer(embed_model)
            logger.info(f"Loaded embedding model: {embed_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def load_inputs(self, csv_path, emb_path):
        """
        Load cleaned comments CSV and pre-computed embeddings.
        
        Args:
            csv_path: Path to cleaned comments CSV
            emb_path: Path to embeddings numpy file
            
        Returns:
            Tuple of (DataFrame, texts, embeddings)
        """
        csv_path = Path(csv_path)
        emb_path = Path(emb_path)
        
        # Validate file existence
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
        
        # Load CSV
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")
        
        # Validate DataFrame
        if df.empty:
            raise ValueError("CSV file contains no data")
        
        if "text_clean" not in df.columns:
            raise ValueError("CSV missing 'text_clean' column")
        
        # Load embeddings
        try:
            embeddings = np.load(emb_path)
        except Exception as e:
            raise ValueError(f"Failed to load embeddings: {e}")
        
        # Validate embeddings
        if embeddings.size == 0:
            raise ValueError("Embeddings array is empty")
        
        # Extract texts
        texts = df["text_clean"].fillna("").astype(str).tolist()
        
        # Filter out empty texts and corresponding embeddings
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        
        if not valid_indices:
            raise ValueError("No valid texts found after filtering")
        
        texts = [texts[i] for i in valid_indices]
        embeddings = embeddings[valid_indices]
        df = df.iloc[valid_indices].reset_index(drop=True)
        
        # Validate dimension match
        if len(texts) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(texts)} texts but {len(embeddings)} embeddings"
            )
        
        logger.info(f"Loaded {len(texts)} texts and embeddings")
        
        return df, texts, embeddings
    
    def fit(self, texts, embeddings):
        """
        Fit BERTopic model on texts using pre-computed embeddings.
        
        Args:
            texts: List of text strings
            embeddings: Numpy array of embeddings
            
        Returns:
            Tuple of (topic_model, topics, probabilities)
        """
        if len(texts) < 10:
            logger.warning("Very few texts (<10) - results may not be meaningful")
        
        try:
            # Create BERTopic model
            # Note: We pass embedding_model but will use pre-computed embeddings
            representation_model = KeyBERTInspired()
            topic_model = BERTopic(
                embedding_model=self.embedder,
                representation_model=representation_model,
                n_gram_range=(1, 2),
                min_topic_size=min(10, max(2, len(texts) // 20)),  # Adaptive min_topic_size
                verbose=True,
                calculate_probabilities=True
            )
            
            logger.info("Fitting BERTopic model...")
            topics, probs = topic_model.fit_transform(texts, embeddings)
            
            # Generate and set clean custom labels for topics
            topic_labels = topic_model.generate_topic_labels(nr_words=3, topic_prefix=False, word_length_limit=15, separator=", ")
            topic_model.set_topic_labels(topic_labels)
            
            # Get topic info
            num_topics = len(set(topics)) - (1 if -1 in topics else 0)
            num_outliers = sum(1 for t in topics if t == -1)
            
            logger.info(f"Discovered {num_topics} topics")
            logger.info(f"{num_outliers} outlier documents (topic -1)")
            
            return topic_model, topics, probs
            
        except Exception as e:
            raise RuntimeError(f"Failed to fit topic model: {e}")
    
    def save_outputs(self, topic_model, df, topics, probs, out_dir, video_id):
        """
        Save the trained model and results.
        
        Args:
            topic_model: Trained BERTopic model
            df: DataFrame with comments
            topics: Topic assignments
            probs: Topic probabilities
            out_dir: Output directory
            video_id: Video identifier
            
        Returns:
            Tuple of (model_path, info_path, summary_path)
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = out_dir / f"{video_id}_bertopic_model"
        info_path = out_dir / f"{video_id}_topics.csv"
        summary_path = out_dir / f"{video_id}_topic_summary.txt"
        
        # Save model
        try:
            topic_model.save(str(model_path))
            logger.info(f"Saved BERTopic model -> {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
        
        # Attach topics to dataframe
        try:
            df = df.copy()
            df["topic"] = topics
            
            # Add top probability if available
            if probs is not None and len(probs) > 0:
                df["topic_probability"] = [max(prob) if len(prob) > 0 else 0.0 for prob in probs]
            
            df.to_csv(info_path, index=False, encoding="utf-8")
            logger.info(f"Saved topics CSV -> {info_path}")
        except Exception as e:
            logger.error(f"Failed to save topics CSV: {e}")
            raise
        
        # Save topic summary
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(f"Video ID: {video_id}\n")
                f.write(f"Total Comments: {len(df)}\n")
                f.write(f"Model: {self.embed_model_name}\n\n")
                
                # Topic distribution
                topic_counts = df["topic"].value_counts().sort_index()
                num_topics = len(topic_counts) - (1 if -1 in topic_counts.index else 0)
                
                f.write(f"Number of Topics: {num_topics}\n")
                f.write(f"Outliers (topic -1): {topic_counts.get(-1, 0)}\n\n")
                
                f.write("=" * 50 + "\n")
                f.write("TOPIC SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                # Get topic info
                topic_info = topic_model.get_topic_info()
                
                for _, row in topic_info.iterrows():
                    topic_id = row["Topic"]
                    if topic_id == -1:
                        continue
                    
                    count = row["Count"]
                    # Get top words for this topic
                    topic_words = topic_model.get_topic(topic_id)
                    
                    if topic_words:
                        top_words = ", ".join([word for word, _ in topic_words[:5]])
                        f.write(f"Topic {topic_id} ({count} comments):\n")
                        f.write(f"  Keywords: {top_words}\n\n")
            
            logger.info(f"Saved topic summary -> {summary_path}")
        except Exception as e:
            logger.warning(f"Failed to save summary: {e}")

        # Save topic info CSV (for dashboard)
        topic_info_path = out_dir / f"{video_id}_topic_info.csv"
        try:
            topic_info = topic_model.get_topic_info()
            topic_info.to_csv(topic_info_path, index=False)
            logger.info(f"Saved topic info CSV -> {topic_info_path}")
        except Exception as e:
            logger.warning(f"Failed to save topic info CSV: {e}")
        
        return str(model_path), str(info_path), str(summary_path)
    
    def run(self, csv_path, emb_path, out_dir="data/topics"):
        """
        Run the complete topic modeling pipeline.
        
        Args:
            csv_path: Path to cleaned comments CSV
            emb_path: Path to embeddings numpy file
            out_dir: Output directory for results
            
        Returns:
            Tuple of output paths or None on failure
        """
        video_id = Path(csv_path).stem
        
        try:
            logger.info("Loading inputs...")
            df, texts, embeddings = self.load_inputs(csv_path, emb_path)
            
            logger.info("Fitting topics... this may take a few minutes")
            topic_model, topics, probs = self.fit(texts, embeddings)
            
            logger.info("Saving outputs...")
            return self.save_outputs(topic_model, df, topics, probs, out_dir, video_id)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    csv_path = input("Enter cleaned CSV path: ")
    emb_path = input("Enter embeddings path (.npy): ")
    
    try:
        modeler = TopicModeler()
        result = modeler.run(csv_path, emb_path)
        
        if result:
            model_path, info_path, summary_path = result
            logger.info(f"\n✓ Topic modeling complete!")
            logger.info(f"  Model: {model_path}")
            logger.info(f"  Topics CSV: {info_path}")
            logger.info(f"  Summary: {summary_path}")
        else:
            logger.info("\n✗ Topic modeling failed.")
    except Exception as e:
        logger.info(f"\n✗ Error: {e}")