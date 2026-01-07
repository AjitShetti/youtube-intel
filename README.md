# YouTube Intelligence Analysis

A comprehensive project for scraping, analyzing, and visualizing YouTube comments using NLP and sentiment analysis techniques.

## Project Overview

This project provides an end-to-end pipeline for:
- Scraping YouTube comments
- Text preprocessing and cleaning
- Sentiment analysis (rule-based + ML)
- Topic modeling using BERTopic/LDA
- Generating embeddings with BERT/SentenceTransformer
- Interactive dashboard visualization

## Project Structure

```
youtube-intel/
├── data/                    # Data storage
│   ├── raw/                # Raw scraped comments
│   ├── cleaned/            # Preprocessed text data
│   └── embeddings/         # Saved embeddings (.npy/.pkl)
├── src/                    # Source code
│   ├── scraper/            # YouTube scraper
│   ├── preprocessing/      # Text cleaning
│   ├── sentiment/          # Sentiment analysis
│   ├── topic_modeling/     # Topic discovery
│   ├── embeddings/         # Embedding generation
│   ├── dashboard/          # Streamlit/Dash UI
│   └── utils.py            # Utility functions
├── notebooks/              # Jupyter notebooks
│   ├── 01_eda.ipynb       # Exploratory Data Analysis
│   ├── 02_sentiment_dev.ipynb
│   └── 03_topic_modeling_dev.ipynb
├── requirements.txt        # Python dependencies
├── config.yaml            # Configuration file
└── README.md              # This file
```

## Installation

1. Clone the repository or create the project directory
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure settings in `config.yaml`

## Usage

### Data Collection
```python
from src.scraper.youtube_scraper import scrape_video_comments
comments = scrape_video_comments(video_id="YOUR_VIDEO_ID")
```

### Text Preprocessing
```python
from src.preprocessing.clean_text import clean_text
cleaned = clean_text(raw_text)
```

### Sentiment Analysis
```python
from src.sentiment.sentiment_model import predict_sentiment
sentiment = predict_sentiment(text)
```

### Topic Modeling
```python
from src.topic_modeling.topic_model import fit_topic_model
model = fit_topic_model(documents)
```

### Generate Embeddings
```python
from src.embeddings.embedder import generate_embeddings
embeddings = generate_embeddings(texts)
```

### Run Dashboard
```bash
streamlit run src/dashboard/app.py
```

## Notebooks

- **01_eda.ipynb**: Exploratory data analysis
- **02_sentiment_dev.ipynb**: Sentiment model development
- **03_topic_modeling_dev.ipynb**: Topic modeling pipeline

## Configuration

Update `config.yaml` for:
- YouTube API credentials
- Model parameters
- Data paths
- Processing settings

## Requirements

See `requirements.txt` for all dependencies.

## License

This project is open source and available under the MIT License.
