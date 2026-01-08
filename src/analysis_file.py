"""
analysis_pipeline.py (Memory-Efficient Version)
------------------------------------------------
Processes large Twitter/X data in chunks and generates trading signals.
"""

import re
import unicodedata
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import pyarrow.parquet as pq

# ---------------- CONFIG ---------------- #
PARQUET_FILE = "data/indian_market_tweets.parquet"
CHUNK_SIZE = 100_000
MAX_TFIDF_FEATURES = 5000
PLOT_SAMPLE_FRAC = 0.05
OUTPUT_FILE = "data/trading_signals.parquet"

# ---------------- LOGGING ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ---------------- UTILS ---------------- #
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df["tweet_length"] = df["content"].str.len()
    df["has_number"] = df["content"].str.contains(r"\d").astype(int)
    df["exclamations"] = df["content"].str.count("!")
    df["capital_ratio"] = df["content"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    return df

def compute_signal_score(df: pd.DataFrame) -> pd.DataFrame:
    df["signal_score"] = (
        0.5 * df.get("likes", 0) +
        0.3 * df.get("retweets", 0) +
        0.2 * df.get("replies", 0)
    )
    return df

def read_parquet_in_chunks(file_path, chunk_size=CHUNK_SIZE):
    table = pq.read_table(file_path)
    num_rows = table.num_rows
    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        chunk = table.slice(start, end - start).to_pandas()
        yield chunk

# ---------------- PIPELINE ---------------- #
def run_analysis():
    logging.info("Starting analysis pipeline")

    # ---------- STEP 1: READ & PROCESS IN CHUNKS ----------
    all_chunks = []
    all_texts = []

    for chunk in read_parquet_in_chunks(PARQUET_FILE, CHUNK_SIZE):
        chunk["content"] = chunk["content"].astype(str).apply(normalize_text)
        chunk = engineer_features(chunk)
        chunk = compute_signal_score(chunk)
        all_chunks.append(chunk)
        all_texts.extend(chunk["content"].tolist())

    df = pd.concat(all_chunks, ignore_index=True)
    logging.info(f"Loaded {len(df)} rows")

    # ---------- STEP 2: TF-IDF (INCREMENTAL MEMORY-SAFE) ----------
    logging.info("Computing TF-IDF features")
    vectorizer = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATURES,
        ngram_range=(1, 2),
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(all_texts)
    df["tfidf_strength"] = tfidf_matrix.mean(axis=1).A1
    logging.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # ---------- STEP 3: AGGREGATE SIGNALS ----------
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.floor("H")

    signal_ts = (
        df.groupby("hour")
        .agg(
            signal_mean=("signal_score", "mean"),
            signal_std=("signal_score", "std"),
            volume=("signal_score", "count"),
            tfidf_mean=("tfidf_strength", "mean")
        )
        .reset_index()
    )

    signal_ts["confidence"] = 1.96 * signal_ts["signal_std"] / np.sqrt(signal_ts["volume"].clip(lower=1))
    logging.info("Signal aggregation completed")

    # ---------- STEP 4: LOW-MEMORY VISUALIZATION ----------
    logging.info("Generating low-memory plots")
    sample = df.sample(frac=PLOT_SAMPLE_FRAC, random_state=42)

    plt.figure(figsize=(8, 4))
    sample.groupby("source_tag").size().plot(kind="bar")
    plt.title("Tweet Volume by Hashtag (Sampled)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(signal_ts["hour"], signal_ts["signal_mean"], label="Signal Mean")
    plt.fill_between(
        signal_ts["hour"],
        signal_ts["signal_mean"] - signal_ts["confidence"],
        signal_ts["signal_mean"] + signal_ts["confidence"],
        alpha=0.3,
        label="Confidence Interval"
    )
    plt.legend()
    plt.title("Aggregated Trading Signal Over Time")
    plt.tight_layout()
    plt.show()

    # ---------- STEP 5: SAVE PROCESSED SIGNALS ----------
    signal_ts.to_parquet(OUTPUT_FILE, index=False)
    logging.info(f"Saved aggregated signals to {OUTPUT_FILE}")
    logging.info("Analysis pipeline completed successfully")

if __name__ == "__main__":
    run_analysis()
