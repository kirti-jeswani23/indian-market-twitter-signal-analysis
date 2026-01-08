# Indian Market Twitter Signal Analysis 

## Project Overview
This project collects real-time Twitter/X data related to the Indian stock market
and converts unstructured social media text into quantitative trading signals.

The pipeline includes:
- Authenticated Twitter scraping using Playwright
- Large-scale data storage using Parquet
- Memory-efficient text processing
- Feature engineering and TF-IDF analysis
- Aggregated sentiment-based trading signals

The project is designed with scalability, efficiency, and real-world constraints in mind.

---

## Tech Stack
- Python
- Playwright
- Pandas, NumPy
- Scikit-learn
- PyArrow (Parquet)
- Matplotlib

---

## Repository Structure
src/ → Source code (scraping + analysis)
data/ → Sample output data
logs/ → Scraper logs

---

## Setup Instructions

1. Clone the repository

git clone https://github.com/kirti-jeswani23/indian-market-twitter-signal-analysis.git
cd indian-market-twitter-signal-analysis

2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

 ## Running the Project

1. Save Twitter Session

python src/save_session.py

Login manually when the browser opens.

2. Scrape Tweets

python src/twitter_scraper.py

3. Run Analysis Pipeline

python src/analysis_pipeline.py
  
 ## Output

indian_market_tweets.parquet → Raw tweet data

trading_signals.parquet → Aggregated trading signals
