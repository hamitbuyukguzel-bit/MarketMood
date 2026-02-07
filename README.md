# MarketMood: Behavioral Finance & Sentiment Analysis Engine

MarketMood is a financial analytics dashboard that bridges the gap between **Quantitative Finance** and **Behavioral Economics**. It explores the hypothesis that market prices are not solely driven by fundamentals but are significantly influenced by mass psychology and public sentiment.

## 🔬 Research Objective
In the era of the Digital Economy, information spreads instantly. This project aims to visualize the correlation between **Market Sentiment** (derived from NLP analysis) and **Asset Volatility**, challenging the traditional "Efficient Market Hypothesis" (EMH).

## 📊 Key Features

* **Real-Time Data Pipeline:** Fetches live financial data using the Yahoo Finance API (`yfinance`).
* **NLP Sentiment Scoring:** Implements a polarity-based sentiment scoring system (mock-up integrated for demo stability) to quantify market optimism vs. pessimism.
* **Interactive Visualization:** Uses `Plotly` for dynamic Candlestick charts overlaid with sentiment trends.
* **Statistical Correlation:** Automatically computes and visualizes the correlation matrix between Price, Volume, and Sentiment.

## 🛠 Tech Stack

* **Python 3.9+**
* **Streamlit:** Web Application Framework.
* **Yfinance:** Financial Market Data.
* **Plotly:** Interactive Financial Charting.
* **Pandas & NumPy:** Statistical Computing.

## 🚀 Installation & Usage

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/market-mood-analyzer.git](https://github.com/YOUR_USERNAME/market-mood-analyzer.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the dashboard:
    ```bash
    streamlit run app.py
    ```

## 📈 Future Roadmap (PhD Research Scope)
* **Integration of Twitter/X API:** To fetch real-time social media data for "Meme Stock" analysis.
* **LSTM Models:** To predict future volatility based on the lag between Sentiment spikes and Price reaction.
* **VADER Sentiment Analysis:** Refining the NLP model specifically for financial lexicon.

---
*Developed as part of a research initiative on Digital Economy and Market Behavior.*
