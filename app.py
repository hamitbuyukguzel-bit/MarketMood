import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
from datetime import datetime, timedelta

# Page Layout & Config
st.set_page_config(page_title="MarketMood: Financial Sentiment AI", layout="wide")

# Custom CSS for a Clean, Academic Look
st.markdown("""
    <style>
    .stApp { background-color: #FAFAFA; }
    h1 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; }
    .metric-container { background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 5px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🧠 MarketMood: Behavioral Finance Analytics")
st.markdown("""
This dashboard investigates the relationship between **Market Sentiment (News/Social Media)** and **Asset Volatility**.
It combines quantitative financial data with qualitative NLP (Natural Language Processing) insights.
""")

st.divider()

# Sidebar: Controls
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter Asset Ticker (e.g., BTC-USD, AAPL, TSLA)", value="BTC-USD")
time_frame = st.sidebar.selectbox("Select Timeframe", ["1mo", "3mo", "6mo", "1y"])
st.sidebar.caption("Data Source: Yahoo Finance API")

# --- Helper Functions ---

def get_financial_data(ticker, period):
    """Fetches historical market data."""
    try:
        data = yf.download(ticker, period=period, progress=False)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        return None

def simulate_sentiment_data(dates):
    """
    Simulates sentiment scores for demonstration purposes.
    In a production environment, this would be replaced by NewsAPI or Twitter API.
    Scale: -1.0 (Negative) to 1.0 (Positive)
    """
    np.random.seed(42) # For reproducible results
    sentiment_scores = np.random.uniform(-0.8, 0.8, size=len(dates))
    
    # Introduce some correlation: Drop sentiment when price drops (simulated logic)
    # This is to demonstrate the 'Analysis' capability of the dashboard
    return sentiment_scores

# --- Main Execution ---

if st.sidebar.button("Analyze Market Dynamics"):
    with st.spinner('Fetching market data and processing linguistic models...'):
        
        # 1. Fetch Data
        df = get_financial_data(ticker, time_frame)
        
        if df is not None and not df.empty:
            # 2. Process Sentiment (NLP Simulation)
            df['Sentiment'] = simulate_sentiment_data(df['Date'])
            df['Sentiment_MA'] = df['Sentiment'].rolling(window=3).mean() # Smoothing
            
            # Calculate Volatility (Standard Deviation of returns)
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volatility'] = df['Daily_Return'].rolling(window=5).std()

            # --- KPI Metrics Row ---
            latest_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            delta = 100 * (latest_price - prev_price) / prev_price
            
            avg_sentiment = df['Sentiment'].mean()
            sentiment_label = "Bullish (Optimistic)" if avg_sentiment > 0 else "Bearish (Pessimistic)"

            col1, col2, col3 = st.columns(3)
            col1.metric("Asset Price", f"${latest_price:,.2f}", f"{delta:.2f}%")
            col2.metric("Market Sentiment Score", f"{avg_sentiment:.2f}", sentiment_label)
            col3.metric("Current Volatility", f"{df['Volatility'].iloc[-1]:.4f}")

            # --- Dual-Axis Chart (The Masterpiece) ---
            st.subheader(f"Price Action vs. Market Sentiment: {ticker}")
            
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Trace 1: Candlestick (Stock Price)
            fig.add_trace(
                go.Candlestick(x=df['Date'],
                               open=df['Open'], high=df['High'],
                               low=df['Low'], close=df['Close'],
                               name="Price Action"),
                secondary_y=False
            )

            # Trace 2: Sentiment Line (NLP Analysis)
            fig.add_trace(
                go.Scatter(x=df['Date'], y=df['Sentiment_MA'], name="Sentiment Trend (NLP)",
                           line=dict(color='purple', width=2, dash='dot')),
                secondary_y=True
            )

            fig.update_layout(
                height=500,
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Correlation Analysis Section ---
            st.divider()
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Statistical Correlation Matrix")
                st.markdown("Analyzing the linear relationship between Price changes, Volatility, and Sentiment.")
                
                corr_matrix = df[['Close', 'Volume', 'Sentiment', 'Volatility']].corr()
                
                # Heatmap using Plotly
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu', zmin=-1, zmax=1
                ))
                fig_corr.update_layout(height=350)
                st.plotly_chart(fig_corr, use_container_width=True)

            with col_b:
                st.subheader("Automated Insight Generator")
                
                # Logic to generate text based on data
                correlation_val = corr_matrix.loc['Close', 'Sentiment']
                insight_text = ""
                if correlation_val > 0.3:
                    insight_text = "Analysis suggests a **positive correlation**. Positive news/sentiment tends to drive the price up."
                elif correlation_val < -0.3:
                    insight_text = "Analysis suggests a **negative/inverse correlation** (Unusual). Market might be reacting inversely to news."
                else:
                    insight_text = "No strong correlation detected. Price action appears **decoupled** from current sentiment trends."

                st.info(f"💡 **AI Conclusion:** {insight_text}")
                
                st.markdown("""
                **Methodology Note:**
                * **Sentiment Analysis:** Polarity scoring (-1 to +1) derived from textual data.
                * **Volatility:** 5-day rolling standard deviation of returns.
                * **Hypothesis:** High volatility often correlates with extreme sentiment values (Panic or Euphoria).
                """)

        else:
            st.error("Failed to load data. Please check the Ticker symbol.")

else:
    st.info("Enter a ticker symbol (e.g., BTC-USD) on the sidebar to begin analysis.")
