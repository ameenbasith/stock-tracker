# Stock Tracker

A comprehensive machine learning-based stock price prediction tool that combines technical analysis, feature engineering, and advanced ML algorithms to forecast market movements.

## 📌 Project Overview
This application provides a robust framework for analyzing historical stock data, generating technical indicators, training predictive models, and visualizing market trends through an interactive dashboard. Designed to aid investment decision-making, it provides data-driven insights into potential future price movements.

## 🔑 Key Features

### 📥 Data Collection
- Seamless integration with Yahoo Finance API for historical and real-time market data
- Support for customizable time periods (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
- Reliable storage of stock data in both CSV and SQLite formats
- Automated cleaning and preprocessing of raw datasets

### 🧠 Feature Engineering
**Price-Based Indicators**
- Simple and Exponential Moving Averages (5, 10, 20, 50, 200 days)
- Price Channels, Support and Resistance levels
- Volatility measures including daily return volatility and price ranges

**Momentum Indicators**
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Stochastic Oscillator
- Price Momentum & Rate of Change

**Volatility Indicators**
- Bollinger Bands
- Average True Range (ATR)
- Rolling Standard Deviation

### 🤖 Machine Learning Pipeline
- Trains Random Forest and XGBoost regression models
- Advanced feature selection using importance metrics
- Time-series aware cross-validation to avoid data leakage
- Evaluation metrics: RMSE, MAE, R², and Directional Accuracy
- Model persistence with `joblib` for continuous forecasting

### 📊 Interactive Dashboard
- Built with Streamlit for ease of use and interactivity
- Dynamic price charts with overlays of selected indicators
- Visual representation of predictions with confidence intervals
- Backtesting results and trading signal accuracy display
- Multi-stock comparison and filterable insights

## 🏗️ Technical Architecture
```
stock-tracker/
│
├── data/                   # Data storage
│   ├── raw/                # Unprocessed stock price data
│   └── processed/          # Engineered features and transformed data
│
├── models/                 # Serialized ML models
│   ├── random_forest/      # Random Forest predictions
│   └── xgboost/            # XGBoost predictions
│
├── src/                    # Main application code
│   ├── data_collection.py  # Data ingestion & validation
│   ├── feature_engineering.py  # Feature extraction
│   ├── model_training.py   # Model training logic
│   └── utils.py            # Utility functions
│
├── notebooks/              # Jupyter Notebooks for EDA & analysis
│   ├── exploratory_analysis.ipynb
│   └── model_evaluation.ipynb
│
├── tests/                  # Unit tests
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
├── app.py                  # Streamlit app launcher
├── requirements.txt        # Dependencies list
├── setup.py                # Installation script
└── README.md               # Project documentation
```

## ⚙️ Installation and Setup
### Prerequisites
- Python 3.8+
- pip
- Git

### Setup Steps
```bash
# Clone the repository
git clone https://github.com/yourusername/stock-tracker.git
cd stock-tracker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p data/raw data/processed
```

## 🚀 Usage Instructions
### Data Collection
```bash
python src/data_collection.py --tickers AAPL,MSFT,GOOGL --period 2y
```

### Feature Engineering
```bash
python src/feature_engineering.py --input data/raw/AAPL_20250423.csv --output data/processed/AAPL_features.csv
```

### Model Training
```bash
python src/model_training.py --ticker AAPL --target Pct_Change_1 --model random_forest
```

### Run Dashboard
```bash
streamlit run app.py
```

## 📈 Performance Evaluation
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)
- Directional Accuracy (e.g., correct up/down prediction %)
- Simulated Profit/Loss calculations for strategy backtesting

**Current Results:** ~55–65% directional accuracy on major stocks.

## 🔮 Future Enhancements
### Short-term
- Sentiment analysis from financial news (e.g., FinBERT, NewsAPI)
- Backtesting engine with visualization of trades
- Enhanced charting (candlestick, heatmaps, sector views)

### Long-term
- Integration of deep learning models (LSTM, Transformers)
- Real-time portfolio optimization module
- Deployment as a web service with prediction API
- Mobile-friendly interface

## 👥 Contributors
- Ameen (https://github.com/ameenbasith)

## 📄 License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Yahoo Finance for market data
- Scikit-learn and XGBoost teams
- Streamlit for UI framework

> **Disclaimer:** This project is for educational and research purposes only. It should not be considered financial advice. Please consult with a certified financial advisor before making investment decisions.
