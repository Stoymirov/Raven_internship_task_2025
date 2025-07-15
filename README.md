# Raven_internship_task_2025

## Project overview
This project, developed for the Raven Scholarship and Internship Program, implements a trading algorithm designed to maximize profitability in financial markets. Built using Python, PyTorch, and Pandas, the algorithm leverages mathematical modeling, including differential equations and applied statistics, to analyze market data and optimize trading decisions. The solution focuses on generating consistent returns by identifying and capitalizing on market opportunities. This project demonstrates my ability to combine computational techniques and mathematical rigor to create effective, data-driven trading strategies.
# 01. Premise & research
This project, outlined in the Jupyter notebook `01.Premise & research.ipynb`, aims to develop and evaluate trading strategies to generate consistent profits using Binance top-of-book data (best bid and ask prices). The notebook serves as the foundational research and planning phase, where we define the problem, explore trading strategies, and establish a rigorous methodology based on the scientific method.

## Key components
- **Purpose**: Analyze trading strategies (mean reversion, momentum, market making) to identify profitable approaches using Binance top-of-book data.
- **Methodology**: Uses the scientific method:
  - **Question**: Can we profit using top-of-book data?
  - **Research**: Study strategies, indicators (e.g., RSI, MACD, Bollinger Bands), and concepts like volatility and risk management.
  - **Hypothesis**: Test a mean reversion strategy (buy below short-term moving average, sell above) and consider alternatives like momentum.
  - **Plan**: Implement strategies, backtest with historical data, and evaluate metrics (e.g., return, Sharpe ratio, drawdown).
- **Key Concepts**:
  - **Top-of-Book Data**: Best bid/ask prices for deciding buy/sell actions.
  - **Strategies**:
    - **Momentum**: Trade with trending prices; use indicators like RSI, MACD.
    - **Mean Reversion**: Trade against price extremes; use RSI, Bollinger Bands.
  - **Dynamic Position Sizing**: Adjust trade size based on volatility (ATR), conviction (momentum/volume), and drawdowns.
- **Theoretical Extension**: Models capital growth with differential equations for adaptive sizing.

## Next Steps
- Collect Binance top-of-book data.
- Code and backtest strategies (e.g., mean reversion with SMA).
- Analyze performance and refine strategies iteratively.

## Resources
- Binance API documentation
- BabyPips (trading education)
- Momentum vs. Mean Reversion article (xlearnonline.com)

# 02. Momentum Strategy
This Jupyter notebook (`02.Momentum strategy.ipynb`) builds on `01.Premise & research.ipynb` by implementing a momentum-based trading strategy with Simple Moving Average (SMA) crossover signals and dynamic position sizing. It uses Binance top-of-book data (BTCUSDT) and Yahoo Finance data (multiple assets) for backtesting and optimization.
## Key Components
- **Strategy**: Combines momentum (buy on strong upward trends, sell on downward trends) with SMA crossover signals (buy when fast SMA crosses above slow SMA, sell when below). Adjusts position sizes using ATR (volatility) and momentum/volume conviction.
- **Implementation**:
  - `MomentumStrategy` class in Backtrader calculates momentum:  
    $$\text{Momentum}_t = \frac{P_t}{P_{t-L}} - 1$$
  - Parameters: lookback (20 days), threshold (0.01), trade_size (0.1).
- **Data**:
  - Binance: 1-hour BTCUSDT kline data.
  - Yahoo Finance: 3-year daily data for NVDA, MTUM, BTC-USD, ICLN, SPMO, AAPL, AMD, TSLA, QQQ, IWM.
- **Backtesting**:
  - Uses Backtrader’s Cerebro engine with cross-validation (`TimeSeriesSplit`).
  - Grid search optimizes parameters (lookback, threshold, SMA periods, ATR period).
- **Results**:
  - Strong performance on NVDA ($14,596, Sharpe: 1.25) and AMD ($11,404, Sharpe: 0.54).
  - Poor performance on ICLN ($9,800, Sharpe: -0.60).
  - Strategy favors high-volume tech stocks; safer trades limit upside but reduce risk.
- **Improvements**:
  - Add take-profit logic and indicators (RSI, MACD, Bollinger Bands).
  - Use Sortino/Calmar ratios, walk-forward optimization, and early stopping in grid search.

## Next Steps
- Enhance strategy with take-profit and additional indicators.
- Improve optimization with Random Search/Bayesian methods.
- Expand visualization and test on more assets/time frames.

## Resources
- Medium: Momentum Strategy Optimization
- Investopedia: Stop-Loss Orders
- Articles on lot sizing and SMA-momentum relationships

# 03. Mean reversion strategy
This Jupyter notebook (`03.Mean reversion strategy.ipynb`) develops a mean reversion trading strategy inspired by physics-based models, specifically the harmonic oscillator. Unlike the momentum strategy, it uses stochastic differential equations (SDEs) to model price movements, focusing on the Ornstein-Uhlenbeck (OU) process for mean-reverting behavior.

## Key Components
- **Strategy**: Models prices as a damped harmonic oscillator, reverting to a mean value, using the OU process:
  $$
  dX_t = \theta (\mu - X_t)dt + \sigma dW_t
  $$
  - \(\mu\): Long-term mean
  - \(\theta\): Speed of mean reversion
  - \(\sigma\): Volatility
  - \(W_t\): Wiener process (Brownian motion)
- **Implementation**:
  - Uses Euler-Maruyama approximation for numerical simulation:
    $$
    X_{t+\Delta t} = X_t + \theta(\mu - X_t)\Delta t + \sigma \sqrt{\Delta t} \cdot Z_t
    $$
  - Forecasts US 10-Year Treasury Yield for June 2025 using a damped oscillator model.
- **Data**:
  - Fetches historical data via `yfinance` and `pandas_datareader`.
  - Train: Up to May 31, 2025; Test: June 1–30, 2025.
- **Results**:
  - Forecast shows strong damping, converging to the mean without oscillations, indicating overdamping.
  - Highlights limitations of the model for capturing oscillatory behavior in financial data.
- **Improvements**:
  - Tune parameters (\(\gamma\), \(\omega\)) for oscillatory behavior.
  - Incorporate additional stochastic models or indicators for robustness.
  - Explore walk-forward optimization and alternative SDEs.

## Next Steps
- Refine model parameters to capture oscillations.
- Test on diverse assets and time frames.
- Integrate additional financial indicators for hybrid strategies.

## Resources
- QuantStart: OU Simulation
- ScienceDirect: Financial Asset Price Forecasting with Damped Oscillator
- Bing: Damped Harmonic Oscillator in Finance, SDEs

# 04. Ai in trading
This Jupyter notebook (`04.AI in trading.ipynb`) explores the application of artificial intelligence, specifically LSTM neural networks, to predict stock price movements (up or down) for trading. It contrasts traditional physics-based approaches (like momentum and mean-reversion strategies from previous notebooks) with data-driven machine learning models, highlighting their flexibility in capturing complex, non-linear market patterns without rigid assumptions.

## Key Components
- **Objective**: Build a classification model to predict if the next day's closing price of AAPL stock will increase (1) or decrease (0).
- **Data**:
  - Source: AAPL stock data from 2010-01-01 to 2023-12-31 via `yfinance`.
  - Features: Close, Volume, High, Low prices; engineered features include:
    - Returns (percentage change)
    - 5-day moving average (MA_5)
    - 10-day volatility (Volatility_10)
    - 14-day RSI
    - 5-day momentum
    - MACD (12-day EMA - 26-day EMA)
    - Bollinger Band width
    - Normalized volume
    - Lagged prices (1, 3, 5 days)
    - 14-day ATR
  - Target: Binary (1 if next day's close > current close, else 0).
- **Preprocessing**:
  - Drops NaN values.
  - Scales features using `StandardScaler`.
  - Creates sequences (10-day lookback) for LSTM input.
  - Uses time-based train-test split (80% train, 20% test) to avoid data leakage.
- **Model**:
  - LSTMClassifier with 2 LSTM layers (64 hidden units, 0.3 dropout) followed by a fully connected layer with ReLU and sigmoid activation.
  - Loss: Binary Cross-Entropy (BCELoss) with class weights to handle imbalance.
  - Optimizer: Adam (learning rate 0.0005, weight decay 1e-5 for L2 regularization).
  - Training: 30 epochs with early stopping (patience=5) based on validation loss.
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1 Score, AUC-ROC.
  - Plots ROC curve and training/validation loss.

## Results
- **Initial Model**:
  - Test Accuracy: ~53.57% (close to random guessing).
  - Indicates weak predictive power, likely due to noisy financial data and insufficient feature signal.
- **Enhanced Model**:
  - Added features (Momentum, MACD, Bollinger Bands, ATR, lagged prices, normalized volume).
  - Early stopping triggered at epoch 6.
  - Test Metrics: Accuracy: ~52.65%, Precision: 52.32%, Recall: 45.92%, F1: 48.91%, AUC: 47.89%.
  - Still no significant improvement over random guessing, highlighting challenges in financial prediction.

## Challenges
- Financial markets are noisy, non-stationary, and subject to regime shifts, making prediction difficult.
- Historical data may lack sufficient predictive signal or contain biases.
- LSTM struggles to capture weak, unstable patterns in financial time series.

## Suggested Improvements
- **Feature Engineering**: Incorporate external data (e.g., sentiment from news or X posts, macroeconomic indicators).
- **Model Enhancements**: Explore ensemble methods, transformers, or reinforcement learning.
- **Data Quality**: Use higher-frequency data (e.g., intraday) or alternative assets.
- **Hyperparameter Tuning**: Optimize sequence length, learning rate, or model architecture.
- **Walk-Forward Validation**: Further refine to simulate real-world trading.

## Resources
- QuantStart: Machine Learning for Trading
- ScienceDirect: Deep Learning in Financial Markets
- PyTorch Documentation: LSTM and Time Series
