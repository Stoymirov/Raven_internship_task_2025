# Raven_internship_task_2025

## Project overview
This project, developed for the Raven Scholarship and Internship Program, implements a trading algorithm designed to maximize profitability in financial markets. Built using Python, PyTorch, and Pandas, the algorithm leverages mathematical modeling, including differential equations and applied statistics, to analyze market data and optimize trading decisions. The solution focuses on generating consistent returns by identifying and capitalizing on market opportunities. This project demonstrates my ability to combine computational techniques and mathematical rigor to create effective, data-driven trading strategies.
# Sctructure
## 01. Premise & research
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

## 02. Momentum Strategy
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
