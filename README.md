# Raven_internship_task_2025

## Project overview
This project, developed for the Raven Scholarship and Internship Program, implements a trading algorithm designed to maximize profitability in financial markets. Built using Python, PyTorch, and Pandas, the algorithm leverages mathematical modeling, including differential equations and applied statistics, to analyze market data and optimize trading decisions. The solution focuses on generating consistent returns by identifying and capitalizing on market opportunities. This project demonstrates my ability to combine computational techniques and mathematical rigor to create effective, data-driven trading strategies.
# Sctructure
## 01. Premise & research
This project, outlined in the Jupyter notebook `01.Premise & research.ipynb`, aims to develop and evaluate trading strategies to generate consistent profits using Binance top-of-book data (best bid and ask prices). The notebook serves as the foundational research and planning phase, where we define the problem, explore trading strategies, and establish a rigorous methodology based on the scientific method.

## What This Notebook Does
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

This project aims to build a robust, data-driven trading strategy optimized for various market conditions.
