# Quantitative Crypto Backtesting Engine

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An institutional-grade backtesting framework for developing and evaluating high-frequency cryptocurrency trading strategies. This engine interrogates market microstructure data to find predictive signals (alpha), rigorously tests strategies against historical data, and provides a clear, data-driven verdict on their viability before any capital is deployed.

## Project Thesis

The core hypothesis of this research was to determine if specific market microstructure features—namely Order Book Imbalance (OBI), Bid-Ask Spread, and Trade Flow—contain statistically significant predictive power for short-term price movements in the BTC/USDT pair. This framework was built to provide a definitive answer to that question.

## System Architecture

The engine is designed as a modular pipeline, mirroring the workflow used in professional quantitative trading firms.

1. **Data Ingestion:** A robust data collector connects to the Binance API via WebSockets or REST to capture Level 2 order book snapshots and tick-level trade data, saving it in the highly efficient Parquet format.
2. **Feature Engineering:** Raw market data is processed into a feature matrix. This is the "alpha generation" phase, where raw data is transformed into potential predictive signals.
3. **Predictive Modeling:** An XGBoost model is trained on the engineered features to classify the next price movement into one of three states: UP, DOWN, or FLAT. The system is designed to easily accommodate other models (e.g., LSTMs, LightGBM).
4. **Backtesting Simulation:** A vectorized, event-driven backtester simulates the strategy's performance on out-of-sample data. It crucially incorporates realistic transaction costs to avoid over-optimistic results.
5. **Performance Analytics & Visualization:** The final stage computes key quantitative metrics (Sharpe Ratio, Max Drawdown, PnL) and visualizes the strategy's equity curve against a benchmark, all presented in an interactive Streamlit dashboard.

## Tech Stack

* **Core Data Science:** Pandas, NumPy, Scikit-learn
* **Modeling:** XGBoost
* **Data Exchange & Storage:** CCXT, PyArrow
* **Visualization:** Matplotlib, Seaborn, Streamlit
* **Development:** Python 3.11, JupyterLab, venv

## Results & Performance Analysis

The primary goal of a backtesting engine is to provide an unbiased evaluation of a strategy's viability. In this case, the framework successfully identified and quantified the strategy's lack of alpha, thereby achieving its main objective: **preventing capital deployment into an unprofitable model.**

<img width="2878" height="940" alt="image" src="https://github.com/user-attachments/assets/5d13a962-37e4-418a-8c05-5fceeae139db" />
<img width="2666" height="1512" alt="image" src="https://github.com/user-attachments/assets/ce429d41-c1b5-4c2d-bc48-8e1334783c1c" />

| Metric                   | Value              | Interpretation                                                  |
| ------------------------ | ------------------ | --------------------------------------------------------------- |
| **Strategy PnL**   | **$-809.25** | The model's signals led to a significant net loss.              |
| **Buy & Hold PnL** | −21.51**−**21.51 | The underlying market was slightly down during the test period. |
| **Max Drawdown**   | -8.13%             | The largest peak-to-trough decline experienced by the strategy. |
| **Sharpe Ratio**   | **-12.98**   | Indicates a highly unfavorable risk-adjusted return.            |

The results unequivocally show that the initial feature set did not provide a profitable trading edge after accounting for transaction costs. The framework performed its job perfectly by invalidating the initial hypothesis, saving significant potential losses.

## Setup and Execution

### Prerequisites

* Python 3.11+
* [Homebrew](https://brew.sh/) (for macOS users)

### Installation & Execution Workflow

1. **Clone the Repository:**

   <pre class="p-0 m-0 rounded-xl"><div class="rt-Box relative"><div class="rt-Flex rt-r-fd-column rt-r-py-1 rt-r-w absolute top-2 z-10 px-[14px]"><div class="rt-Flex rt-r-fd-row rt-r-ai-center rt-r-jc-space-between"><span data-accent-color="gray" class="rt-Text">bash</span></div></div><pre><code class="language-bash"><span class="token">git</span><span> clone https://github.com/your-username/your-repo-name.git
   </span><span></span><span class="token">cd</span><span> your-repo-name</span></code></pre></div></pre>
2. **Set Up Virtual Environment:**

   <pre class="p-0 m-0 rounded-xl"><div class="rt-Box relative"><div class="rt-Flex rt-r-fd-column rt-r-py-1 rt-r-w absolute top-2 z-10 px-[14px]"><div class="rt-Flex rt-r-fd-row rt-r-ai-center rt-r-jc-space-between"><span data-accent-color="gray" class="rt-Text">bash</span></div></div><pre><code class="language-bash"><span>python3 -m venv quant-env
   </span><span></span><span class="token">source</span><span> quant-env/bin/activate</span></code></pre></div></pre>
3. **Install Dependencies:**

   <pre class="p-0 m-0 rounded-xl"><div class="rt-Box relative"><div class="rt-Flex rt-r-fd-column rt-r-py-1 rt-r-w absolute top-2 z-10 px-[14px]"><div class="rt-Flex rt-r-fd-row rt-r-ai-center rt-r-jc-space-between"><span data-accent-color="gray" class="rt-Text">bash</span></div></div><pre><code class="language-bash"><span>pip </span><span class="token">install</span><span> -r requirements.txt</span></code></pre></div></pre>
4. **Step 1: Collect Market Data**
   Run the collector for a sufficient period (e.g., several hours).

   <pre class="p-0 m-0 rounded-xl"><div class="rt-Box relative"><div class="rt-Flex rt-r-fd-column rt-r-py-1 rt-r-w absolute top-2 z-10 px-[14px]"><div class="rt-Flex rt-r-fd-row rt-r-ai-center rt-r-jc-space-between"><span data-accent-color="gray" class="rt-Text">bash</span></div></div><pre><code class="language-bash"><span>python data_collector.py</span></code></pre></div></pre>

   Stop with Ctrl + C when complete.
5. **Step 2: Engineer Features & Train Model**
   Open and run all cells in the Jupyter Notebook to process the raw data and train the model.

   <pre class="p-0 m-0 rounded-xl"><div class="rt-Box relative"><div class="rt-Flex rt-r-fd-column rt-r-py-1 rt-r-w absolute top-2 z-10 px-[14px]"><div class="rt-Flex rt-r-fd-row rt-r-ai-center rt-r-jc-space-between"><span data-accent-color="gray" class="rt-Text">bash</span></div></div><pre><code class="language-bash"><span>jupyter lab research.ipynb</span></code></pre></div></pre>
6. **Step 3: Run the Backtest**
   Execute the backtester to simulate the strategy on the test data.

   <pre class="p-0 m-0 rounded-xl"><div class="rt-Box relative"><div class="rt-Flex rt-r-fd-column rt-r-py-1 rt-r-w absolute top-2 z-10 px-[14px]"><div class="rt-Flex rt-r-fd-row rt-r-ai-center rt-r-jc-space-between"><span data-accent-color="gray" class="rt-Text">bash</span></div></div><pre><code class="language-bash"><span>python backtester.py</span></code></pre></div></pre>
7. **Step 4: Launch the Dashboard**
   Start the Streamlit server to view the final performance report.

   <pre class="p-0 m-0 rounded-xl"><div class="rt-Box relative"><div class="rt-Flex rt-r-fd-column rt-r-py-1 rt-r-w absolute top-2 z-10 px-[14px]"><div class="rt-Flex rt-r-fd-row rt-r-ai-center rt-r-jc-space-between"><span data-accent-color="gray" class="rt-Text">bash</span></div></div><pre><code class="language-bash"><span>streamlit run dashboard.py</span></code></pre></div></pre>

## Future Work & Scalability

This engine serves as a powerful baseline for further quantitative research. Potential enhancements include:

* **Alpha Diversification:** Engineering more sophisticated features (e.g., volatility clusters, time-weighted averages, order flow toxicity metrics).
* **Advanced Modeling:** Implementing more complex models like LSTMs or Transformers to better capture time-series dependencies.
* **Hyperparameter Optimization:** Using techniques like GridSearchCV or Bayesian Optimization to fine-tune the model for peak performance.
* **Robustness Testing:** Integrating walk-forward validation and Monte Carlo analysis to ensure the strategy is not overfit.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://app.outlier.ai/playground/LICENSE.md) file for details.
