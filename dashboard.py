import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Page Configuration ---
st.set_page_config(
    page_title="Crypto Trading Bot Performance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Functions from Step 7 (to make the script self-contained) ---

def load_data(results_path, processed_data_path, test_size):
    """Loads all necessary data files."""
    results_df = pd.read_csv(results_path, parse_dates=['timestamp'])
    results_df.set_index('timestamp', inplace=True)
    
    df = pd.read_parquet(processed_data_path)
    split_index = int(len(df) * (1 - test_size))
    test_set_prices = df[split_index:].copy()
    
    return results_df, test_set_prices

def calculate_metrics(results_df, test_set_prices, risk_free_rate=0.02):
    """Calculates key performance metrics."""
    initial_capital = results_df['portfolio_value'].iloc[0]
    
    # Buy & Hold calculation
    initial_price = test_set_prices['mid_price'].iloc[0]
    buy_and_hold_values = (test_set_prices['mid_price'] / initial_price) * initial_capital

    # Strategy returns
    strategy_returns = results_df['portfolio_value'].pct_change().dropna()
    
    # Sharpe Ratio
    daily_risk_free_rate = (1 + risk_free_rate)**(1/365) - 1
    excess_returns = strategy_returns - daily_risk_free_rate
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0

    # Max Drawdown
    cumulative_returns = (1 + strategy_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    metrics = {
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "final_pnl_strategy": results_df['portfolio_value'].iloc[-1] - initial_capital,
        "final_pnl_buy_hold": buy_and_hold_values.iloc[-1] - initial_capital,
        "final_value_strategy": results_df['portfolio_value'].iloc[-1],
        "final_value_buy_hold": buy_and_hold_values.iloc[-1]
    }
    return metrics, buy_and_hold_values

def create_equity_curve_plot(results_df, buy_and_hold_values):
    """Creates the equity curve matplotlib plot."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(results_df.index, results_df['portfolio_value'], label='XGBoost Strategy', color='royalblue', linewidth=2)
    ax.plot(buy_and_hold_values.index, buy_and_hold_values, label='Buy and Hold BTC', color='gray', linestyle='--', linewidth=2)

    ax.set_title('Strategy Performance vs. Buy and Hold', fontsize=16)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True)
    
    formatter = plt.FuncFormatter(lambda x, pos: f'${int(x):,}')
    ax.yaxis.set_major_formatter(formatter)
    
    fig.autofmt_xdate() # Auto-formats the x-axis dates for better readability
    
    return fig


# --- Main Application ---

st.title("📈 Predictive Crypto Trading Bot")
st.markdown("""
This dashboard showcases the backtested performance of a machine learning-based trading strategy for the **BTC/USDT** pair.
The strategy uses an **XGBoost model** to predict short-term price movements based on live order book and trade data.
""")

# --- Load Data ---
try:
    results_df, test_set_prices = load_data('backtest_results.csv', 'processed_market_data.parquet', test_size=0.2)

    # --- Calculate and Display Metrics ---
    st.header("Key Performance Indicators")
    metrics, buy_and_hold_values = calculate_metrics(results_df, test_set_prices)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    col2.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    col3.metric("Strategy PnL", f"${metrics['final_pnl_strategy']:,.2f}")
    col4.metric("Buy & Hold PnL", f"${metrics['final_pnl_buy_hold']:,.2f}")

    # --- Display Equity Curve ---
    st.header("Equity Curve")
    st.write("This chart shows the growth of an initial $10,000 investment over the test period.")
    
    equity_curve_fig = create_equity_curve_plot(results_df, buy_and_hold_values)
    st.pyplot(equity_curve_fig)

except FileNotFoundError:
    st.error("Error: Make sure 'backtest_results.csv' and 'processed_market_data.parquet' are in the same directory as this script.")
except Exception as e:
    st.error(f"An error occurred: {e}")
