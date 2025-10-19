import pandas as pd
import joblib

# --- Configuration ---
PROCESSED_DATA_PATH = 'processed_market_data.parquet'
MODEL_PATH = 'model.joblib'
TEST_SIZE = 0.2
INITIAL_CAPITAL = 10000  # Start with $10,000 USD
TRANSACTION_FEE = 0.001  # 0.1% fee per trade (typical for market orders)

# --- Load Data and Model ---
print("Loading data and model...")
# 1. Load the same processed data used for training
df = pd.read_parquet(PROCESSED_DATA_PATH)
model = joblib.load(MODEL_PATH)

# 2. Isolate the test set exactly as we did in the training script
split_index = int(len(df) * (1 - TEST_SIZE))
test_df = df[split_index:].copy() # Use .copy() to avoid warnings

# 3. Separate features (X_test) from the data we'll use for simulation
X_test = test_df.drop(columns=['mid_price', 'future_mid_price', 'target'])

print("Data and model loaded successfully.")

# --- Backtesting Simulation ---
print("Starting backtest simulation...")

# Account variables
usd_balance = INITIAL_CAPITAL
btc_balance = 0.0
current_position = 0  # 0 = flat, 1 = long (holding BTC), -1 = short (not implemented)

# Lists to track our portfolio value over time
timestamps = []
portfolio_values = []

# Loop through each row (each moment in time) in our test data
for i in range(len(test_df)):
    # 4. Get the features for the current timestamp
    current_features = X_test.iloc[[i]] # Select row 'i' as a DataFrame
    current_price = test_df.iloc[i]['mid_price']

    # 5. Use the model to make a prediction
    # model.predict returns the mapped label (0, 1, 2)
    predicted_mapped_label = model.predict(current_features)[0]

    # 6. CRITICAL: Convert the prediction back to our trading signal {-1, 0, 1}
    # 0 -> -1 (DOWN), 1 -> 0 (FLAT), 2 -> 1 (UP)
    prediction = predicted_mapped_label - 1

    # --- Trading Logic ---
    # If model predicts UP (1) and we are not already holding BTC
    if prediction == 1 and current_position == 0:
        # Simulate BUY
        btc_to_buy = (usd_balance / current_price) * (1 - TRANSACTION_FEE)
        btc_balance += btc_to_buy
        usd_balance = 0
        current_position = 1
        print(f"{test_df.index[i]}: Model predicts UP. Buying {btc_to_buy:.6f} BTC at ${current_price:.2f}")

    # If model predicts DOWN (-1) and we are currently holding BTC
    elif prediction == -1 and current_position == 1:
        # Simulate SELL
        usd_to_get = (btc_balance * current_price) * (1 - TRANSACTION_FEE)
        usd_balance += usd_to_get
        btc_balance = 0
        current_position = 0
        print(f"{test_df.index[i]}: Model predicts DOWN. Selling BTC for ${usd_to_get:.2f}")

    # --- Record Portfolio Value ---
    # Calculate the total value of our assets (USD + BTC valued at current price)
    current_portfolio_value = usd_balance + (btc_balance * current_price)
    timestamps.append(test_df.index[i])
    portfolio_values.append(current_portfolio_value)

# --- Final Results ---
print("\nBacktest simulation finished.")

# Calculate final portfolio value at the end of the test period
final_portfolio_value = portfolio_values[-1]
total_return_pct = ((final_portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

print(f"\n--- Backtest Results ---")
print(f"Initial Portfolio Value: ${INITIAL_CAPITAL:,.2f}")
print(f"Final Portfolio Value:   ${final_portfolio_value:,.2f}")
print(f"Total Return:            {total_return_pct:.2f}%")

# --- Save Results for Visualization ---
# We save the results so we can plot them in our Jupyter notebook
results_df = pd.DataFrame({
    'timestamp': timestamps,
    'portfolio_value': portfolio_values
})
results_df.to_csv('backtest_results.csv', index=False)
print("\nBacktest results saved to backtest_results.csv")
