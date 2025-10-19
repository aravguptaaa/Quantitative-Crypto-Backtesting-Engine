import ccxt
import pandas as pd
import time
import os
from datetime import datetime

# --- Configuration ---
SYMBOL = 'BTC/USDT'  # The trading pair to collect data for
SLEEP_SECONDS = 5    # Time to wait between each data fetch
DATA_DIR = 'data'    # The folder where we'll save the data files

# --- Setup ---
# 1. Initialize the Binance exchange connection using ccxt
# We are using the public API, so no API keys are required.
exchange = ccxt.binance()

# 2. Create the data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created directory: {DATA_DIR}")

print(f"Starting data collection for {SYMBOL}...")
print(f"A new data file will be saved every {SLEEP_SECONDS} seconds in the '{DATA_DIR}' folder.")

# --- Main Loop ---
# This loop will run forever until you stop it manually (Ctrl + C)
while True:
    try:
        # 3. Get the current timestamp for our filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(DATA_DIR, f'market_data_{timestamp}.parquet')

        # 4. Fetch the Level 2 Order Book (L2) and the latest trades
        # fetch_l2_order_book gives more depth than fetch_order_book
        order_book = exchange.fetch_l2_order_book(SYMBOL, limit=100) # Fetch top 100 bids and asks
        trades = exchange.fetch_trades(SYMBOL, limit=100) # Fetch the 100 most recent trades

        # 5. Combine the data into a single dictionary
        # This creates one "snapshot" of the market at this moment
        market_snapshot = {
            'timestamp': [pd.to_datetime('now', utc=True)],
            'order_book': [order_book],
            'trades': [trades]
        }

        # 6. Convert the dictionary to a Pandas DataFrame
        df = pd.DataFrame(market_snapshot)

        # 7. Save the DataFrame to a Parquet file
        # PyArrow is the engine that handles the conversion to the Parquet format.
        df.to_parquet(filename, engine='pyarrow')

        print(f"Successfully saved data to {filename}")

    except ccxt.NetworkError as e:
        print(f"Network error: {e}. Retrying in {SLEEP_SECONDS} seconds...")
    except ccxt.ExchangeError as e:
        print(f"Exchange error: {e}. Retrying in {SLEEP_SECONDS} seconds...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Retrying in {SLEEP_SECONDS} seconds...")

    # 8. Wait for a few seconds before the next loop iteration
    time.sleep(SLEEP_SECONDS)
