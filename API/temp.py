import json
import yfinance as yf

def fetch_data(ticker, interval, period="5d"):
    """
    Fetch historical stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol
        interval (str): Data interval
        period (str): Period to fetch data for (e.g., '60d' for 60 days)

    Returns:
        DataFrame: Historical stock data
    """
    print(f"Fetching {interval} data for {ticker} over {period}...")

    stock_data = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        progress=False
    )

    print(f"Retrieved {len(stock_data)} data points")
    return stock_data

df = fetch_data("TCS.NS", "15m", "1d")
# Keep only the 'Price' level names as column names
df.columns = df.columns.get_level_values('Price')

# Reset index to access 'Datetime' as a column
df_reset = df.reset_index()

# Build the desired list of dicts
result = [
    {"Datetime": row["Datetime"], "Close": row["Close"]}
    for _, row in df_reset.iterrows()
]

# Convert to JSON string (optional)
json_result = json.dumps(result, indent=4, default=str)  # default=str to handle datetime

print(json_result)