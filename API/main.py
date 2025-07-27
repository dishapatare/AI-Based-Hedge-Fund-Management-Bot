from flask import Flask, request, jsonify
from datetime import datetime
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
import string
import json
import threading

import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database file
db = SQLAlchemy(app)

# Load the scaler models
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

LOOKBACK_PERIOD = 30
INTERVAL = '15m'
SYMBOLS=["AAPL", "GOOGL", "TCS.NS"]
ALLOWED_TOKENS=[]
tokens_lock = threading.Lock()


# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

# # Initialize the database
# with app.app_context():
#     db.create_all()

#     # Add a sample user for testing (you can remove this in production)
#     if not User.query.filter_by(username='admin').first():
#         hashed_password = generate_password_hash('password')
#         sample_user = User(username='admin', password_hash=hashed_password)
#         db.session.add(sample_user)
#         db.session.commit()


def generate_token(length=16):
    # Define the set of characters that we want to include in our token
    characters = string.ascii_letters + string.digits
    # Generate a secure random token
    token = ''.join(secrets.choice(characters) for _ in range(length))
    return token


def prepare_features(data):
    """
    Prepare features from stock data.

    Args:
        data (DataFrame): Stock data

    Returns:
        DataFrame: Prepared features
    """

    df = data.copy()

    # Make sure all necessary columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: {col} column not found in data. Available columns: {df.columns.tolist()}")
            if col == 'Volume' and 'vol' in df.columns:
                df['Volume'] = df['vol']
            elif col == 'Close' and 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']

    # Create a dummy Close column if it doesn't exist (unlikely but just in case)
    if 'Close' not in df.columns:
        print("Warning: Creating dummy Close column from average of Open, High, Low")
        cols_to_avg = [c for c in ['Open', 'High', 'Low'] if c in df.columns]
        if cols_to_avg:
            df['Close'] = df[cols_to_avg].mean(axis=1)
        else:
            raise ValueError("Cannot create features: no price data available")

    # Technical indicators
    # 1. Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()

    # 2. Price changes
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_1'] = df['Close'].pct_change(periods=1)
    df['Price_Change_5'] = df['Close'].pct_change(periods=5)

    # 3. Volume features - only if Volume exists
    if 'Volume' in df.columns:
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    else:
        # Create dummy volume features
        df['Volume'] = 0
        df['Volume_Change'] = 0
        df['Volume_MA5'] = 0

    # 4. Volatility (using standard deviation of returns)
    df['Volatility'] = df['Price_Change'].rolling(window=10).std()

    # 5. OHLC features - create what we can based on available columns
    if all(col in df.columns for col in ['High', 'Low']):
        df['HL_Diff'] = df['High'] - df['Low']
        if 'Low' in df.columns:
            df['HL_PCT'] = (df['High'] - df['Low']) / df['Low']

    if all(col in df.columns for col in ['Open', 'Close']):
        df['OC_Diff'] = df['Open'] - df['Close']

    # Add copy of Close to features to help with evaluation
    df['Close_Feature'] = df['Close']

    # 6. Time-based features (hour of day might be important for intraday data)
    if hasattr(df.index, 'hour'):
        df['Hour'] = df.index.hour
        df['Day_of_Week'] = df.index.dayofweek

        # One-hot encode hour (important for 15-min data)
        for hour in range(24):
            df[f'Hour_{hour}'] = (df['Hour'] == hour).astype(int)

    # Drop NaN values created by rolling windows
    df.dropna(inplace=True)

    # Direction label (1 if price went up, 0 if down or unchanged)
    df['Direction'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    return df


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


# Utility function to predict
def predict_next_price(ticker):
    try:
        model_path = f"{ticker}_{INTERVAL}_model.h5"
        model = load_model(model_path)

        latest_data = fetch_data(ticker, INTERVAL)

        if len(latest_data) < LOOKBACK_PERIOD:
            return {"error": "Not enough data to make a prediction."}

        prepared_data = prepare_features(latest_data)
        feature_cols = [col for col in prepared_data.columns if col != 'Direction' and col != 'Close']

        input_data = prepared_data[feature_cols].values[-LOOKBACK_PERIOD:]

        # Scale input data
        input_scaled = scaler_x.fit_transform(input_data)
        input_sequence = np.array([input_scaled])

        # Predict
        pred_scaled = model.predict(input_sequence)
        scaler_y.fit(prepared_data[['Close']].values)  # Fit scaler_y to current close prices
        predicted_price = scaler_y.inverse_transform(pred_scaled)[0][0]

        last_price = float(latest_data['Close'].iloc[-1])
        direction = "UP" if predicted_price > last_price else "DOWN"
        
        now = datetime.now()
        print(f"\n--- {now.strftime('%Y-%m-%d %H:%M:%S')} ---")

        return {
            "ticker": ticker,
            "datetime": now.strftime('%Y-%m-%d %H:%M:%S'),
            "last_price": str(round(last_price, 2)),
            "predicted_price": str(round(predicted_price, 2)),
            "predicted_direction": str(direction)
        }

    except Exception as e:
        return {"error": str(e)}


@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker')
    token = request.args.get('token')

    with tokens_lock:
        if token not in ALLOWED_TOKENS:
            return {"error": "Invalid token"}, 401

    if not ticker:
        return jsonify({"error": "Please provide a ticker symbol using the 'ticker' query parameter."}), 400
    
    if ticker.upper() not in SYMBOLS:
        return jsonify({"error": f"Ticker {ticker} is not supported. Supported tickers: {SYMBOLS}"}), 400

    result = predict_next_price(ticker.upper())

    return jsonify(result)


@app.route('/get_live_data', methods=['GET'])
def get_live_data():
    token = request.args.get('token')
    with tokens_lock:
        if token not in ALLOWED_TOKENS:
            return {"error": "Invalid token"}, 401

    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Please provide a ticker symbol using the 'ticker' query parameter."}), 400
    
    if ticker.upper() not in SYMBOLS:
        return jsonify({"error": f"Ticker {ticker} is not supported. Supported tickers: {SYMBOLS}"}), 400

    df = fetch_data(ticker.upper(), INTERVAL, period="1d")
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
    json_result = json.dumps(result, indent=4, default=str)
    return json_result, 200


@app.route('/list_symbols', methods=['GET'])
def list_symbols():
    return jsonify(list(SYMBOLS))


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"})


@app.route('/predict_all', methods=['GET'])
def predict_all():
    token = request.args.get('token')
    with tokens_lock:
        if token not in ALLOWED_TOKENS:
            return {"error": "Invalid token"}, 401
    
    result = {}
    for ticker in SYMBOLS:
        result[ticker] = predict_next_price(ticker)
    return jsonify(result)


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    # Query the database for the user
    user = User.query.filter_by(username=username).first()

    if user and check_password_hash(user.password_hash, password):
        random_token = generate_token()
        with tokens_lock:
            ALLOWED_TOKENS.append(random_token)
        return jsonify({'token': random_token}), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/logout', methods=['POST'])
def logout():
    token = request.json.get('token')
    with tokens_lock:
        if token in ALLOWED_TOKENS:
            ALLOWED_TOKENS.remove(token)
            return jsonify({'message': 'Logged out'}), 200
        else:
            return jsonify({'error': 'Invalid token'}), 401


@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    user = User.query.filter_by(username=username).first()
    if user:
        return jsonify({'error': 'Username already exists'}), 400
    user = User(username=username, password_hash=generate_password_hash(password))
    db.session.add(user)
    db.session.commit()
    return jsonify({'message': 'User created'}), 200



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
