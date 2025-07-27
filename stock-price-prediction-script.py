# Cell 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Cell 2: Global variables and configuration
ticker = "AAPL"  # Stock ticker symbol
interval = '15m'  # Data interval ('15m', '30m', '1h', etc.)
lookback_period = 30  # Number of time steps to look back for prediction
model = None
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

# Cell 3: Fetch data function
def fetch_data(ticker, interval, period="60d"):
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
    
    # For 15-minute data, Yahoo Finance provides at most 60 days of historical data
    stock_data = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        progress=False
    )
    
    print(f"Retrieved {len(stock_data)} data points")
    return stock_data

# Cell 4: Prepare features function
def prepare_features(data):
    """
    Prepare features from stock data.
    
    Args:
        data (DataFrame): Stock data
        
    Returns:
        DataFrame: Prepared features
    """
    # Make a copy to avoid modifying the original data
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

# Cell 5: Create sequences function
def create_sequences(data, lookback_period, target_col='Close'):
    """
    Create sequences for LSTM model.
    
    Args:
        data (DataFrame): Prepared data
        lookback_period (int): Number of time steps to look back
        target_col (str): Target column to predict
        
    Returns:
        tuple: (X, y, direction) - Input sequences, target values, direction
    """
    global scaler_x, scaler_y
    
    # Extract features and target
    # For features, exclude the target column and the Direction column
    feature_cols = [col for col in data.columns if col != 'Direction' and col != target_col]
    X_data = data[feature_cols].values
    y_data = data[[target_col]].values
    direction_data = data[['Direction']].values
    
    # Scale the data
    X_scaled = scaler_x.fit_transform(X_data)
    y_scaled = scaler_y.fit_transform(y_data)
    
    X_sequences = []
    y_sequences = []
    direction_sequences = []
    
    # Create sequences
    for i in range(len(X_scaled) - lookback_period):
        X_sequences.append(X_scaled[i:i+lookback_period])
        y_sequences.append(y_scaled[i+lookback_period])
        direction_sequences.append(direction_data[i+lookback_period])
        
    return np.array(X_sequences), np.array(y_sequences), np.array(direction_sequences), feature_cols

# Cell 6: Build model function
def build_model(input_shape):
    """
    Build an LSTM model for stock price prediction.
    
    Args:
        input_shape (tuple): Shape of input data (lookback_period, n_features)
        
    Returns:
        Model: Compiled Keras model
    """
    model = Sequential()
    
    # Bidirectional LSTM layers
    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.2))
    
    model.add(Bidirectional(LSTM(50)))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model using Huber loss for robustness against outliers
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(),
        metrics=['mae']
    )
    
    return model

# Cell 7: Train model function
def train_model(X_train, y_train, input_shape, ticker, epochs=50, batch_size=32, validation_split=0.2):
    """
    Train the LSTM model.
    
    Args:
        X_train (array): Training input sequences
        y_train (array): Training target values
        input_shape (tuple): Shape of input data
        ticker (str): Stock ticker symbol
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        validation_split (float): Fraction of data to use for validation
        
    Returns:
        tuple: (model, history) - Trained model and training history
    """
    global model
    
    # Define callbacks for early stopping and model checkpointing
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(f'{ticker}_model.h5', save_best_only=True, monitor='val_loss')
    ]
    
    # Build model
    model = build_model(input_shape)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# Cell 8: Evaluate model function
def evaluate_model(model, X_test, y_test, direction_test, ticker, interval):
    """
    Evaluate the model performance.
    
    Args:
        model (Model): Trained Keras model
        X_test (array): Test input sequences
        y_test (array): Test target values
        direction_test (array): Actual price direction (1=up, 0=down)
        ticker (str): Stock ticker symbol
        interval (str): Data interval
        
    Returns:
        dict: Evaluation metrics
    """
    global scaler_y
    
    # Make predictions
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform the predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)
    
    # Calculate regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # For directional accuracy, we'll just compare consecutive predictions
    pred_direction = np.where(np.diff(y_pred.flatten()) > 0, 1, 0)
    actual_direction = np.where(np.diff(y_true.flatten()) > 0, 1, 0)
    
    # Make sure they have the same length
    min_len = min(len(pred_direction), len(actual_direction))
    direction_accuracy = accuracy_score(actual_direction[:min_len], pred_direction[:min_len])
    
    # Print evaluation results
    print(f"\nModel Evaluation for {ticker} ({interval} interval):")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Directional Accuracy: {direction_accuracy:.4f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'direction_accuracy': direction_accuracy
    }

# Cell 9: Predict next function (fixed version)
def predict_next(model, latest_data, lookback_period):
    """
    Predict the next price and direction.
    
    Args:
        model (Model): Trained Keras model
        latest_data (DataFrame): Latest data points
        lookback_period (int): Number of time steps to look back
        
    Returns:
        tuple: (predicted_price, predicted_direction)
    """
    global scaler_x, scaler_y
    
    # Prepare features
    prepared_data = prepare_features(latest_data)
    
    # Extract features (excluding Direction and Close columns)
    feature_cols = [col for col in prepared_data.columns if col != 'Direction' and col != 'Close']
    input_data = prepared_data[feature_cols].values[-lookback_period:]
    
    # Scale the input data
    input_scaled = scaler_x.transform(input_data)
    input_sequence = np.array([input_scaled])
    
    # Make prediction
    pred_scaled = model.predict(input_sequence)
    predicted_price = scaler_y.inverse_transform(pred_scaled)[0][0]
    
    # Determine predicted direction - convert to scalar with float()
    last_price = float(latest_data['Close'].iloc[-1])
    predicted_direction = 1 if predicted_price > last_price else 0
    
    return predicted_price, predicted_direction

# Cell 10: Real-time prediction function
def real_time_prediction(model, ticker, interval, lookback_period, interval_minutes=15):
    """
    Set up real-time prediction loop.
    
    Args:
        model (Model): Trained Keras model
        ticker (str): Stock ticker symbol
        interval (str): Data interval
        lookback_period (int): Number of time steps to look back
        interval_minutes (int): Interval in minutes between predictions
    """
    # Get current time
    now = datetime.now()
    
    # Get the latest data
    latest_data = fetch_data(ticker, interval, period="5d")  # Get recent data
    
    # Make sure we have enough data points
    if len(latest_data) >= lookback_period:
        # Make prediction
        predicted_price, predicted_direction = predict_next(model, latest_data, lookback_period)
        
        # Get the last close price
        last_price = float(latest_data['Close'].iloc[-1])
        
        # Print prediction
        direction_text = "UP ↑" if predicted_direction == 1 else "DOWN ↓"
        change_pct = (predicted_price - last_price) / last_price * 100
        
        print(f"\n--- {now.strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"Last {ticker} price: ${last_price:.2f}")
        print(f"Predicted next price: ${predicted_price:.2f} ({change_pct:.2f}%)")
        print(f"Predicted direction: {direction_text}")
        
        # In a real application, you would implement a timing mechanism
        # to make predictions at regular intervals
        print(f"Next prediction would be in {interval_minutes} minutes")
    else:
        print("Not enough data points for prediction. Please wait for more data.")

# Cell 11: Plot training history function
def plot_training_history(history, ticker):
    """
    Plot training history.
    
    Args:
        history: Training history from model.fit()
        ticker (str): Stock ticker symbol
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{ticker} - Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Cell 12: Plot predictions function
def plot_predictions(model, X_test, y_test, ticker):
    """
    Plot predictions vs actual values.
    
    Args:
        model (Model): Trained Keras model
        X_test (array): Test input sequences
        y_test (array): Test target values
        ticker (str): Stock ticker symbol
    """
    global scaler_y
    
    # Make predictions
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(f'{ticker} - Predictions vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot directional accuracy
    plt.figure(figsize=(12, 6))
    
    # Calculate actual direction
    actual_direction = np.where(np.diff(y_true.flatten()) > 0, 1, 0)
    
    # Calculate predicted direction
    predicted_direction = np.where(np.diff(y_pred.flatten()) > 0, 1, 0)
    
    # Make sure they're the same length
    min_len = min(len(actual_direction), len(predicted_direction))
    actual_direction = actual_direction[:min_len]
    predicted_direction = predicted_direction[:min_len]
    
    # Calculate accuracy
    direction_accuracy = accuracy_score(actual_direction, predicted_direction)
    
    plt.plot(actual_direction, label='Actual Direction')
    plt.plot(predicted_direction, label='Predicted Direction', alpha=0.7)
    plt.title(f'{ticker} - Directional Accuracy: {direction_accuracy:.4f}')
    plt.xlabel('Time Steps')
    plt.ylabel('Direction (1=Up, 0=Down)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Cell 13: Save model function
def save_model(model, ticker, interval, filename=None):
    """
    Save the trained model.
    
    Args:
        model (Model): Trained Keras model
        ticker (str): Stock ticker symbol
        interval (str): Data interval
        filename (str): Filename to save the model
    """
    if filename is None:
        filename = f"{ticker}_{interval}_model.h5"
        
    model.save(filename)
    print(f"Model saved to {filename}")

# Cell 14: Load model function
def load_model_from_file(filename):
    """
    Load a trained model.
    
    Args:
        filename (str): Filename of the saved model
        
    Returns:
        Model: Loaded Keras model
    """
    loaded_model = load_model(filename)
    print(f"Model loaded from {filename}")
    return loaded_model

# Cell 15: Complete training and evaluation pipeline
def train_and_evaluate_pipeline(ticker, interval, lookback_period, test_size=0.2):
    """
    Complete pipeline to train and evaluate the model.
    
    Args:
        ticker (str): Stock ticker symbol
        interval (str): Data interval
        lookback_period (int): Number of time steps to look back
        test_size (float): Proportion of data to use for testing
        
    Returns:
        tuple: (model, metrics, history) - Trained model, evaluation metrics, and training history
    """
    try:
        # 1. Fetch data
        data = fetch_data(ticker, interval)
        
        if len(data) < lookback_period * 2:
            print(f"Not enough data points. Got {len(data)}, need at least {lookback_period * 2}")
            return None, None, None
        
        # 2. Prepare features
        prepared_data = prepare_features(data)
        print(f"Prepared data shape: {prepared_data.shape}")
        
        # Print feature columns for debugging
        print(f"Feature columns: {prepared_data.columns.tolist()}")
        
        # 3. Create sequences
        X, y, direction, feature_cols = create_sequences(prepared_data, lookback_period)
        print(f"Sequences created: X shape: {X.shape}, y shape: {y.shape}")
        
        if len(X) < 10:  # Arbitrary small number to ensure we have enough data
            print(f"Not enough sequences. Got {len(X)}, need more for effective training.")
            return None, None, None
        
        # 4. Split into train and test sets
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        direction_train, direction_test = direction[:split_idx], direction[split_idx:]
        
        print(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")
        
        # 5. Train model
        input_shape = (X_train.shape[1], X_train.shape[2])
        trained_model, history = train_model(X_train, y_train, input_shape, ticker)
        
        # 6. Evaluate model
        metrics = evaluate_model(trained_model, X_test, y_test, direction_test, ticker, interval)
        
        # 7. Plot training history
        plot_training_history(history, ticker)
        
        # 8. Plot predictions vs actual
        plot_predictions(trained_model, X_test, y_test, ticker)
        
        # 9. Save the model
        save_model(trained_model, ticker, interval)
        
        return trained_model, metrics, history
        
    except Exception as e:
        print(f"Error in train_and_evaluate_pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Cell 16: Main execution
if __name__ == "__main__":
    # Define parameters
    ticker = "AAPL"  # Stock ticker symbol
    interval = '15m'  # Data interval
    lookback_period = 30  # Number of time steps to look back
    
    # Train and evaluate the model
    model, metrics, history = train_and_evaluate_pipeline(ticker, interval, lookback_period)
    
    # If model training was successful, make real-time predictions
    if model is not None:
        print("\nStarting real-time prediction...")
        real_time_prediction(model, ticker, interval, lookback_period)
    
# Cell 17: Example for multiple stocks
"""
# Define a list of stocks to analyze
tickers = ['AAPL', 'MSFT', 'GOOGL']
results = {}

for ticker in tickers:
    print(f"\n{'='*50}")
    print(f"Training model for {ticker}")
    print(f"{'='*50}")
    
    model, metrics, _ = train_and_evaluate_pipeline(ticker, interval='15m', lookback_period=30)
    
    if metrics:
        results[ticker] = metrics

# Compare results
for ticker, metrics in results.items():
    print(f"\n{ticker}:")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.4f}")
"""
