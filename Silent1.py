# Silent1_Final_WebSocket_Version.py - 
# Version: 5.0.0 (Refactored with separate SDK)

import logging
import os
import asyncio
import datetime
import json
import threading
import joblib
import random
import numpy as np
import pandas as pd
import pandas_ta as ta
# websockets is now used by the SDK, not directly here.
from dotenv import load_dotenv
from flask import Flask, jsonify
from logging.handlers import RotatingFileHandler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Import the new SDK ---
from coinex_sdk import CoinEx

# --- Global Status & State Variables ---
system_status = {
    "api_authenticated": False,
    "websocket_connected": False,
    "trading_active": False,
    "server_running": True,
    "last_error": None,
    "connection_errors": [],
    "market_activity": {}
}
active_positions = {}
# Thread-safe lock for accessing shared data like DataFrames
data_lock = asyncio.Lock()

# === Load Environment Variables ===
load_dotenv()
API_KEY = os.getenv('COINEX_API_KEY')
API_SECRET = os.getenv('COINEX_API_SECRET')

if not API_KEY or not API_SECRET:
    raise ValueError("CRITICAL: API_KEY or API_SECRET not found in .env file.")

# === CoinEx SDK Integration ===
# The SDK classes are now in coinex_sdk.py
# We just need to initialize the main client here
coinex_client = CoinEx(API_KEY, API_SECRET)

# === Enhanced Logging Setup ===
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_handler = RotatingFileHandler('bot_diagnostics.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
log_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[log_handler, console_handler])

# === Connection Diagnostics ===
async def perform_connection_tests():
    system_status["connection_errors"] = []
    logging.info("Testing API Connection...")
    ping_res = await asyncio.to_thread(coinex_client.market.ping)
    if ping_res.get('code') != 0:
        system_status["connection_errors"].append(f"API Connection Failed: {ping_res.get('message', 'Unknown Error')}")

    if not system_status["connection_errors"]:
        logging.info("Testing API Authentication...")
        auth_res = await asyncio.to_thread(coinex_client.account.get_account_info)
        if auth_res.get('code') != 0:
            system_status["connection_errors"].append(f"API Authentication Failed: {auth_res.get('message')}")
        else:
            system_status["api_authenticated"] = True
    
    return {"success": not system_status["connection_errors"], "errors": system_status["connection_errors"]}

# === Order Functions ===
async def fetch_balance():
    balance_data = await asyncio.to_thread(coinex_client.account.get_account_info)
    if balance_data.get('code') == 0 and 'data' in balance_data:
        usdt_balance = next((float(asset.get('available', 0)) for asset in balance_data['data']['spot'] if asset['ccy'] == 'USDT'), 0.0)
        return {'USDT': {'free': usdt_balance}}
    raise ConnectionError(f"Failed to fetch balance: {balance_data.get('message')}")

async def execute_real_trade(symbol, side, amount):
    params = {"market": symbol.replace('/', ''), "side": side.lower(), "type": "market"}
    if side.lower() == 'buy':
        params["amount"] = str(round(amount, 8)) # amount is in quote currency (USDT)
    else: # sell
        params["quantity"] = str(round(amount, 8)) # quantity is in base currency (e.g., BTC)
        
    order_result = await asyncio.to_thread(coinex_client.request, 'POST', '/spot/order', params, need_sign=True)
    if order_result.get('code') == 0:
        logging.info(f"Successfully placed {side} order for {symbol}: {order_result['data']}")
        return order_result['data']
    logging.error(f"Failed to place {side} order for {symbol}: {order_result}")
    return None

# === TA & ML Functions ===
def calculate_indicators(df):
    if df.empty: return df
    try:
        df['SMA'] = ta.sma(df['close'], length=20)
        df['RSI'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df['MACD'] = macd[f'MACD_{12}_{26}_{9}']
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None and not bbands.empty:
            df['upper_band'] = bbands[f'BBU_20_2.0']
            df['lower_band'] = bbands[f'BBL_20_2.0']
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None and not stoch.empty:
            df['stoch_k'] = stoch[f'STOCHk_14_3_3']
        return df.fillna(method='ffill')
    except Exception as e:
        logging.warning(f"Could not calculate indicators: {e}")
        return df

def generate_ml_signals(df):
    model_path = 'trading_model.pkl'
    features = ['SMA', 'RSI', 'MACD', 'stoch_k', 'ATR', 'upper_band', 'lower_band']
    
    if not all(f in df.columns for f in features) or df[features].isnull().values.any():
        return df # Not ready for prediction yet

    df_clean = df.dropna(subset=features).copy()
    if df_clean.empty: return df

    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            # Simple logic to create a placeholder model if one doesn't exist
            logging.info("Model not found. Creating a placeholder model.")
            df_clean['target'] = (df_clean['close'].shift(-5) > df_clean['close']).astype(int)
            df_clean = df_clean.dropna()
            X, y = df_clean[features], df_clean['target']
            if len(X) < 50: return df # Not enough data to train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(X_train, y_train)
            joblib.dump(model, model_path)
            logging.info(f"New model created and saved with accuracy: {accuracy_score(y_test, model.predict(X_test))}")

        predictions = model.predict(df_clean[features])
        df.loc[df_clean.index, 'ml_signal'] = predictions
        return df

    except Exception as e:
        logging.error(f"Error in ML signal generation: {e}")
        return df

def is_market_weak(df):
    if len(df) < 15: return True
    volatility = (df['high'] - df['low']).mean() / df['close'].mean()
    volume_change = df['volume'].pct_change().abs().mean()
    return volatility < 0.003 and volume_change < 0.1

# === Real-Time WebSocket Handler (Refactored) ===
async def websocket_data_handler(symbols_to_watch, dfs):
    ALL_SYMBOLS_FLAT = {s.replace('/', ''): s for s in ALL_SYMBOLS}
    
    async def on_message_callback(message):
        try:
            data = json.loads(message)
            if data.get('method') == 'state.update' and 'params' in data:
                update_data = data['params'][0]
                market_flat = update_data.get('market')
                
                if market_flat in ALL_SYMBOLS_FLAT:
                    symbol_key = ALL_SYMBOLS_FLAT[market_flat]
                    last_price = float(update_data.get('last', 0))
                    if last_price == 0: return

                    timestamp = datetime.datetime.now(datetime.timezone.utc)
                    new_row = pd.DataFrame([{
                        'timestamp': timestamp, 'open': last_price, 'high': last_price,
                        'low': last_price, 'close': last_price,
                        'volume': float(update_data.get('deal', 0))
                    }], index=[timestamp])
                    
                    async with data_lock:
                        df_current = dfs[symbol_key]
                        dfs[symbol_key] = pd.concat([df_current, new_row]).last('24h') # Keep last 24h of data
                        
                        # Resample to 1-minute candles for consistent analysis
                        df_resampled = dfs[symbol_key]['close'].resample('1T').ohlc()
                        df_resampled['volume'] = dfs[symbol_key]['volume'].resample('1T').sum()
                        
                        df_with_indicators = calculate_indicators(df_resampled)
                        dfs[symbol_key] = generate_ml_signals(df_with_indicators)
        except Exception as e:
            logging.error(f"Error processing WebSocket message: {e} | Message: {message}")

    if await coinex_client.websocket.connect(on_message_callback=on_message_callback):
        system_status["websocket_connected"] = True
        await coinex_client.websocket.subscribe_to_tickers(symbols_to_watch)
        while system_status["server_running"]:
            if not coinex_client.websocket.connected:
                system_status["websocket_connected"] = False
                logging.warning("Detected WebSocket disconnect. SDK will attempt to reconnect.")
            await asyncio.sleep(30)
    else:
        system_status["websocket_connected"] = False
        logging.critical("Could not establish WebSocket connection after multiple retries.")


# === Helper Function for Dynamic Symbols ===
ALL_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT']
def get_daily_symbols():
    daily_count = random.randint(3, 5)
    return random.sample(ALL_SYMBOLS, daily_count)

# === Main Trading Engine ===
async def run_trading_engine():
    try:
        logging.info("--- Starting Trading Engine ---")
        await perform_connection_tests()
        if system_status["connection_errors"]:
            raise ConnectionError(f"Initial diagnostics failed: {system_status['connection_errors']}")
        
        symbols_for_today = get_daily_symbols()
        logging.info(f"Selected symbols for today: {symbols_for_today}")
        
        balance_info = await fetch_balance()
        total_balance = balance_info.get('USDT', {}).get('free', 0)
        if total_balance < 20:
            raise ValueError(f"Insufficient balance ({total_balance} USDT) to start.")
        
        usdt_per_trade = max(10, (total_balance / len(symbols_for_today)) * 0.2) # Min $10, 20% of slice
        logging.info(f"Allocating ~${usdt_per_trade:.2f} per trade.")

        # Initialize DataFrames
        dfs = {s: pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']).set_index('timestamp') for s in ALL_SYMBOLS}

        # Start the single WebSocket handler for all symbols
        asyncio.create_task(websocket_data_handler(symbols_for_today, dfs))
        
        await asyncio.sleep(10) # Wait for some data to arrive
        system_status["trading_active"] = True
        logging.info("--- Trading Engine is now LIVE using WebSocket ---")

        while system_status["server_running"]:
            async with data_lock:
                for symbol in symbols_for_today:
                    if not dfs[symbol].empty:
                        await trade_and_manage_position(dfs[symbol].copy(), symbol, usdt_per_trade)
            await asyncio.sleep(random.uniform(5, 15))

    except Exception as e:
        logging.critical(f"Trading engine crashed: {e}", exc_info=True)
    finally:
        system_status["trading_active"] = False
        logging.info("--- Closing all open positions on shutdown ---")
        for symbol, position in list(active_positions.items()):
            await execute_real_trade(symbol, "sell", position['quantity'])
        logging.info("--- Trading Engine Shutdown Complete ---")

async def trade_and_manage_position(df, symbol, usdt_per_trade):
    global active_positions
    if df.empty or 'ml_signal' not in df.columns: return

    last_row = df.iloc[-1]
    last_signal = last_row['ml_signal']
    current_price = last_row['close']

    if is_market_weak(df.tail(15)):
        system_status["market_activity"][symbol] = "Weak"
        return
    else:
        system_status["market_activity"][symbol] = "Active"

    # Position Management
    if symbol in active_positions:
        pos = active_positions[symbol]
        pnl = (current_price - pos['entry_price']) / pos['entry_price']
        
        # Take profit or stop loss
        if last_signal == 0 or pnl <= -0.02: # Sell signal or 2% stop loss
            logging.info(f"SELL signal for {symbol}. PNL: {pnl:.2%}. Closing position.")
            if await execute_real_trade(symbol, "sell", pos['quantity']):
                del active_positions[symbol]
        # Trailing stop
        else:
            new_stop = current_price * 0.98 # 2% trailing stop
            if new_stop > pos['trailing_stop_price']:
                pos['trailing_stop_price'] = new_stop
                logging.debug(f"Updated trailing stop for {symbol} to {new_stop:.4f}")

    # Entry Logic
    elif last_signal == 1:
        logging.info(f"BUY signal for {symbol} at {current_price:.4f}. Executing trade.")
        order_result = await execute_real_trade(symbol, "buy", usdt_per_trade)
        if order_result and 'filled_quantity' in order_result:
            filled_quantity = float(order_result['filled_quantity'])
            entry_price = float(order_result['avg_price'])
            active_positions[symbol] = {
                'quantity': filled_quantity,
                'entry_price': entry_price,
                'trailing_stop_price': entry_price * 0.98
            }
            logging.info(f"New position opened for {symbol}: {filled_quantity} @ ${entry_price}")

# === Web Interface & Startup ===
app = Flask(__name__)
@app.route('/')
def home():
    return "<h1>Trading Bot Status</h1><p><a href='/status'>View JSON Status</a></p>"
@app.route('/status')
def status():
    system_status["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    return jsonify({"status": system_status, "positions": active_positions})
def run_server():
    port = int(os.getenv("PORT", 5000))
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=port, threaded=True)

async def main():
    threading.Thread(target=run_server, daemon=True).start()
    try:
        await run_trading_engine()
    except KeyboardInterrupt:
        logging.info("Shutdown signal received.")
    finally:
        system_status["server_running"] = False
        await coinex_client.websocket.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Program exiting.")
