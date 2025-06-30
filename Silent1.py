# Silent1_final.py
# Version: 3.0.0 (Stable - Based on Official SDK Practices)

import requests
import json
import time
import hmac
import hashlib
import pandas as pd
import asyncio
import websockets
import logging
import ssl
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import os
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, jsonify
import threading
import datetime

# --- Global Status & State Variables ---
system_status = {
    "api_authenticated": False,
    "websocket_connected": False,
    "model_loaded": False,
    "trading_active": False,
    "server_running": True,
    "last_error": None,
    "connection_errors": [],
    "timestamp": None
}
active_positions = {}

# === Load Environment Variables ===
load_dotenv()
API_KEY = os.getenv('COINEX_API_KEY')
API_SECRET = os.getenv('COINEX_API_SECRET')

# === Enhanced Logging Setup ===
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_handler = RotatingFileHandler('bot_diagnostics.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
log_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[log_handler, console_handler])

# === URLs (V2 Endpoints) ===
REST_URL = "https://api.coinex.com/v2"
SPOT_WS_URL = "wss://socket.coinex.com/v2/spot"
PING_URL = f"{REST_URL}/ping"

# === Connection Diagnostics ===
# <--- MODIFIED: Removed time sync test as it's unreliable and not used in official SDKs.
async def perform_connection_tests():
    """Run essential connection tests."""
    system_status["connection_errors"] = []
    tests = {
        "API Connection": test_api_connection,
        "API Authentication": test_api_authentication,
        "WebSocket Connection": test_websocket_connection_async,
    }
    results = {}
    for name, test_func in tests.items():
        try:
            success, message = await test_func() if asyncio.iscoroutinefunction(test_func) else test_func()
            results[name] = {"success": success, "message": message}
            if not success:
                error_msg = f"{name}: {message}"
                system_status["connection_errors"].append(error_msg)
                # Stop if basic connection or auth fails
                if name in ["API Connection", "API Authentication"]: break
        except Exception as e:
            results[name] = {"success": False, "message": str(e)}
            system_status["connection_errors"].append(f"{name}: {str(e)}")
            break
    
    system_status["api_authenticated"] = results.get("API Authentication", {}).get("success", False)
    system_status["websocket_connected"] = results.get("WebSocket Connection", {}).get("success", False)
    return results

def test_api_connection():
    """Test basic API connectivity using the public ping endpoint."""
    try:
        response = requests.get(PING_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('code') == 0 and data.get('message') == 'OK':
            return True, "API is reachable and responding correctly."
        return False, f"API reachable but responded with an error: {data}"
    except Exception as e:
        return False, str(e)

async def test_api_authentication():
    """Test API key authentication by fetching balance."""
    try:
        # A successful call to any private endpoint confirms authentication.
        await private_api_call("asset/spot/balance", "GET")
        return True, "Authentication successful."
    except Exception as e:
        return False, f"Authentication failed: {e}"

async def test_websocket_connection_async():
    """Async WebSocket connection test."""
    try:
        async with websockets.connect(SPOT_WS_URL, ping_interval=20, ping_timeout=30, ssl=ssl.create_default_context()) as ws:
            ping_req = {"method": "common.ping", "params": [], "id": int(time.time())}
            await ws.send(json.dumps(ping_req))
            response = await asyncio.wait_for(ws.recv(), timeout=10)
            if '"pong"' in response:
                return True, "WebSocket connected and responded to ping."
            return False, f"Unexpected WebSocket response: {response}"
    except Exception as e:
        return False, str(e)

# === V2 API Client (Stable Signing Mechanism) ===
async def private_api_call(endpoint, method="GET", params=None):
    """Make an authenticated API request for V2."""
    url = f"{REST_URL}/{endpoint}"
    req_params = params.copy() if params else {}
    
    try:
        timestamp = str(int(time.time() * 1000))
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-COINEX-APIKEY': API_KEY,
            'X-COINEX-TIMESTAMP': timestamp,
        }
        
        if method == 'GET':
            query_string = '&'.join(f'{k}={v}' for k, v in sorted(req_params.items()))
            string_to_sign = f"{query_string}&timestamp={timestamp}{API_SECRET}" if query_string else f"timestamp={timestamp}{API_SECRET}"
            req_params['timestamp'] = timestamp
            response_future = asyncio.to_thread(requests.get, url, params=req_params, headers=headers, timeout=15)
        else: # POST
            body_str = json.dumps(req_params, separators=(',', ':'))
            string_to_sign = f"{body_str}timestamp={timestamp}{API_SECRET}"
            post_body = req_params
            post_body['timestamp'] = timestamp
            response_future = asyncio.to_thread(requests.post, url, data=json.dumps(post_body), headers=headers, timeout=15)

        signature = hashlib.sha256(string_to_sign.encode('utf-8')).hexdigest()
        headers['X-COINEX-SIGNATURE'] = signature
        
        response = await response_future

        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        data = response.json()
        if data.get('code') != 0:
            raise ConnectionError(f"API Error Code {data.get('code')}: {data.get('message')}")
        
        return data.get('data')

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred in private_api_call: {e}")
        raise

# === Market Data & Order Functions (No changes needed) ===
def get_market_info(markets=None):
    url = f"{REST_URL}/market/list"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == 0:
            return {item["name"]: item for item in data.get("data", []) if not markets or item["name"] in markets}
    except Exception as e:
        logging.critical(f"Network error fetching market info: {e}")
        return None

async def fetch_balance():
    balances = await private_api_call("asset/spot/balance", "GET")
    usdt_balance = next((float(asset.get('available', 0)) for asset in balances if asset['ccy'] == 'USDT'), 0.0)
    return {'USDT': {'free': usdt_balance}}

def fetch_ohlcv(symbol, timeframe='5m', limit=100):
    url = f"{REST_URL}/market/kline"
    params = {'market': symbol.replace('/', ''), 'interval': timeframe, 'limit': limit}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get('code') != 0: raise ConnectionError(f"API error: {data.get('message')}")
        df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'amount'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
    except Exception as e:
        logging.error(f"[{symbol}] Failed to fetch OHLCV: {e}")
        raise

async def execute_real_trade(symbol, side, amount):
    if amount <= 0: return None
    try:
        endpoint = "spot/order"
        params = {"market": symbol.replace('/', ''), "side": side, "type": "market"}
        if side.lower() == 'buy': params["amount"] = str(round(amount, 8))
        else: params["quantity"] = str(round(amount, 8))
        return await private_api_call(endpoint, "POST", params)
    except Exception as e:
        logging.error(f"Error executing {side} trade for {symbol}: {e}")
        return None

# === TA, ML, and Logic Functions (No changes needed) ===
def calculate_indicators(df):
    try:
        df['SMA'] = ta.sma(df['close'], length=20)
        df['RSI'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9); df['MACD'] = macd[f'MACD_{12}_{26}_{9}']
        bbands = ta.bbands(df['close'], length=20, std=2); df['upper_band'] = bbands[f'BBU_20_2.0']; df['lower_band'] = bbands[f'BBL_20_2.0']
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3); df['stoch_k'] = stoch[f'STOCHk_14_3_3']
        return df
    except Exception: return df

def generate_ml_signals(df):
    try:
        features = ['SMA', 'RSI', 'MACD', 'stoch_k', 'ATR', 'upper_band', 'lower_band']
        if not all(f in df.columns for f in features): return df
        df_clean = df.dropna(subset=features).copy()
        if df_clean.empty: return df
        model_path = 'trading_model.pkl'
        if os.path.exists(model_path): model = joblib.load(model_path)
        else:
            df_clean['target'] = (df_clean['close'].shift(-1) > df_clean['close']).astype(int)
            X, y = df_clean[features], df_clean['target']
            if len(X) < 50: return df
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(X_train, y_train)
            joblib.dump(model, model_path)
        df['ml_signal'] = model.predict(df[features].fillna(method='ffill').fillna(method='bfill'))
        return df
    except Exception: return df

async def trade_and_manage_position(df, symbol, usdt_per_trade, market_info):
    global active_positions
    if df.empty or 'ml_signal' not in df.columns: return
    last_signal, current_price = df['ml_signal'].iloc[-1], df['close'].iloc[-1]
    min_trade = float(market_info.get('min_amount', 10.0))
    
    if symbol in active_positions:
        pos = active_positions[symbol]
        if last_signal == 0 or current_price <= pos['trailing_stop_price']:
            if await execute_real_trade(symbol, "sell", pos['quantity']): del active_positions[symbol]
            return
        new_stop = current_price * 0.98 # 2% trailing stop
        if new_stop > pos['trailing_stop_price']: pos['trailing_stop_price'] = new_stop
    elif last_signal == 1 and usdt_per_trade >= min_trade:
        if await execute_real_trade(symbol, "buy", usdt_per_trade):
            active_positions[symbol] = {
                'quantity': usdt_per_trade / current_price,
                'trailing_stop_price': current_price * 0.98
            }

# === Main Trading Loop ===
async def run_trading_engine():
    global active_positions
    try:
        logging.info("--- Starting Trading Engine ---")
        await perform_connection_tests()
        if system_status["connection_errors"]:
            raise ConnectionError(f"Initial diagnostics failed: {system_status['connection_errors']}")
        
        symbols = ['BTC/USDT', 'ETH/USDT']
        all_market_info = get_market_info([s.replace("/", "") for s in symbols])
        if not all_market_info: raise ConnectionError("Failed to fetch market info.")
        
        balance_info = await fetch_balance()
        total_balance = balance_info.get('USDT', {}).get('free', 0)
        if total_balance < 20: raise ValueError("Insufficient balance to start trading.")
        
        usdt_per_trade = (total_balance / len(symbols)) * 0.1 # Use 10% of portfolio per trade
        
        dfs = {s: generate_ml_signals(calculate_indicators(fetch_ohlcv(s, '5m', 200))) for s in symbols}
        system_status["trading_active"] = True
        logging.info("--- Trading Engine is now LIVE ---")

        while system_status["server_running"]:
            try:
                for symbol in symbols:
                    latest_data = fetch_ohlcv(symbol, '5m', 2)
                    if not latest_data.empty and latest_data['timestamp'].iloc[-1] > dfs[symbol]['timestamp'].iloc[-1]:
                       dfs[symbol] = pd.concat([dfs[symbol].iloc[1:], latest_data.iloc[[-1]]], ignore_index=True)
                       dfs[symbol] = generate_ml_signals(calculate_indicators(dfs[symbol]))
                    await trade_and_manage_position(dfs[symbol], symbol, usdt_per_trade, all_market_info.get(symbol.replace("/", "")))
                await asyncio.sleep(30)
            except Exception as e:
                logging.error(f"Error in main trading loop: {e}", exc_info=True)
                await asyncio.sleep(60)
    except Exception as e:
        logging.critical(f"Trading engine crashed: {e}")
    finally:
        system_status["trading_active"] = False
        logging.info("--- Closing all open positions on shutdown ---")
        for symbol, position in list(active_positions.items()):
            await execute_real_trade(symbol, "sell", position['quantity'])
        logging.info("--- Trading Engine Shutdown Complete ---")

# === Web Interface & Startup ===
app = Flask(__name__)
@app.route('/')
def home(): return "<h1>Bot Status</h1><p><a href='/status'>JSON</a> | <a href='/diagnostics'>Diagnostics</a></p>"
@app.route('/status')
def status():
    system_status["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    return jsonify({"status": system_status, "positions": active_positions})
@app.route('/diagnostics')
def diagnostics():
    return jsonify(asyncio.run(perform_connection_tests()))

def run_server():
    port = int(os.getenv("PORT", 5000))
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=port, threaded=True)

async def main():
    threading.Thread(target=run_server, daemon=True).start()
    try: await run_trading_engine()
    except KeyboardInterrupt: logging.info("Shutdown signal received.")
    finally: system_status["server_running"] = False

if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        logging.critical("CRITICAL: API_KEY or API_SECRET not found in .env file. Bot cannot start.")
    else:
        asyncio.run(main())
