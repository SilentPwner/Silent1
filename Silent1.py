# Silent1_final.py -
# Version: 2.5.1 (Corrected Time Sync Endpoint)

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
async def perform_connection_tests():
    """Run comprehensive connection tests in a specific order."""
    system_status["connection_errors"] = []
    tests = {
        "API Connection": test_api_connection,
        "System Time Sync": test_time_synchronization,
        "API Authentication": test_api_authentication,
        "WebSocket Connection": test_websocket_connection_async,
    }
    results = {}
    for name, test_func in tests.items():
        try:
            if asyncio.iscoroutinefunction(test_func):
                success, message = await test_func()
            else:
                success, message = test_func()
            results[name] = {"success": success, "message": message}
            if not success:
                error_msg = f"{name}: {message}"
                if error_msg not in system_status["connection_errors"]:
                    system_status["connection_errors"].append(error_msg)
                if name in ["API Connection", "System Time Sync"]: break
        except Exception as e:
            results[name] = {"success": False, "message": str(e)}
            error_msg = f"{name}: {str(e)}"
            if error_msg not in system_status["connection_errors"]:
                system_status["connection_errors"].append(error_msg)
            break
    
    system_status["api_authenticated"] = results.get("API Authentication", {}).get("success", False)
    system_status["websocket_connected"] = results.get("WebSocket Connection", {}).get("success", False)
    return results

def test_api_connection():
    """Test basic API connectivity."""
    try:
        response = requests.get(PING_URL, timeout=10)
        if response.status_code == 200 and response.json().get('code') == 0:
            return True, "API is reachable"
        return False, f"HTTP {response.status_code} - {response.text}"
    except Exception as e:
        return False, str(e)

async def test_api_authentication():
    """Test API key authentication by fetching balance."""
    try:
        result = await private_api_call("asset/spot/balance", "GET")
        if result is None:
            return False, "Authentication failed, API call returned None"
        return True, "Authentication successful"
    except Exception as e:
        return False, str(e)

async def test_websocket_connection_async():
    """Async WebSocket connection test."""
    try:
        async with websockets.connect(SPOT_WS_URL, ping_interval=20, ping_timeout=30, ssl=ssl.create_default_context()) as ws:
            ping_req = {"method": "common.ping", "params": [], "id": int(time.time())}
            await ws.send(json.dumps(ping_req))
            response = await asyncio.wait_for(ws.recv(), timeout=10)
            if '"pong"' in response:
                return True, "WebSocket connected and responded to ping"
            return False, f"Unexpected WebSocket response: {response}"
    except Exception as e:
        return False, str(e)

# <--- FIX: Corrected the URL used for time synchronization ---
def test_time_synchronization():
    """Check system time synchronization against the server's ping endpoint."""
    try:
        # Use the correct public /ping endpoint which contains server_time
        response = requests.get(PING_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check if the response is valid and contains the necessary data
        if data.get('code') == 0 and 'data' in data and 'server_time' in data['data']:
            api_time = data['data']['server_time']
            local_time = int(time.time() * 1000)
            diff = abs(api_time - local_time)
            
            if diff > 5000:  # 5 seconds tolerance
                return False, f"Time out of sync (Diff: {diff/1000:.2f}s)"
            return True, "Time synchronized"
        else:
            # Handle cases where the ping response is malformed or an error occurred
            return False, f"Invalid response from ping endpoint: {data.get('message', 'No data field')}"
            
    except Exception as e:
        return False, str(e)


# === V2 API Client with Corrected Signing ===
async def private_api_call(endpoint, method="GET", params=None):
    """Make authenticated API request for V2 with corrected signing."""
    url = f"{REST_URL}/{endpoint}"
    # Use a new dictionary for params to avoid modifying the original
    req_params = params.copy() if params else {}
    
    for attempt in range(3):
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
                signature = hashlib.sha256(string_to_sign.encode('utf-8')).hexdigest()
                headers['X-COINEX-SIGNATURE'] = signature
                
                req_params['timestamp'] = timestamp
                response = requests.get(url, params=req_params, headers=headers, timeout=15)

            else: # POST
                body_str = json.dumps(req_params, separators=(',', ':'))
                string_to_sign = f"{body_str}timestamp={timestamp}{API_SECRET}"
                signature = hashlib.sha256(string_to_sign.encode('utf-8')).hexdigest()
                headers['X-COINEX-SIGNATURE'] = signature
                
                post_body = req_params.copy()
                post_body['timestamp'] = timestamp
                response = requests.post(url, data=json.dumps(post_body), headers=headers, timeout=15)

            if response.status_code != 200:
                raise ConnectionError(f"HTTP {response.status_code} | {response.text[:200]}")
            
            data = response.json()
            if data.get('code') != 0:
                raise ConnectionError(f"API Error {data.get('code')}: {data.get('message')}")
            
            return data.get('data')

        except requests.exceptions.RequestException as e:
            error = f"API request failed on attempt {attempt + 1}: {e}"
            logging.warning(error)
            if attempt == 2: raise
            await asyncio.sleep(2)
        except Exception as e:
            logging.error(f"An unexpected error occurred in private_api_call: {e}")
            raise

# --- The rest of the code is unchanged and correct ---

# === Market Data Functions ===
def get_market_info(markets=None):
    """Fetch market information for V2."""
    url = f"{REST_URL}/market/list"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == 0:
            market_data = {}
            for item in data.get("data", []):
                market_name = item["name"]
                if markets and market_name not in markets: continue
                market_data[market_name] = {
                    "taker_fee_rate": float(item.get("taker_fee_rate", "0")),
                    "maker_fee_rate": float(item.get("maker_fee_rate", "0")),
                    "min_amount": float(item.get("min_amount", "0")),
                    "base_ccy": item.get("base_ccy", ""),
                    "quote_ccy": item.get("quote_ccy", ""),
                    "base_precision": item.get("base_ccy_precision", 8),
                    "quote_precision": item.get("quote_ccy_precision", 8),
                    "is_api_trading_available": item.get("is_api_trading_available", False),
                }
            return market_data
        else:
            return None
    except requests.exceptions.RequestException as e:
        logging.critical(f"Network error fetching market info: {e}")
        return None

async def fetch_balance():
    """Fetch account balance for V2."""
    try:
        balances = await private_api_call("asset/spot/balance", "GET")
        if balances is None: raise ConnectionError("API returned None response for balance")
        usdt_balance = next((float(asset.get('available', 0)) for asset in balances if asset['ccy'] == 'USDT'), 0.0)
        return {'USDT': {'free': usdt_balance}}
    except Exception as e:
        logging.error(f"Balance fetch error: {e}")
        raise

def fetch_ohlcv(symbol, timeframe='5m', limit=100):
    """Fetch OHLCV data for V2."""
    url = f"{REST_URL}/market/kline"
    params = {'market': symbol.replace('/', ''), 'interval': timeframe, 'limit': limit}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get('code') != 0: raise ConnectionError(f"API error: {data.get('message', 'Unknown error')}")
        df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'amount'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.astype(float)
        return df
    except Exception as e:
        logging.error(f"[{symbol}] Failed to fetch OHLCV: {e}")
        raise

# === Order Execution Functions ===
async def execute_real_trade(symbol, side, amount):
    """Execute trade with comprehensive error handling."""
    if amount <= 0: return None
    try:
        endpoint = "spot/order"
        params = {"market": symbol.replace('/', ''), "side": side, "type": "market"}
        if side.lower() == 'buy':
            params["amount"] = str(round(amount, 8))
        else:
            params["quantity"] = str(round(amount, 8))
        return await private_api_call(endpoint, "POST", params)
    except Exception as e:
        logging.error(f"Error executing {side} trade for {symbol}: {e}")
        return None

# === TA, ML, and Logic Functions ===
def calculate_indicators(df):
    try:
        df['SMA'] = ta.sma(df['close'].astype(float), length=20)
        df['RSI'] = ta.rsi(df['close'].astype(float), length=14)
        macd = ta.macd(df['close'].astype(float), fast=12, slow=26, signal=9)
        if macd is not None: df['MACD'] = macd[f'MACD_{12}_{26}_{9}']
        bollinger = ta.bbands(df['close'].astype(float), length=20, std=2)
        if bollinger is not None:
            df['upper_band'] = bollinger['BBU_20_2.0']
            df['lower_band'] = bollinger['BBL_20_2.0']
        df['ATR'] = ta.atr(df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), length=14)
        stoch = ta.stoch(df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), k=14, d=3)
        if stoch is not None: df['stoch_k'] = stoch['STOCHk_14_3_3']
        return df
    except Exception as e: return df

def generate_ml_signals(df):
    try:
        features = ['SMA', 'RSI', 'MACD', 'stoch_k', 'ATR', 'upper_band', 'lower_band']
        if not all(feature in df.columns for feature in features): return df
        df_cleaned = df.dropna(subset=features).copy()
        if df_cleaned.empty: return df
        model_path = 'trading_model.pkl'
        if os.path.exists(model_path): model = joblib.load(model_path)
        else:
            df_cleaned['signal_target'] = np.where(df_cleaned['close'].shift(-1) > df_cleaned['close'], 1, 0)
            X = df_cleaned[features]
            y = df_cleaned['signal_target']
            if len(X) < 50: return df
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)
        df['ml_signal'] = model.predict(df[features].fillna(method='ffill').fillna(method='bfill'))
        return df
    except Exception as e: return df

async def get_real_balance():
    try:
        balance_info = await fetch_balance()
        return float(balance_info.get('USDT', {}).get('free', 0))
    except Exception: return 0.0

async def trade_and_manage_position(df, symbol, usdt_per_trade, market_info):
    global active_positions
    if df.empty or 'ml_signal' not in df.columns: return
    last_signal = df['ml_signal'].iloc[-1]
    current_price = df['close'].iloc[-1]
    min_trade_amount = market_info.get('min_amount', 10.0)
    trailing_stop_percent = 0.02
    if symbol in active_positions:
        pos = active_positions[symbol]
        if last_signal == 0 or current_price <= pos['trailing_stop_price']:
            if await execute_real_trade(symbol, "sell", pos['quantity']):
                del active_positions[symbol]
            return
        new_stop = current_price * (1 - trailing_stop_percent)
        if new_stop > pos['trailing_stop_price']:
            pos['trailing_stop_price'] = new_stop
    elif last_signal == 1 and usdt_per_trade >= min_trade_amount:
        if await execute_real_trade(symbol, "buy", usdt_per_trade):
            active_positions[symbol] = {
                'buy_price': current_price,
                'quantity': usdt_per_trade / current_price,
                'trailing_stop_price': current_price * (1 - trailing_stop_percent)
            }

# === Main Trading Loop ===
async def run_trading_engine():
    global active_positions
    system_status["trading_active"] = True
    symbols = ['BTC/USDT', 'ETH/USDT']
    try:
        await perform_connection_tests()
        if system_status["connection_errors"]:
            raise ConnectionError(f"Cannot start: {system_status['connection_errors']}")
        total_balance = await get_real_balance()
        if total_balance < 20:
            logging.critical("Balance too low.")
            return
        usdt_per_trade = (total_balance / len(symbols)) * 0.1
        all_market_info = get_market_info([s.replace("/", "") for s in symbols])
        if not all_market_info: raise ConnectionError("Failed market info.")
        dfs = {s: generate_ml_signals(calculate_indicators(fetch_ohlcv(s, '5m', 200))) for s in symbols}
        while system_status["server_running"]:
            try:
                if int(time.time()) % 900 == 0: await perform_connection_tests()
                for symbol in symbols:
                    latest_data = fetch_ohlcv(symbol, '5m', 2)
                    if not latest_data.empty and latest_data['timestamp'].iloc[-1] > dfs[symbol]['timestamp'].iloc[-1]:
                       dfs[symbol] = pd.concat([dfs[symbol].iloc[1:], latest_data.iloc[[-1]]], ignore_index=True)
                       dfs[symbol] = generate_ml_signals(calculate_indicators(dfs[symbol]))
                    await trade_and_manage_position(dfs[symbol], symbol, usdt_per_trade, all_market_info.get(symbol.replace("/", "")))
                await asyncio.sleep(30)
            except Exception as e:
                logging.error(f"Main loop error: {e}", exc_info=True)
                await asyncio.sleep(30)
    except Exception as e:
        logging.critical(f"Trading engine crashed: {e}")
    finally:
        system_status["trading_active"] = False
        for symbol, position in list(active_positions.items()):
            await execute_real_trade(symbol, "sell", position['quantity'])

# === Web Interface & Startup ===
app = Flask(__name__)
@app.route('/')
def home(): return "<h1>Bot</h1><p><a href='/status'>Status</a></p><p><a href='/diagnostics'>Diagnostics</a></p>"
@app.route('/status')
def status():
    system_status["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    return jsonify({"status": system_status, "positions": active_positions})
@app.route('/diagnostics')
def diagnostics():
    try:
        return jsonify({"report": asyncio.run(perform_connection_tests()), "status": system_status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_server():
    port = int(os.getenv("PORT", "5000"))
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=port, threaded=True)

async def main():
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    try:
        await asyncio.sleep(2)
        await run_trading_engine()
    except KeyboardInterrupt: pass
    finally:
        system_status["server_running"] = False
        logging.info("Shutdown complete.")

if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        logging.critical("API keys not found in .env file.")
    else:
        asyncio.run(main())
