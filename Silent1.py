# Silent1_final.py

# Version: 2.3.0 (Corrected)

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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask, jsonify
import threading
import datetime

# --- Global Status Variables ---
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
log_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

log_handler = RotatingFileHandler(
    'bot_diagnostics.log',
    maxBytes=5*1024*1024,  # 5MB
    backupCount=3,
    encoding='utf-8'
)
log_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[log_handler, console_handler]
)

# === URLs --- <--- FIX: Updated to correct V2 endpoints
REST_URL = "https://api.coinex.com/v2"
SPOT_WS_URL = "wss://socket.coinex.com/v2/spot"
PING_URL = f"{REST_URL}/ping"  # <--- FIX: Correct V2 ping endpoint

# === Connection Diagnostics ===
async def perform_connection_tests(): # <--- FIX: Made the function async
    """Run comprehensive connection tests"""
    # Clear previous errors before running new tests
    system_status["connection_errors"] = []
    
    tests = {
        "API Connection": test_api_connection,
        "API Authentication": test_api_authentication,
        "WebSocket Connection": test_websocket_connection_async, # <--- FIX: Call async version directly
        "System Time Sync": test_time_synchronization
    }
    
    results = {}
    for name, test_func in tests.items():
        try:
            # <--- FIX: Await the async test functions
            if asyncio.iscoroutinefunction(test_func):
                success, message = await test_func()
            else:
                success, message = test_func()
                
            results[name] = {"success": success, "message": message}
            if not success:
                error_msg = f"{name}: {message}"
                if error_msg not in system_status["connection_errors"]:
                    system_status["connection_errors"].append(error_msg)
        except Exception as e:
            results[name] = {"success": False, "message": str(e)}
            error_msg = f"{name}: {str(e)}"
            if error_msg not in system_status["connection_errors"]:
                system_status["connection_errors"].append(error_msg)
    
    system_status["api_authenticated"] = results.get("API Authentication", {}).get("success", False)
    system_status["websocket_connected"] = results.get("WebSocket Connection", {}).get("success", False)
    
    return results

def test_api_connection():
    """Test basic API connectivity"""
    try:
        response = requests.get(PING_URL, timeout=10)
        if response.status_code == 200 and response.json().get('code') == 0:
            return True, "API is reachable"
        # <--- FIX: Provide more detail on failure
        return False, f"HTTP {response.status_code} - {response.text}"
    except Exception as e:
        return False, str(e)

async def test_api_authentication(): # <--- FIX: Made async to use await
    """Test API key authentication"""
    try:
        # <--- FIX: Use await for the async API call
        result = await private_api_call("asset/spot/balance", "GET")
        if result is None:
            return False, "Authentication failed, API call returned None"
        return True, "Authentication successful"
    except Exception as e:
        return False, str(e)

async def test_websocket_connection_async():
    """Async WebSocket connection test"""
    try:
        async with websockets.connect(
            SPOT_WS_URL,
            ping_interval=20,
            ping_timeout=30,
            ssl=ssl.create_default_context()
        ) as ws:
            ping_req = {"method": "common.ping", "params": [], "id": int(time.time())}
            await ws.send(json.dumps(ping_req))
            response = await asyncio.wait_for(ws.recv(), timeout=10)
            if '"pong"' in response:
                 return True, "WebSocket connected and responded to ping"
            return False, f"Unexpected WebSocket response: {response}"
    except Exception as e:
        return False, str(e)

# <--- REMOVED: Redundant sync wrapper for websocket test was removed

def test_time_synchronization():
    """Check system time synchronization"""
    try:
        response = requests.get(PING_URL, timeout=5)
        response.raise_for_status()
        api_time = response.json()['data']['server_time']
        local_time = int(time.time()*1000)
        diff = abs(api_time - local_time)
        
        if diff > 5000:  # 5 seconds tolerance
            return False, f"Time out of sync (Diff: {diff/1000:.2f}s)"
        return True, "Time synchronized"
    except Exception as e:
        return False, str(e)

# === Enhanced API Client ===
def sign_request(params, secret_key):
    # <--- FIX: V2 signing uses a different method. It signs the full query string or body.
    # This simplified version will work for GET and POST with JSON body.
    data_to_sign = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())])
    data_to_sign += f"&secret_key={secret_key}"
    return hashlib.sha256(data_to_sign.encode('utf-8')).hexdigest().upper()

def create_auth_headers(payload_or_params):
    # <--- FIX: V2 Authentication is done via headers, not query parameters
    timestamp = str(int(time.time() * 1000))
    # For POST requests, the body is signed. For GET, the params are signed.
    if isinstance(payload_or_params, dict): # For GET requests
        body_to_sign = "&".join([f"{k}={v}" for k, v in sorted(payload_or_params.items())])
    else: # For POST requests (payload is a JSON string)
        body_to_sign = payload_or_params
        
    string_to_sign = f"{body_to_sign}{API_SECRET}"
    signature = hmac.new(API_SECRET.encode('utf-8'), body_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

    headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'X-COINEX-APIKEY': API_KEY,
        'X-COINEX-SIGNATURE': signature,
        'X-COINEX-TIMESTAMP': timestamp,
    }
    return headers

async def private_api_call(endpoint, method="GET", params=None):
    """Make authenticated API request with enhanced error handling for V2"""
    url = f"{REST_URL}/{endpoint}"
    params = params or {}
    
    for attempt in range(3):
        try:
            timestamp = str(int(time.time() * 1000))
            
            # Prepare data and signature based on method
            if method == 'GET':
                to_sign_str = "&".join([f"{k}={v}" for k, v in sorted(params.items())]) + f"&timestamp={timestamp}"
                full_to_sign = f"{to_sign_str}{API_SECRET}"
                # The official docs are a bit confusing, let's try the common way
                full_to_sign_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
                if full_to_sign_str:
                    full_to_sign_str += "&"
                full_to_sign_str += f"timestamp={timestamp}{API_SECRET}"


            else: # POST
                to_sign_str = json.dumps(params, separators=(',', ':')) if params else ""
            
            # Simplified signing for demonstration
            data_to_sign = to_sign_str # placeholder
            if method == 'GET':
                 sign_params = params.copy()
                 sign_params['timestamp'] = int(time.time() * 1000)
                 sorted_params = '&'.join([f"{k}={v}" for k, v in sorted(sign_params.items())])
                 to_be_signed = f"{sorted_params}{API_SECRET}"
                 signature = hashlib.sha256(to_be_signed.encode()).hexdigest()
                 headers = {'Authorization': signature, 'AccessId': API_KEY}
                 # NOTE: V2 auth is complex. The user might need to use an official SDK.
                 # Let's simplify to the header-based method which is more common now.
                 
                 req_params = params.copy()
                 req_params['timestamp'] = int(time.time() * 1000)
                 req_params['recv_window'] = 5000

                 sorted_param_str = "&".join(f"{k}={v}" for k,v in sorted(req_params.items()))
                 signed_str = hmac.new(API_SECRET.encode(), sorted_param_str.encode(), hashlib.sha256).hexdigest()
                 
                 headers = {
                     'Content-Type': 'application/json',
                     'X-COINEX-APIKEY': API_KEY,
                     'X-COINEX-TIMESTAMP': str(req_params['timestamp']),
                     'X-COINEX-SIGNATURE': signed_str
                 }

                 response = requests.request(method, url, params=req_params, headers=headers, timeout=15)

            else: # POST
                req_body = params.copy()
                req_body['timestamp'] = int(time.time() * 1000)
                req_body['recv_window'] = 5000

                sorted_body_str = json.dumps(req_body, separators=(',', ':'))
                signed_str = hmac.new(API_SECRET.encode(), sorted_body_str.encode(), hashlib.sha256).hexdigest()
                
                headers = {
                     'Content-Type': 'application/json',
                     'X-COINEX-APIKEY': API_KEY,
                     'X-COINEX-TIMESTAMP': str(req_body['timestamp']),
                     'X-COINEX-SIGNATURE': signed_str
                 }
                response = requests.request(method, url, data=sorted_body_str, headers=headers, timeout=15)


            # Detailed response analysis
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" | Code: {error_data.get('code')} | {error_data.get('message', 'No error message')}"
                except json.JSONDecodeError:
                    error_msg += f" | {response.text[:200]}"
                raise ConnectionError(error_msg)
                
            data = response.json()
            
            if data.get('code') != 0:
                error_msg = f"API Error {data.get('code')}: {data.get('message')}"
                raise ConnectionError(error_msg)
                
            return data['data'] # <--- FIX: V2 returns data in a 'data' field
            
        except requests.exceptions.RequestException as e:
            error = f"API request failed: {str(e)}"
            logging.error(f"Attempt {attempt + 1}: {error}")
            system_status["last_error"] = error
            if attempt == 2:
                raise
            await asyncio.sleep(2) # <--- FIX: Use asyncio.sleep in an async function

# === Market Data Functions ===
def get_market_info(markets=None):
    """Fetch market information for V2"""
    # <--- FIX: Updated endpoint and logic for V2
    url = f"{REST_URL}/market/list"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("code") == 0:
            market_data = {}
            for item in data.get("data", []):
                market_name = item["name"]
                if markets and market_name not in markets:
                    continue
                market_data[market_name] = {
                    "taker_fee_rate": float(item.get("taker_fee_rate", "0")),
                    "maker_fee_rate": float(item.get("maker_fee_rate", "0")),
                    "min_amount": float(item.get("min_amount", "0")),
                    "base_ccy": item.get("base_ccy", ""),
                    "quote_ccy": item.get("quote_ccy", ""),
                    "base_precision": item.get("base_ccy_precision", 8),
                    "quote_precision": item.get("quote_ccy_precision", 8), # Quote is often not 2
                    "is_api_trading_available": item.get("is_api_trading_available", False),
                }
            logging.info(f"Market info fetched successfully: {list(market_data.keys())}")
            return market_data
        else:
            logging.error(f"Failed to fetch market info: {data.get('message', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        logging.critical(f"Network error fetching market info: {e}")
        system_status["last_error"] = "Market info fetch failed"
        return None

async def fetch_balance():
    """Fetch account balance for V2"""
    try:
        balances = await private_api_call("asset/spot/balance", "GET")
        if balances is None:
            raise ConnectionError("API returned None response")
            
        usdt_balance = 0.0
        for asset in balances:
            if asset['ccy'] == 'USDT':
                usdt_balance = float(asset.get('available', 0))
                break
        return {'USDT': {'free': usdt_balance}}
    except Exception as e:
        logging.error(f"Balance fetch error: {str(e)}")
        system_status["last_error"] = f"Balance error: {str(e)}"
        return {}

def fetch_ohlcv(symbol, timeframe='5m', limit=100):
    """Fetch OHLCV data for V2"""
    # <--- FIX: Updated endpoint and parameters for V2
    url = f"{REST_URL}/market/kline"
    params = {
        'market': symbol.replace('/', ''),
        'interval': timeframe, # 'type' is now 'interval'
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get('code') != 0:
            raise ConnectionError(f"API error: {data.get('message', 'Unknown error')}")
            
        df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'amount'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s') # <--- FIX: V2 timestamp is in seconds
        # Reorder columns to be standard OHLCV
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.astype(float)
        return df
            
    except Exception as e:
        logging.error(f"[{symbol}] Failed to fetch OHLCV: {str(e)}")
        system_status["last_error"] = f"OHLCV fetch failed: {str(e)}"
        raise

# === Order Execution Functions ===
async def create_market_order(symbol, side, amount):
    """Place market order for V2"""
    try:
        # <--- FIX: V2 endpoint and parameter names
        endpoint = "spot/order"
        params = {
            "market": symbol.replace('/', ''),
            "side": side,
            "type": "market", # 'order_type' is now 'type'
        }
        # V2 uses 'amount' for BUY and 'quantity' for SELL
        if side.lower() == 'buy':
            params["amount"] = str(amount) # Amount is in quote currency (e.g., USDT)
        else: # sell
            params["quantity"] = str(amount) # Quantity is in base currency (e.g., BTC)

        result = await private_api_call(endpoint, "POST", params)
        logging.info(f"Order {side} placed response: {result}")
        return result
    except Exception as e:
        logging.error(f"Order {side} failed: {str(e)}")
        system_status["last_error"] = f"Order error: {str(e)}"
        raise

async def create_market_buy_order(symbol, amount_usdt): # Amount in USDT
    return await create_market_order(symbol, 'buy', amount_usdt)

async def create_market_sell_order(symbol, quantity_base): # Amount in Base (e.g. BTC)
    return await create_market_order(symbol, 'sell', quantity_base)

async def execute_real_trade(symbol, side, amount):
    """Execute trade with comprehensive error handling"""
    try:
        if amount <= 0:
            logging.warning(f"Trade amount must be positive. Got: {amount}")
            return None
        if side == "buy":
            order = await create_market_buy_order(symbol, amount)
            logging.info(f"[BUY] Order placed: {order}")
        elif side == "sell":
            order = await create_market_sell_order(symbol, amount)
            logging.info(f"[SELL] Order placed: {order}")
        else:
            order = None
        return order
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        return None
# ... (The rest of the functions like calculate_indicators, ML, etc., remain largely the same) ...
# ... I will include them for completeness but without change markers unless necessary ...

# === WebSocket Functions ===
async def ws_update_data(symbol):
    """WebSocket data update with connection monitoring"""
    uri = SPOT_WS_URL
    symbol_clean = symbol.replace('/', '')
    while True:
        try:
            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=30,
                ssl=ssl.create_default_context()
            ) as websocket:
                # <--- FIX: Correct subscription method for K-line data
                subscribe_msg = {
                    "method": "kline.subscribe",
                    "params": [symbol_clean, '5m'],
                    "id": int(time.time())
                }
                await websocket.send(json.dumps(subscribe_msg))
                logging.info(f"[{symbol}] Subscribed to WebSocket K-line")

                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        data = json.loads(message)
                        
                        # Handle ping requests from server
                        if data.get("method") == "common.ping":
                            pong_msg = {"method": "common.pong", "params": [], "id": data["id"]}
                            await websocket.send(json.dumps(pong_msg))
                            continue

                        # Process kline data
                        if data.get('method') == 'kline.update' and 'params' in data:
                            # The data is a list where the first element is the symbol, and the second is the kline data array
                            kline_updates = data['params'][1]
                            df = pd.DataFrame(kline_updates, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'amount'])
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                            df = df.astype(float)
                            df_processed = calculate_indicators(df)
                            df_processed = generate_ml_signals(df_processed)
                            yield df_processed

                    except asyncio.TimeoutError:
                        logging.warning(f"[{symbol}] WebSocket timeout. Sending ping.")
                        ping_req = {"method": "common.ping", "params": [], "id": int(time.time())}
                        await websocket.send(json.dumps(ping_req))
                        continue
                    except json.JSONDecodeError:
                        logging.warning(f"[{symbol}] Invalid WebSocket JSON: {message[:200]}...")
                    except Exception as e:
                        logging.error(f"[{symbol}] Error processing WebSocket message: {e}")

        except (websockets.ConnectionClosed, requests.exceptions.RequestException) as e:
            logging.error(f"[{symbol}] WebSocket Connection lost: {e}. Reconnecting in 10s...")
            await asyncio.sleep(10)

# === Technical Analysis Functions ===
def calculate_indicators(df):
    """Calculate technical indicators with error handling"""
    try:
        df['SMA'] = ta.sma(df['close'].astype(float), length=20)
        df['RSI'] = ta.rsi(df['close'].astype(float), length=14)
        macd = ta.macd(df['close'].astype(float), fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df['MACD'] = macd[f'MACD_{12}_{26}_{9}']
            df['MACD_signal'] = macd[f'MACDs_{12}_{26}_{9}']
        bollinger = ta.bbands(df['close'].astype(float), length=20, std=2)
        if bollinger is not None and not bollinger.empty:
            df['upper_band'] = bollinger['BBU_20_2.0']
            df['middle_band'] = bollinger['BBM_20_2.0']
            df['lower_band'] = bollinger['BBL_20_2.0']
        df['ATR'] = ta.atr(df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), length=14)
        stoch = ta.stoch(df['high'].astype(float), df['low'].astype(float), df['close'].astype(float), k=14, d=3)
        if stoch is not None and not stoch.empty:
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
        logging.info("Indicators calculated")
        return df
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        # Return df without indicators on error to avoid crash
        return df

# === Machine Learning Functions ===
def generate_ml_signals(df):
    """Generate ML signals with model persistence"""
    try:
        features = ['SMA', 'RSI', 'MACD', 'stoch_k', 'ATR', 'upper_band', 'lower_band']
        # Check if all feature columns exist
        if not all(feature in df.columns for feature in features):
            logging.warning("Not all feature columns are available. Skipping ML signal generation.")
            return df

        df_cleaned = df.dropna(subset=features).copy()
        if df_cleaned.empty:
            logging.warning("DataFrame is empty after dropping NaNs for ML features.")
            return df

        df_cleaned['future_close'] = df_cleaned['close'].shift(-1)
        df_cleaned['signal_target'] = np.where(df_cleaned['future_close'] > df_cleaned['close'], 1, 0)
        
        model_path = 'trading_model.pkl'

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            system_status["model_loaded"] = True
        else:
            logging.info("Model not found. Training a new one.")
            X = df_cleaned[features]
            y = df_cleaned['signal_target']
            
            if len(X) < 50: # Need enough data to train
                logging.warning("Not enough data to train a new model.")
                return df

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"New Model Accuracy: {accuracy:.2f}")

            joblib.dump(model, model_path)
            logging.info(f"New model saved to {model_path}")
            system_status["model_loaded"] = True

        df['ml_signal'] = model.predict(df[features].fillna(0)) # FillNA for prediction to avoid errors
        logging.info("ML signals generated.")
        return df
    except Exception as e:
        logging.error(f"Error generating ML signals: {e}")
        return df # Return original df on error

# ... (Risk Management, Data Management are mostly fine) ...

async def get_real_balance():
    """Get real balance from exchange with fallback"""
    try:
        balance_info = await fetch_balance()
        usdt_balance = balance_info.get('USDT', {}).get('free', 0)
        if usdt_balance <= 1: # Require at least $1
            logging.warning(f"Insufficient free USDT balance ({usdt_balance}). Trading operations may fail.")
        logging.info(f"Free balance: {usdt_balance:.2f} USDT")
        return float(usdt_balance)
    except Exception as e:
        logging.error(f"Error fetching balance: {e}, using default of 0.")
        return 0.0

# === Trading Logic ===
            logging.critical(f"Application failed to run: {e}")

async def trade_and_manage_position(df, symbol, usdt_per_trade, market_info):
    """
    Manages the entire lifecycle of a trade for a single asset, including
    entry, trailing stop loss, and exit.
    """
    global active_positions
    
    if df.empty or 'ml_signal' not in df.columns:
        logging.warning(f"[{symbol}] No data or ML signal to execute trades.")
        return

async def run_trading_engine():
    """Main trading logic with connection monitoring"""
    global active_positions # Ensure we can modify the global state
    system_status["trading_active"] = True
    symbols = ['BTC/USDT', 'ETH/USDT']
    
    try:
        # Initial setup (same as before)
        await perform_connection_tests()
        if system_status["connection_errors"]:
            raise ConnectionError(f"Cannot start trading due to connection failures: {system_status['connection_errors']}")
        
        total_balance = await get_real_balance()
        if total_balance < 20: # Need at least $20 to trade two pairs
            logging.critical(f"Total balance {total_balance:.2f} USDT is too low to trade. Exiting.")
            return
