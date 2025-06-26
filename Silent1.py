
# Version: 2.2.0

import requests
import json
import time
import hmac
import hashlib
import pandas as pd
import asyncio
import websockets
import logging
import socket
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

# === URLs ===
REST_URL = "https://api.coinex.com/v2"
SPOT_WS_URL = "wss://socket.coinex.com/v2/spot"
PING_URL = f"{REST_URL}/spot/ping"

# === Connection Diagnostics ===
def perform_connection_tests():
    """Run comprehensive connection tests"""
    tests = {
        "API Connection": test_api_connection,
        "API Authentication": test_api_authentication,
        "WebSocket Connection": test_websocket_connection,
        "System Time Sync": test_time_synchronization
    }
    
    results = {}
    for name, test in tests.items():
        try:
            success, message = test()
            results[name] = {"success": success, "message": message}
            if not success:
                system_status["connection_errors"].append(f"{name}: {message}")
        except Exception as e:
            results[name] = {"success": False, "message": str(e)}
            system_status["connection_errors"].append(f"{name}: {str(e)}")
    
    system_status["api_authenticated"] = results["API Authentication"]["success"]
    system_status["websocket_connected"] = results["WebSocket Connection"]["success"]
    
    return results

def test_api_connection():
    """Test basic API connectivity"""
    try:
        response = requests.get(PING_URL, timeout=10)
        if response.status_code == 200:
            return True, "API is reachable"
        return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def test_api_authentication():
    """Test API key authentication"""
    try:
        result = private_api_call("spot/balance", "GET")
        if result is None:
            return False, "Authentication failed"
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
            await ws.ping()
            return True, "WebSocket connected"
    except Exception as e:
        return False, str(e)

def test_websocket_connection():
    """Sync wrapper for WebSocket test"""
    try:
        success, message = asyncio.run(test_websocket_connection_async())
        return success, message
    except Exception as e:
        return False, str(e)

def test_time_synchronization():
    """Check system time synchronization"""
    try:
        api_time = requests.get(PING_URL, timeout=5).json()['data']['server_time']
        local_time = int(time.time()*1000)
        diff = abs(api_time - local_time)
        
        if diff > 30000:  # 30 seconds
            return False, f"Time out of sync (Diff: {diff//1000}s)"
        return True, "Time synchronized"
    except Exception as e:
        return False, str(e)

# === Enhanced API Client ===
def sign_request(params, secret):
    """Generate API signature"""
    sorted_params = '&'.join([f"{k}={params[k]}" for k in sorted(params)])
    return hmac.new(
        secret.encode('utf-8'),
        sorted_params.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def private_api_call(endpoint, method="GET", params=None):
    """Make authenticated API request with enhanced error handling"""
    url = f"{REST_URL}/{endpoint}"
    timestamp = int(time.time() * 1000)
    
    headers = {
        'Content-Type': 'application/json',
        'X-COINEX-APIKEY': API_KEY,
    }
    
    params = params or {}
    params.update({
        'timestamp': timestamp,
        'signature_type': 2,
        'recv_window': 5000
    })
    
    for attempt in range(3):  # Retry up to 3 times
        try:
            params['sign'] = sign_request(params, API_SECRET)
            
            response = requests.request(
                method,
                url,
                params=params if method == "GET" else None,
                json=params if method != "GET" else None,
                headers=headers,
                timeout=15
            )
            
            # Detailed response analysis
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" | {error_data.get('message', 'No error message')}"
                except:
                    error_msg += f" | {response.text[:200]}"
                raise ConnectionError(error_msg)
                
            data = response.json()
            
            if data.get('code') != 0:
                error_msg = f"API Error {data.get('code')}: {data.get('message')}"
                raise ConnectionError(error_msg)
                
            return data
            
        except requests.exceptions.SSLError as e:
            error = f"SSL Error: {str(e)} - Check SSL certificates"
            logging.critical(error)
            system_status["last_error"] = error
            if attempt == 2:
                raise
            time.sleep(2)
        except requests.exceptions.Timeout:
            error = "Connection timeout (15s)"
            logging.error(error)
            system_status["last_error"] = error
            if attempt == 2:
                raise
            time.sleep(2)
        except Exception as e:
            error = f"API request failed: {str(e)}"
            logging.error(error)
            system_status["last_error"] = error
            if attempt == 2:
                raise
            time.sleep(2)

# === Market Data Functions ===
def get_market_info(markets=None):
    """Fetch market information with enhanced error handling"""
    url = f"{REST_URL}/spot/market"
    params = {}
    if markets:
        if len(markets) > 10:
            raise ValueError("Max 10 markets allowed.")
        params["market"] = ",".join(markets)

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            try:
                data = response.json()
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON response: {response.text[:200]}...")
                time.sleep(2)
                continue

            if data.get("code") == 0:
                market_data = {}
                for item in data.get("data", []):
                    market_name = item["market"]
                    market_data[market_name] = {
                        "taker_fee_rate": float(item.get("taker_fee_rate", "0")),
                        "maker_fee_rate": float(item.get("maker_fee_rate", "0")),
                        "min_amount": float(item.get("min_amount", "0")),
                        "base_ccy": item.get("base_ccy", ""),
                        "quote_ccy": item.get("quote_ccy", ""),
                        "base_precision": item.get("base_ccy_precision", 8),
                        "quote_precision": item.get("quote_ccy_precision", 2),
                        "status": item.get("status", "offline"),
                        "is_api_trading_available": item.get("is_api_trading_available", False),
                    }
                logging.info(f"Market info fetched successfully: {list(market_data.keys())}")
                return market_data
            else:
                error_msg = data.get("message", "Unknown error")
                logging.error(f"Failed to fetch market info: {error_msg}")
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            logging.warning(f"Network error ({attempt + 1}/3): {e}")
            time.sleep(2)
    
    logging.critical("Failed to fetch market info after multiple attempts")
    system_status["last_error"] = "Market info fetch failed"
    return None

def fetch_balance():
    """Fetch account balance with error handling"""
    try:
        result = private_api_call("spot/balance", "GET")
        if result is None:
            raise ConnectionError("API returned None response")
            
        balances = result.get('data', {})
        usdt_free = float(balances.get('USDT', {}).get('available', 0))
        return {'USDT': {'free': usdt_free}}
    except Exception as e:
        logging.error(f"Balance fetch error: {str(e)}")
        system_status["last_error"] = f"Balance error: {str(e)}"
        return {}

def fetch_ohlcv(symbol, timeframe='5m', limit=100):
    """Fetch OHLCV data with retry logic"""
    url = f"{REST_URL}/spot/kline-data"
    params = {
        'market': symbol.replace('/', ''),
        'type': timeframe,
        'limit': limit
    }
    
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data.get('code') != 0:
                error_msg = data.get('message', 'Unknown error')
                raise ConnectionError(f"API error: {error_msg}")
                
            df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except Exception as e:
            logging.warning(f"[{symbol}] Attempt {attempt + 1}/3 failed: {str(e)}")
            if attempt == 2:
                logging.error(f"[{symbol}] Failed to fetch OHLCV after 3 attempts")
                system_status["last_error"] = f"OHLCV fetch failed: {str(e)}"
                raise
            time.sleep(2)

# === Order Execution Functions ===
def create_market_order(symbol, side, amount):
    """Place market order with enhanced logging"""
    try:
        if amount < 0.0001:
            raise ValueError("Amount too small")
            
        endpoint = "spot/order/place"
        params = {
            "market": symbol.replace('/', ''),
            "side": side,
            "amount": amount,
            "order_type": "market"
        }
        
        result = private_api_call(endpoint, "POST", params)
        logging.info(f"Order {side} executed: {result}")
        return result
    except Exception as e:
        logging.error(f"Order {side} failed: {str(e)}")
        system_status["last_error"] = f"Order error: {str(e)}"
        raise

def create_market_buy_order(symbol, amount):
    return create_market_order(symbol, 'buy', amount)

def create_market_sell_order(symbol, amount):
    return create_market_order(symbol, 'sell', amount)

def execute_real_trade(symbol, side, amount):
    """Execute trade with comprehensive error handling"""
    try:
        if amount < 0.0001:
            logging.warning(f"Insufficient amount: {amount:.8f}")
            return None
        if side == "buy":
            order = create_market_buy_order(symbol, amount)
            logging.info(f"[BUY] Order executed: {order}")
        elif side == "sell":
            order = create_market_sell_order(symbol, amount)
            logging.info(f"[SELL] Order executed: {order}")
        return order
    except Exception as e:
        logging.error(f"Error executing trade: {e}")
        return None

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
                subscribe_msg = {
                    "method": "state",
                    "params": [symbol_clean, '5m', 100],
                    "id": int(time.time())
                }
                await websocket.send(json.dumps(subscribe_msg))
                logging.info(f"[{symbol}] Subscribed to WebSocket")

                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        try:
                            data = json.loads(message)
                            if 'data' in data:
                                ohlcv = data['data']
                                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                                df = calculate_indicators(df)
                                df = generate_ml_signals(df)
                                yield df
                        except json.JSONDecodeError:
                            logging.warning(f"[{symbol}] Invalid WebSocket JSON: {message[:200]}...")

                    except asyncio.TimeoutError:
                        await websocket.ping()
                        continue

        except (websockets.ConnectionClosed, requests.exceptions.RequestException) as e:
            logging.error(f"[{symbol}] Connection lost. Reconnecting...")
            await asyncio.sleep(10)

# === Technical Analysis Functions ===
def calculate_indicators(df):
    """Calculate technical indicators with error handling"""
    try:
        df['SMA'] = ta.sma(df['close'], length=20)
        df['RSI'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd[f'MACD_{12}_{26}_{9}']
        df['MACD_signal'] = macd[f'MACDs_{12}_{26}_{9}']
        bollinger = ta.bbands(df['close'], length=20, std=2)
        df['upper_band'] = bollinger['BBU_20_2.0']
        df['middle_band'] = bollinger['BBM_20_2.0']
        df['lower_band'] = bollinger['BBL_20_2.0']
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']
        logging.info("Indicators calculated")
        return df
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        raise

# === Machine Learning Functions ===
def generate_ml_signals(df):
    """Generate ML signals with model persistence"""
    try:
        features = ['SMA', 'RSI', 'MACD', 'stoch_k', 'ATR', 'upper_band', 'lower_band']
        df = df.dropna(subset=features)
        df['future_close'] = df['close'].shift(-1)
        df['ml_signal'] = np.where(df['future_close'] > df['close'], 1, 0)

        model_path = 'trading_model.pkl'

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logging.info("Loaded existing model.")
        else:
            X = df[features]
            y = df['ml_signal']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
            
            model = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Model Accuracy: {accuracy:.2f}")

            if not os.path.exists(model_path) or accuracy > get_previous_model_accuracy():
                joblib.dump(model, model_path)
                save_model_accuracy(accuracy)
                logging.info("New model saved.")

        df['ml_signal'] = model.predict(df[features])
        logging.info("ML signals generated.")
        return df
    except Exception as e:
        logging.error(f"Error generating ML signals: {e}")
        raise

def get_previous_model_accuracy():
    """Get previous model accuracy from file"""
    acc_file = 'model_accuracy.txt'
    if os.path.exists(acc_file):
        with open(acc_file, 'r') as f:
            try:
                return float(f.read().strip())
            except:
                return 0.0
    return 0.0

def save_model_accuracy(accuracy):
    """Save model accuracy to file"""
    with open('model_accuracy.txt', 'w') as f:
        f.write(f"{accuracy:.4f}")

# === Risk Management Functions ===
def calculate_var(returns, window=20, confidence_level=0.95):
    """Calculate Value at Risk"""
    if len(returns) < window:
        return 0.02
    recent_returns = returns[-window:]
    var = -np.percentile(recent_returns, 100 * (1 - confidence_level))
    return abs(var)

def get_real_balance():
    """Get real balance from exchange with fallback"""
    try:
        balance_info = fetch_balance()
        usdt_balance = balance_info.get('USDT', {}).get('free', 0)
        if usdt_balance <= 0:
            logging.warning("No free USDT balance. Using default $100.")
            return 100.0
        logging.info(f"Free balance: {usdt_balance:.2f} USDT")
        return float(usdt_balance)
    except Exception as e:
        logging.error(f"Error fetching balance: {e}")
        return 100.0

# === Data Management Functions ===
def fetch_new_data_only(df_old, symbol, timeframe):
    """Fetch only new data since last timestamp"""
    try:
        latest_timestamp = df_old['timestamp'].iloc[-1]
        new_df = fetch_ohlcv(symbol, timeframe, limit=10)
        new_data = new_df[new_df['timestamp'] > latest_timestamp]
        if not new_data.empty:
            updated_df = pd.concat([df_old, new_data], ignore_index=True)
            logging.info(f"[{symbol}] Data updated. New candles: {len(new_data)}")
            return updated_df
        else:
            logging.info(f"[{symbol}] No new data available.")
            return df_old
    except Exception as e:
        logging.error(f"[{symbol}] Error updating data: {e}")
        return df_old

# === Trading Logic ===
def execute_trades(df, symbol, per_asset_balance):
    """Execute trading strategy with position management"""
    try:
        position = 0
        trade_count = 0
        max_trades_per_day = 5
        last_trade_time = 0
        consecutive_losses = 0

        for i in range(1, len(df)):
            current_time = time.time()
            if current_time - last_trade_time < 1800 or trade_count >= max_trades_per_day:
                continue

            current_price = df['close'].iloc[i]

            if 'ml_signal' not in df.columns:
                logging.warning("No ML signal yet. Skipping trades.")
                continue

            returns = df['close'].pct_change().dropna()
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            win_prob = len(wins) / len(returns)
            avg_win = wins.mean() if not wins.empty else 0.005
            avg_loss = abs(losses.mean()) if not losses.empty else 0.005
            kelly = win_prob - ((1 - win_prob) / (avg_win / avg_loss))
            kelly = max(0.01, min(kelly, 0.2))
            risk_amount = calculate_var(returns, window=20)

            if df['ml_signal'].iloc[i - 1] == 1 and position == 0:
                amount_to_invest = per_asset_balance * kelly
                amount = amount_to_invest / current_price
                buy_price = current_price
                stop_loss = buy_price * (1 - risk_amount)
                take_profit = buy_price * (1 + risk_amount * 2)
                execute_real_trade(symbol, "buy", amount)
                position = 1
                last_trade_time = current_time
                trade_count += 1

            elif df['ml_signal'].iloc[i - 1] == 0 and position == 1:
                execute_real_trade(symbol, "sell", amount)
                sell_price = current_price
                profit_percent = (sell_price - buy_price) / buy_price * 100
                profit_value = amount * profit_percent / 100
                per_asset_balance += profit_value
                position = 0
                trade_count += 1
                if profit_percent < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

            if position == 1:
                trailing_stop = current_price * 0.98
                if current_price <= stop_loss or current_price <= trailing_stop:
                    execute_real_trade(symbol, "sell", amount)
                    sell_price = current_price
                    profit_percent = (sell_price - buy_price) / buy_price * 100
                    profit_value = amount * profit_percent / 100
                    per_asset_balance += profit_value
                    position = 0
                    if profit_percent < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

            if consecutive_losses >= 3:
                logging.info("Max consecutive losses reached.")
                break

        logging.info(f"[{symbol}] Final balance: {per_asset_balance:.2f} USD")
        return per_asset_balance
    except Exception as e:
        logging.error(f"Error in trading logic: {e}")
        return per_asset_balance

# === Main Trading Loop ===
async def run_trading_engine():
    """Main trading logic with connection monitoring"""
    system_status["trading_active"] = True
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    try:
        # Initial connection check
        connection_report = perform_connection_tests()
        if system_status["connection_errors"]:
            raise ConnectionError("Initial connection tests failed")
        
        # Initialize trading
        total_balance = get_real_balance()
        investment_capital = total_balance * 0.2
        per_asset_balance = investment_capital / len(symbols)
        
        # Get market info
        markets_list = [s.replace("/", "") for s in symbols]
        market_info = get_market_info(markets_list)
        if not market_info:
            raise ConnectionError("Failed to get market info")

        # Prepare initial data
        dfs = {}
        for symbol in symbols:
            market_key = symbol.replace("/", "")
            if not market_info[market_key].get("is_api_trading_available", False):
                logging.error(f"API trading disabled for {market_key}")
                continue

            df = fetch_ohlcv(symbol, '5m', limit=100)
            df = calculate_indicators(df)
            df = generate_ml_signals(df)
            dfs[symbol] = df
            logging.info(f"[{symbol}] Data and signals prepared.")

        # Main trading loop
        while system_status["server_running"]:
            try:
                # Periodic connection check (every 10 minutes)
                if int(time.time()) % 600 == 0:
                    perform_connection_tests()
                    if system_status["connection_errors"]:
                        raise ConnectionError("Periodic connection check failed")
                
                # Update data and execute trades
                for symbol in symbols:
                    if symbol not in dfs:
                        continue
                        
                    df = dfs[symbol]
                    df = fetch_new_data_only(df, symbol, '5m')
                    if len(df) > len(dfs[symbol]):
                        df = calculate_indicators(df)
                        df = generate_ml_signals(df)
                        dfs[symbol] = df
                        logging.info(f"[{symbol}] Data & signals updated.")
                    
                    final_balance = execute_trades(df, symbol, per_asset_balance)
                    logging.info(f"[{symbol}] Return: {final_balance:.2f} USD")

                overall_return = sum([final_balance for _, final_balance in dfs.items()])
                logging.info(f"Total return after diversification: {overall_return:.2f} USD")
                
                await asyncio.sleep(60)
                
            except ConnectionError as e:
                logging.error(f"Connection error: {str(e)}")
                system_status["last_error"] = str(e)
                await asyncio.sleep(60)  # Wait before retry
            except Exception as e:
                logging.error(f"Trading error: {str(e)}")
                system_status["last_error"] = str(e)
                await asyncio.sleep(30)
                
    except Exception as e:
        logging.critical(f"Trading engine failed: {str(e)}")
        raise
    finally:
        system_status["trading_active"] = False

# === Web Interface ===
app = Flask(__name__)

@app.route('/')
def home():
    return """
    <h1>CoinEx Trading Bot</h1>
    <p><a href="/status">System Status</a></p>
    <p><a href="/diagnostics">Connection Diagnostics</a></p>
    """

@app.route('/status')
def status():
    system_status["timestamp"] = datetime.datetime.utcnow().isoformat()
    return jsonify({
        "status": "operational" if all([
            system_status["api_authenticated"],
            system_status["websocket_connected"],
            not system_status["connection_errors"]
        ]) else "degraded",
        "details": system_status
    })

@app.route('/diagnostics')
def diagnostics():
    perform_connection_tests()  # Refresh diagnostics
    return jsonify({
        "connection_tests": {
            "api_connection": system_status["api_authenticated"],
            "websocket_connection": system_status["websocket_connected"],
            "errors": system_status["connection_errors"]
        },
        "suggestions": generate_fix_suggestions()
    })

def generate_fix_suggestions():
    """Generate automatic fix suggestions based on errors"""
    suggestions = []
    
    for error in system_status["connection_errors"]:
        if "SSL" in error:
            suggestions.append("Update SSL certificates or add ssl_verify=False (not recommended)")
        elif "HTTP 403" in error:
            suggestions.append("Check API keys and IP restrictions")
        elif "time out" in error:
            suggestions.append("Check network connection and firewall settings")
        elif "WebSocket" in error:
            suggestions.append("Verify WebSocket URL and network connectivity")
    
    return suggestions if suggestions else ["No issues detected"]

# === Startup Sequence ===
def run_server():
    """Run Flask server in a thread"""
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True)

async def main():
    """Main entry point with proper shutdown handling"""
    try:
        # Start status server
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Run connection tests
        initial_report = perform_connection_tests()
        logging.info("Initial connection tests completed")
        
        # Start trading engine
        if not system_status["connection_errors"]:
            await run_trading_engine()
        else:
            logging.critical("Cannot start trading due to connection errors")
            
    except KeyboardInterrupt:
        logging.info("Shutdown signal received")
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
    finally:
        system_status["server_running"] = False
        logging.info("System shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
