# Silent1_fixed.py - Trading Bot for CoinEx API v2
# Author: AI Assistant
# Date: 2025-04-05

import requests
import json
import time
import hmac
import hashlib
import pandas as pd
import asyncio
import websockets
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import os
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask
import threading

# === Load Environment Variables ===
load_dotenv()
API_KEY = os.getenv('COINEX_API_KEY')
API_SECRET = os.getenv('COINEX_API_SECRET')

# === Logging Setup ===
log_handler = RotatingFileHandler('bot_logs.log', maxBytes=1024 * 1024, backupCount=5)
logging.basicConfig(
    handlers=[log_handler],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === URLs ===
REST_URL = "https://api.coinex.com/v2 "
SPOT_WS_URL = "wss://socket.coinex.com/v2/spot"

# === Sign Request Function ===
def sign_request(params, secret):
    sorted_params = '&'.join([f"{k}={params[k]}" for k in sorted(params)])
    signature = hmac.new(secret.encode('utf-8'), sorted_params.encode('utf-8'), hashlib.sha256).hexdigest()
    return signature

# === Private API Call Function ===
def private_api_call(endpoint, method="GET", params=None):
    url = f"{REST_URL}/{endpoint}"
    timestamp = int(time.time() * 1000)
    headers = {
        'Content-Type': 'application/json',
        'X-COINEX-APIKEY': API_KEY,
    }

    if params is None:
        params = {}

    params.update({
        'timestamp': timestamp,
        'signature_type': 2,
        'recv_window': 5000
    })

    for attempt in range(3):  # Retry up to 3 times
        try:
            params['sign'] = sign_request(params, API_SECRET)
            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == "POST":
                response = requests.post(url, data=json.dumps(params), headers=headers, timeout=10)
            else:
                raise ValueError("Unsupported HTTP method")

            response.raise_for_status()

            try:
                data = response.json()
            except json.JSONDecodeError:
                logging.error(f"❌ Invalid JSON response: {response.text[:200]}...")
                raise

            if data.get('code') != 0:
                error_msg = data.get('message', 'Unknown error')
                logging.error(f"❌ API Error: {error_msg}")
                raise Exception(error_msg)

            return data

        except requests.exceptions.RequestException as e:
            logging.warning(f"⚠️ Network error ({attempt + 1}/3): {e}")
            time.sleep(2)

    logging.critical("❌ Failed after multiple attempts.")
    raise Exception("Failed to complete private API request.")

# === Fetch Market Info ===
def get_market_info(markets=None):
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
                logging.error(f"❌ Invalid JSON response: {response.text[:200]}...")
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
                logging.info(f"✅ Market info fetched successfully: {list(market_data.keys())}")
                return market_data
            else:
                error_msg = data.get("message", "Unknown error")
                logging.error(f"❌ Failed to fetch market info: {error_msg}")
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            logging.warning(f"⚠️ Network error ({attempt + 1}/3): {e}")
            time.sleep(2)
    return None

# === Fetch Balance ===
def fetch_balance():
    result = private_api_call("spot/balance", method="GET")
    balances = result.get('data', {})
    usdt_free = float(balances.get('USDT', {}).get('available', 0))
    return {'USDT': {'free': usdt_free}}

# === Fetch OHLCV Data ===
def fetch_ohlcv(symbol, timeframe='5m', limit=100):
    url = f"{REST_URL}/spot/kline-data"
    params = {
        'market': symbol.replace('/', ''),
        'type': timeframe,
        'limit': limit
    }

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            try:
                data = response.json()
            except json.JSONDecodeError:
                logging.warning(f"[{symbol}] ⚠️ Invalid JSON response: {response.text[:200]}...")
                time.sleep(2)
                continue

            if data.get('code') != 0:
                error_msg = data.get('message', 'Unknown error')
                logging.error(f"[{symbol}] ❌ Failed to fetch OHLCV: {error_msg}")
                time.sleep(2)
                continue

            df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            logging.info(f"[{symbol}] ✅ OHLCV data fetched via REST.")
            return df

        except requests.exceptions.RequestException as e:
            logging.warning(f"[{symbol}] ⚠️ Request failed ({attempt + 1}/3): {e}")
            time.sleep(2)

    logging.critical(f"[{symbol}] ❌ Failed to fetch OHLCV after several attempts.")
    return pd.DataFrame()

# === Place Order ===
def create_market_order(symbol, side, amount):
    endpoint = "spot/order/place"
    params = {
        "market": symbol.replace('/', ''),
        "side": side,
        "amount": amount,
        "order_type": "market"
    }
    result = private_api_call(endpoint, method="POST", params=params)
    logging.info(f"✅ [{symbol}] Order executed: {result}")
    return result

def create_market_buy_order(symbol, amount):
    return create_market_order(symbol, 'buy', amount)

def create_market_sell_order(symbol, amount):
    return create_market_order(symbol, 'sell', amount)

# === WebSocket Update ===
async def ws_update_data(symbol):
    uri = SPOT_WS_URL
    symbol_clean = symbol.replace('/', '')
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                subscribe_msg = {
                    "method": "state",
                    "params": [symbol_clean, '5m', 100],
                    "id": int(time.time())
                }
                await websocket.send(json.dumps(subscribe_msg))
                logging.info(f"[{symbol}] Subscribed to WebSocket")

                while True:
                    message = await websocket.recv()
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
                        logging.warning(f"[{symbol}] ⚠️ Invalid WebSocket JSON: {message[:200]}...")

                    await asyncio.sleep(1)

        except (websockets.ConnectionClosed, requests.exceptions.RequestException) as e:
            logging.error(f"[{symbol}] ❌ Connection lost. Reconnecting...")
            await asyncio.sleep(10)

# === Get Real Balance from Exchange ===
def get_real_balance():
    try:
        balance_info = fetch_balance()
        usdt_balance = balance_info.get('USDT', {}).get('free', 0)
        if usdt_balance <= 0:
            logging.warning("⚠️ No free USDT balance. Using default $100.")
            return 100.0
        logging.info(f"💰 Free balance: {usdt_balance:.2f} USDT")
        return float(usdt_balance)
    except Exception as e:
        logging.error(f"❌ Error fetching balance: {e}")
        return 100.0

# === Technical Indicators ===
def calculate_indicators(df):
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
        logging.info("✅ Indicators calculated")
        return df
    except Exception as e:
        logging.error(f"❌ Error calculating indicators: {e}")
        raise

# === Generate ML Signals ===
def generate_ml_signals(df):
    try:
        features = ['SMA', 'RSI', 'MACD', 'stoch_k', 'ATR', 'upper_band', 'lower_band']
        df = df.dropna(subset=features)
        df['future_close'] = df['close'].shift(-1)
        df['ml_signal'] = np.where(df['future_close'] > df['close'], 1, 0)

        X = df[features]
        y = df['ml_signal']

        model_path = 'trading_model.pkl'

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logging.info("🔄 Loaded existing model.")
        else:
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
            logging.info(f"🏆 Model Accuracy: {accuracy:.2f}")

        if not os.path.exists(model_path) or accuracy > get_previous_model_accuracy():
            joblib.dump(model, model_path)
            save_model_accuracy(accuracy)
            logging.info("🆕 New model saved.")

        df['ml_signal'] = model.predict(X)
        logging.info("✅ ML signals generated.")
        return df
    except Exception as e:
        logging.error(f"❌ Error generating ML signals: {e}")
        raise

# === Model Accuracy Helper Functions ===
def get_previous_model_accuracy():
    acc_file = 'model_accuracy.txt'
    if os.path.exists(acc_file):
        with open(acc_file, 'r') as f:
            try:
                return float(f.read().strip())
            except:
                return 0.0
    return 0.0

def save_model_accuracy(accuracy):
    with open('model_accuracy.txt', 'w') as f:
        f.write(f"{accuracy:.4f}")

# === VaR Calculation ===
def calculate_var(returns, window=20, confidence_level=0.95):
    if len(returns) < window:
        return 0.02
    recent_returns = returns[-window:]
    var = -np.percentile(recent_returns, 100 * (1 - confidence_level))
    return abs(var)

# === Execute Real Trade via API ===
def execute_real_trade(symbol, side, amount):
    try:
        if amount < 0.0001:
            logging.warning(f"⚠️ Insufficient amount: {amount:.8f}")
            return None
        if side == "buy":
            order = create_market_buy_order(symbol, amount)
            logging.info(f"✅ [BUY] Order executed: {order}")
        elif side == "sell":
            order = create_market_sell_order(symbol, amount)
            logging.info(f"✅ [SELL] Order executed: {order}")
        return order
    except Exception as e:
        logging.error(f"❌ Error executing trade: {e}")
        return None

# === Smart Data Update ===
def fetch_new_data_only(df_old, symbol, timeframe):
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

# === Trading Logic per Asset ===
def execute_trades(df, symbol, per_asset_balance):
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
                logging.warning("⚠️ No ML signal yet. Skipping trades.")
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
                logging.info("🛑 Max consecutive losses reached.")
                break

        logging.info(f"📊 [{symbol}] Final balance: {per_asset_balance:.2f} USD")
        return per_asset_balance
    except Exception as e:
        logging.error(f"❌ Error in trading logic: {e}")
        return per_asset_balance

# === Run Trading Engine on Multiple Assets ===
async def run_trading_engine():
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    total_balance = get_real_balance()
    investment_capital = total_balance * 0.2
    per_asset_balance = investment_capital / len(symbols)
    dfs = {}

    markets_list = [s.replace("/", "") for s in symbols]
    market_info = get_market_info(markets_list)

    if not market_info:
        logging.error("❌ Market info not found. Exiting.")
        return

    for symbol in symbols:
        market_key = symbol.replace("/", "")
        if not market_info[market_key].get("is_api_trading_available", False):
            logging.error(f"❌ API trading disabled for {market_key}.")
            continue

        df = fetch_ohlcv(symbol, '5m', limit=100)
        df = calculate_indicators(df)
        df = generate_ml_signals(df)
        dfs[symbol] = df
        logging.info(f"[{symbol}] Data and signals prepared.")

    while True:
        for symbol in symbols:
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
        logging.info(f"📈 Total return after diversification: {overall_return:.2f} USD")
        await asyncio.sleep(60)

# === Main Loop ===
async def main_loop():
    try:
        await run_trading_engine()
    except KeyboardInterrupt:
        logging.info("🛑 Bot stopped manually.")

# === Web Server (Flask) ===
app = Flask(__name__)
@app.route('/')
def home():
    return "Trading Bot is Running 🚀"

def run_server():
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)

if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    asyncio.run(main_loop())
