# Silent1_final.py - Trading Bot with Advanced Connection Diagnostics & Trailing Stop
# Author: AI Assistant
# Version: 4.1.0 (Simplified Project Structure)

import logging
import os
import asyncio
import datetime
import json
import threading

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import websockets
from dotenv import load_dotenv
from flask import Flask, jsonify
from logging.handlers import RotatingFileHandler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Import from the local SDK file in the same directory ---
from coinex_sdk import CoinEx

# --- Global Status & State Variables ---
system_status = { "api_authenticated": False, "websocket_connected": False, "trading_active": False, "server_running": True, "last_error": None, "connection_errors": [] }
active_positions = {}

# === Load Environment Variables & Initialize SDK ===
load_dotenv()
API_KEY = os.getenv('COINEX_API_KEY')
API_SECRET = os.getenv('COINEX_API_SECRET')
if not API_KEY or not API_SECRET:
    raise ValueError("CRITICAL: API_KEY or API_SECRET not found in .env file.")

coinex_client = CoinEx(API_KEY, API_SECRET)

# === Enhanced Logging Setup ===
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_handler = RotatingFileHandler('bot_diagnostics.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
log_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[log_handler, console_handler])

# === Connection Diagnostics (Using SDK) ===
async def perform_connection_tests():
    system_status["connection_errors"] = []
    
    # Test 1: API Connection
    logging.info("Testing API Connection...")
    ping_res = await asyncio.to_thread(coinex_client.market.ping)
    if ping_res.get('code') != 0:
        system_status["connection_errors"].append(f"API Connection Failed: {ping_res.get('message', 'Unknown Error')}")
    
    # Test 2: API Authentication
    if not system_status["connection_errors"]:
        logging.info("Testing API Authentication...")
        auth_res = await asyncio.to_thread(coinex_client.account.get_account_info)
        if auth_res.get('code') != 0:
            system_status["connection_errors"].append(f"API Authentication Failed: {auth_res.get('message')}")
    
    system_status["api_authenticated"] = not any("Authentication" in s for s in system_status["connection_errors"])
    return {"success": not system_status["connection_errors"], "errors": system_status["connection_errors"]}

# === Data & Order Functions (Using SDK) ===
def fetch_ohlcv(symbol, timeframe='5m', limit=100):
    params = {'market': symbol.replace('/', ''), 'interval': timeframe, 'limit': limit}
    data = coinex_client.request('GET', '/market/kline', params)
    if data.get('code') == 0:
        df = pd.DataFrame(data['data'], columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'amount'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
    raise ConnectionError(f"Failed to fetch OHLCV: {data.get('message')}")

async def fetch_balance():
    balance_data = await asyncio.to_thread(coinex_client.account.get_account_info)
    if balance_data.get('code') == 0:
        usdt_balance = next((float(asset.get('available', 0)) for asset in balance_data['data'] if asset['ccy'] == 'USDT'), 0.0)
        return {'USDT': {'free': usdt_balance}}
    raise ConnectionError(f"Failed to fetch balance: {balance_data.get('message')}")

async def execute_real_trade(symbol, side, amount):
    params = {"market": symbol.replace('/', ''), "side": side, "type": "market"}
    if side.lower() == 'buy': params["amount"] = str(round(amount, 8))
    else: params["quantity"] = str(round(amount, 8))
    order_result = await asyncio.to_thread(coinex_client.request, 'POST', '/spot/order', params, need_sign=True)
    if order_result.get('code') == 0:
        logging.info(f"Successfully placed {side} order for {symbol}: {order_result['data']}")
        return order_result['data']
    logging.error(f"Failed to place {side} order for {symbol}: {order_result.get('message')}")
    return None

# === TA, ML, and Logic Functions (No changes) ===
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

async def trade_and_manage_position(df, symbol, usdt_per_trade):
    global active_positions
    if df.empty or 'ml_signal' not in df.columns: return
    last_signal, current_price = df['ml_signal'].iloc[-1], df['close'].iloc[-1]
    
    if symbol in active_positions:
        pos = active_positions[symbol]
        if last_signal == 0 or current_price <= pos['trailing_stop_price']:
            if await execute_real_trade(symbol, "sell", pos['quantity']): del active_positions[symbol]
            return
        new_stop = current_price * 0.98
        if new_stop > pos['trailing_stop_price']: pos['trailing_stop_price'] = new_stop
    elif last_signal == 1:
        order_result = await execute_real_trade(symbol, "buy", usdt_per_trade)
        if order_result:
            active_positions[symbol] = {
                'quantity': usdt_per_trade / current_price,
                'trailing_stop_price': current_price * 0.98
            }

# === Main Trading Loop (No changes) ===
async def run_trading_engine():
    try:
        logging.info("--- Starting Trading Engine ---")
        await perform_connection_tests()
        if system_status["connection_errors"]:
            raise ConnectionError(f"Initial diagnostics failed: {system_status['connection_errors']}")
        
        symbols = ['BTC/USDT', 'ETH/USDT']
        
        balance_info = await fetch_balance()
        total_balance = balance_info.get('USDT', {}).get('free', 0)
        if total_balance < 20: raise ValueError(f"Insufficient balance ({total_balance} USDT) to start.")
        
        usdt_per_trade = (total_balance / len(symbols)) * 0.1
        
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
                    await trade_and_manage_position(dfs[symbol], symbol, usdt_per_trade)
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

# === Web Interface & Startup (No changes) ===
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
    asyncio.run(main())
