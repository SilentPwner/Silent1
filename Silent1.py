import ccxt
import pandas as pd
import numpy as np
import time
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import os
import asyncio
import websockets
import json
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
from flask import Flask
import threading

# === تحميل المتغيرات البيئية ===
load_dotenv()
API_KEY = os.getenv('COINEX_API_KEY')
API_SECRET = os.getenv('COINEX_API_SECRET')

# === إعداد تسجيل الأخطاء والسجلات باستخدام RotatingFileHandler ===
log_handler = RotatingFileHandler('bot_logs.log', maxBytes=1024*1024, backupCount=5)
logging.basicConfig(
    handlers=[log_handler],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# === تعريف الاتصال الصحيح بمنصة CoinEx ===
try:
    exchange = ccxt.coinex({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'fetchCurrencies': False,
            'adjustForTimeDifference': True,
        },
        'urls': {
            'api': {
                'public': 'https://api.coinex.com/v2/spot', 
                'private': 'https://api.coinex.com/v2/spot', 
            },
            'websocket': 'wss://socket.coinex.com/v2/spot'
        },
        'api': {
            'public': {
                'get': [
                    'common/asset/config',
                    'market/ticker/all',
                    'market/order-book',
                    'trades',
                    'kline-data',
                ],
                'post': [],
            },
            'private': {
                'get': [
                    'balance',
                    'order',
                    'order/history',
                    'order/deals',
                ],
                'post': [
                    'order/place',
                    'order/cancel',
                ],
            },
        }
    })
    logging.info("✅ تم تسجيل الدخول بنجاح إلى CoinEx.")
except Exception as e:
    logging.error(f"❌ فشل تسجيل الدخول إلى CoinEx: {e}")
    raise

# === جلب الرصيد الحقيقي من المنصة ===
def get_real_balance():
    try:
        balance_info = exchange.fetch_balance()
        usdt_balance = balance_info.get('USDT', {}).get('free', 0)
        if usdt_balance <= 0:
            logging.warning("⚠️ لا يوجد رصيد USDT حر. سيتم استخدام رصيد افتراضي قدره $100.")
            return 100.0
        logging.info(f"💰 الرصيد الحر في الحساب: {usdt_balance:.2f} USDT")
        return float(usdt_balance)
    except Exception as e:
        logging.error(f"❌ خطأ أثناء جلب الرصيد من المنصة: {e}")
        return 100.0  # قيمة افتراضية إن فشلت العملية

# === حساب المؤشرات الفنية باستخدام Pandas TA ===
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
        logging.info("✅ المؤشرات الفنية محسوبة")
        return df
    except Exception as e:
        logging.error(f"❌ خطأ في حساب المؤشرات: {e}")
        raise

# === توليد إشارات باستخدام scikit-learn ===
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
            logging.info("🔄 تم تحميل نموذج ML سابق.")
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
            logging.info(f"🏆 دقة النموذج: {accuracy:.2f}")
        if not os.path.exists(model_path) or accuracy > get_previous_model_accuracy():
            joblib.dump(model, model_path)
            save_model_accuracy(accuracy)
            logging.info("🆕 تم حفظ نموذج ML جديد.")
        df['ml_signal'] = model.predict(X)
        logging.info("✅ تم توليد الإشارات الذكية باستخدام Random Forest.")
        return df
    except Exception as e:
        logging.error(f"❌ خطأ أثناء توليد الإشارات الذكية: {e}")
        raise

# === قراءة دقة النموذج السابق من ملف ===
def get_previous_model_accuracy():
    acc_file = 'model_accuracy.txt'
    if os.path.exists(acc_file):
        with open(acc_file, 'r') as f:
            try:
                return float(f.read().strip())
            except:
                return 0.0
    return 0.0

# === حفظ دقة النموذج الحالي في ملف ===
def save_model_accuracy(accuracy):
    with open('model_accuracy.txt', 'w') as f:
        f.write(f"{accuracy:.4f}")

# === حساب VaR باستخدام نافذة زمنية ===
def calculate_var(returns, window=20, confidence_level=0.95):
    if len(returns) < window:
        return 0.02  # قيمة افتراضية إن كانت البيانات غير كافية
    recent_returns = returns[-window:]  # استخدام آخر N عناصر فقط
    var = -np.percentile(recent_returns, 100 * (1 - confidence_level))
    return abs(var)

# === تنفيذ الصفقات الحقيقية عبر API ===
def execute_real_trade(symbol, side, amount):
    try:
        if amount < 0.0001:
            logging.warning(f"⚠️ الكمية غير كافية للصفقة: {amount:.8f}")
            return None
        if side == "buy":
            order = exchange.create_market_buy_order(symbol, amount)
            logging.info(f"✅ [BUY] أمر شراء تم تنفيذه: {order}")
        elif side == "sell":
            order = exchange.create_market_sell_order(symbol, amount)
            logging.info(f"✅ [SELL] أمر بيع تم تنفيذه: {order}")
        return order
    except Exception as e:
        logging.error(f"❌ خطأ في تنفيذ الأمر: {e}")
        return None

# === تحديث البيانات عبر WebSocket ===
async def ws_update_data(symbol):
    uri = 'wss://socket.coinex.com/v2/spot'
    symbol_clean = symbol.replace('/', '')
    async with websockets.connect(uri) as websocket:
        subscribe_msg = {
            "method": "state",
            "params": [symbol_clean, '5m', 100],
            "id": int(time.time())
        }
        await websocket.send(json.dumps(subscribe_msg))
        logging.info(f"[{symbol}] اشتراك في WebSocket")
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                if 'data' in data:
                    ohlcv = data['data']
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = calculate_indicators(df)
                    df = generate_ml_signals(df)
                    yield df
                await asyncio.sleep(1)
            except websockets.ConnectionClosed:
                logging.error(f"[{symbol}] ❌ انقطع الاتصال بـ WebSocket. إعادة الاتصال...")
                await asyncio.sleep(10)
                await websocket.close()
                await websocket.connect()

# === جلب البيانات الحية عبر REST API ===
def fetch_live_data(symbol, timeframe, limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"[{symbol}] تم جلب البيانات عبر REST API.")
        return df
    except Exception as e:
        logging.error(f"[{symbol}] خطأ أثناء جلب البيانات: {e}")
        raise

# === تحديث البيانات بشكل ذكي (جلب الجديد فقط) ===
def fetch_new_data_only(df_old, symbol, timeframe):
    try:
        latest_timestamp = df_old['timestamp'].iloc[-1]
        new_df = fetch_live_data(symbol, timeframe, limit=10)
        new_data = new_df[new_df['timestamp'] > latest_timestamp]
        if not new_data.empty:
            updated_df = pd.concat([df_old, new_data], ignore_index=True)
            logging.info(f"[{symbol}] تم تحديث البيانات. عدد الشموع الجديدة: {len(new_data)}")
            return updated_df
        else:
            logging.info(f"[{symbol}] لا توجد بيانات جديدة بعد آخر تحقق.")
            return df_old
    except Exception as e:
        logging.error(f"[{symbol}] خطأ أثناء تحديث البيانات: {e}")
        return df_old

# === تنفيذ الصفقات لكل عملة ===
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
                logging.warning("⚠️ لا توجد إشارة ذكية حتى الآن. سيتم تخطي الصفقات.")
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
            if df['ml_signal'].iloc[i-1] == 1 and position == 0:
                amount_to_invest = per_asset_balance * kelly
                amount = amount_to_invest / current_price
                buy_price = current_price
                stop_loss = buy_price * (1 - risk_amount)
                take_profit = buy_price * (1 + risk_amount * 2)
                execute_real_trade(symbol, "buy", amount)
                position = 1
                last_trade_time = current_time
                trade_count += 1
            elif df['ml_signal'].iloc[i-1] == 0 and position == 1:
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
                logging.info("🛑 تم الوصول إلى الحد الأقصى للخسائر المتتالية.")
                break
        logging.info(f"📊 [{symbol}] رصيد هذه العملة: {per_asset_balance:.2f} دولار")
        return per_asset_balance
    except Exception as e:
        logging.error(f"❌ خطأ في تنفيذ الصفقات: {e}")
        return per_asset_balance

# === تنفيذ الصفقات على عدة عملات (Diversification) ===
async def run_trading_engine():
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    total_balance = get_real_balance()
    investment_capital = total_balance * 0.2
    per_asset_balance = investment_capital / len(symbols)
    dfs = {}
    for symbol in symbols:
        df = fetch_live_data(symbol, '5m', limit=100)
        df = calculate_indicators(df)
        df = generate_ml_signals(df)
        dfs[symbol] = df
        logging.info(f"[{symbol}] تم تجهيز البيانات والإشارات الذكية.")
    while True:
        for symbol in symbols:
            df = dfs[symbol]
            df = fetch_new_data_only(df, symbol, '5m')
            if len(df) > len(dfs[symbol]):
                df = calculate_indicators(df)
                df = generate_ml_signals(df)
                dfs[symbol] = df
                logging.info(f"[{symbol}] تم تحديث البيانات والإشارات الذكية.")
            final_balance = execute_trades(df, symbol, per_asset_balance)
            logging.info(f"[{symbol}] العائد: {final_balance:.2f} دولار")
        overall_return = sum([final_balance for _, final_balance in dfs.items()])
        logging.info(f"📈 العوائد الإجمالية بعد التنويع: {overall_return:.2f} دولار")
        await asyncio.sleep(60)

# === الحلقة الرئيسية للبرنامج ===
async def main_loop():
    try:
        await run_trading_engine()
    except KeyboardInterrupt:
        logging.info("🛑 البوت توقف يدويًا.")

# =====================================================================================
# ✅ START WEB SERVER HERE (Using Flask)
# =====================================================================================
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
