import ccxt
import pandas as pd
import numpy as np
import time
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import os
import pandas_ta as ta

# 🔥 scikit-learn - للتعلم الآلي (Kelly Criterion)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

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

# === الاتصال بمنصة CoinEx ===
try:
    exchange = ccxt.coinex({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
        },
    })
    logging.info("تم تسجيل الدخول بنجاح إلى CoinEx.")
except Exception as e:
    logging.error(f"فشل تسجيل الدخول إلى CoinEx: {e}")
    raise

# === جلب البيانات الحية عبر REST API ===
def fetch_live_data(symbol, timeframe, limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        logging.info(f"[{symbol}] تم جلب البيانات بنجاح.")
        return df
    except Exception as e:
        logging.error(f"[{symbol}] خطأ أثناء جلب البيانات: {e}")
        raise

# === تحديث البيانات بشكل ذكي (جلب الجديد فقط) ===
def fetch_new_data_only(df_old, symbol, timeframe):
    try:
        latest_timestamp = df_old['timestamp'].iloc[-1]
        new_df = fetch_live_data(symbol, timeframe, limit=10)  # جلب آخر 10 شموع فقط
        new_data = new_df[new_df['timestamp'] > latest_timestamp]  # اختيار الشموع الجديدة فقط

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

# === حساب المؤشرات الفنية باستخدام Pandas TA ===
def calculate_indicators(df):
    try:
        # SMA
        df['SMA'] = ta.sma(df['close'], length=20)
        # RSI
        df['RSI'] = ta.rsi(df['close'], length=14)
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd[f'MACD_{12}_{26}_{9}']
        df['MACD_signal'] = macd[f'MACDs_{12}_{26}_{9}']
        # Bollinger Bands
        bollinger = ta.bbands(df['close'], length=20, std=2)
        df['upper_band'] = bollinger['BBU_20_2.0']
        df['middle_band'] = bollinger['BBM_20_2.0']
        df['lower_band'] = bollinger['BBL_20_2.0']
        # ATR
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        # Stochastic Oscillator
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

        # تحديد الإشارة: إذا كان السعر بعد خطوة أعلى، فهي إشارة شراء (1)، وإلا بيع (0)
        df['future_close'] = df['close'].shift(-1)
        df['ml_signal'] = np.where(df['future_close'] > df['close'], 1, 0)

        X = df[features]
        y = df['ml_signal']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # استخدام GridSearchCV للبحث عن أفضل المعلمات
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }

        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"دقة النموذج: {accuracy:.2f}")

        joblib.dump(best_model, 'trading_model.pkl')  # حفظ النموذج

        df['ml_signal'] = best_model.predict(X)
        logging.info("✅ تم توليد الإشارات الذكية باستخدام Random Forest.")
        return df

    except Exception as e:
        logging.error(f"❌ خطأ أثناء توليد الإشارات الذكية: {e}")
        raise

# === حساب VaR ===
def calculate_var(returns, confidence_level=0.95):
    if len(returns) < 10:
        return 0.02  # قيمة افتراضية
    var = -np.percentile(returns, 100 * (1 - confidence_level))
    return var

# === تنفيذ الصفقات الحقيقية عبر API ===
def execute_real_trade(symbol, side, amount):
    try:
        if side == "buy":
            order = exchange.create_market_buy_order(symbol, amount)
            logging.info(f"✅ [BUY] أمر شراء تم تنفيذه: {order}")
        elif side == "sell":
            order = exchange.create_market_sell_order(symbol, amount)
            logging.info(f"✅ [SELL] أمر بيع تم تنفيذه: {order}")
    except Exception as e:
        logging.error(f"❌ خطأ في تنفيذ الأمر: {e}")

# === تنفيذ الصفقات ===
def execute_trades(df, symbol, balance):
    try:
        initial_balance = balance
        position = 0
        trade_count = 0
        max_trades_per_day = 5
        last_trade_time = 0
        cooldown_period = 1800
        consecutive_losses = 0

        for i in range(1, len(df)):
            current_time = time.time()
            if current_time - last_trade_time < cooldown_period or trade_count >= max_trades_per_day:
                continue

            current_price = df['close'].iloc[i]
            atr = df['ATR'].iloc[i]

            # حساب Kelly
            returns = df['close'].pct_change().dropna()
            wins = returns[returns > 0]
            losses = returns[returns < 0]
            win_prob = len(wins) / len(returns)
            avg_win = wins.mean() if not wins.empty else 0.005
            avg_loss = abs(losses.mean()) if not losses.empty else 0.005
            kelly = win_prob - ((1 - win_prob) / (avg_win / avg_loss))
            kelly = max(0.01, min(kelly, 0.2))

            # حساب VaR
            risk_amount = calculate_var(returns)

            # تنفيذ الصفقات بناءً على الإشارة
            if df['ml_signal'].iloc[i-1] == 1 and position == 0:
                amount_to_invest = balance * kelly
                amount = amount_to_invest / current_price
                position = 1
                buy_price = current_price
                stop_loss = buy_price * (1 - risk_amount)
                take_profit = buy_price * (1 + risk_amount * 2)
                execute_real_trade(symbol, "buy", amount)
                last_trade_time = current_time
                trade_count += 1

            elif df['ml_signal'].iloc[i-1] == 0 and position == 1:
                execute_real_trade(symbol, "sell", amount)
                sell_price = current_price
                profit = (sell_price - buy_price) / buy_price * 100
                balance += profit * amount * buy_price / 100
                position = 0
                trade_count += 1
                if profit < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

            # Trailing Stop
            if position == 1:
                trailing_stop = current_price * 0.98
                if current_price <= stop_loss or current_price <= trailing_stop:
                    execute_real_trade(symbol, "sell", amount)
                    sell_price = current_price
                    profit = (sell_price - buy_price) / buy_price * 100
                    balance += profit * amount * buy_price / 100
                    position = 0
                    if profit < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0

            if consecutive_losses >= 3:
                logging.info("🛑 تم الوصول إلى الحد الأقصى للخسائر المتتالية.")
                break

        logging.info(f"💰 رصيدك النهائي: {balance:.2f} دولار")
        return balance

    except Exception as e:
        logging.error(f"❌ خطأ في تنفيذ الصفقات: {e}")
        return balance

# === تنفيذ الصفقات على عدة عملات (Diversification) ===
def diversified_trading():
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    total_balance = 100  # رأس المال الإجمالي
    per_asset_balance = total_balance / len(symbols)
    results = {}

    dfs = {}
    models_ready = {}

    # جلب البيانات الأولية لكل عملة
    for symbol in symbols:
        df = fetch_live_data(symbol, '5m', limit=100)
        df = calculate_indicators(df)
        df = generate_ml_signals(df)
        dfs[symbol] = df
        models_ready[symbol] = True

    while True:
        for symbol in symbols:
            df = fetch_new_data_only(dfs[symbol], symbol, '5m')
            if len(df) > len(dfs[symbol]):
                df = calculate_indicators(df)
                df = generate_ml_signals(df)
                dfs[symbol] = df
                logging.info(f"[{symbol}] تم تحديث البيانات وحساب المؤشرات والإشارات.")

            final_balance = execute_trades(df, symbol, per_asset_balance)
            results[symbol] = final_balance

        overall_return = sum(results.values())
        logging.info(f"📊 العوائد الإجمالية بعد التنويع: {overall_return:.2f} دولار")
        time.sleep(60)  # الانتظار دقيقة واحدة قبل المحاولة التالية

# === الحلقة الرئيسية للبرنامج ===
def main_loop():
    while True:
        diversified_trading()

# === تشغيل البرنامج ===
if __name__ == "__main__":
    logging.info("🚀 البوت قيد التشغيل...")
    main_loop()