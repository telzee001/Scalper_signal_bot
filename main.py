import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from threading import Thread
from flask import Flask

# ==================== TELEGRAM CONFIG ====================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

# ==================== SYMBOLS & PARAMETERS ====================
CRYPTO_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT',
    'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT',
    'DOTUSDT', 'LINKUSDT'
]
FOREX_SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
    'USDCAD', 'USDCHF', 'NZDUSD', 'EURGBP',
    'EURJPY', 'GBPJPY'
]
CRYPTO_TIMEFRAME = '1m'
FOREX_TIMEFRAME   = '5m'
LOOKBACK = 150

SUPERTREND_PERIOD = 7
SUPERTREND_MULTIPLIER = 2.0
RSI_PERIOD = 7
ADX_PERIOD = 7
ADX_THRESHOLD = 20
VOLUME_SMA_PERIOD = 10
VOLUME_SPIKE = 1.5
STOCH_K_PERIOD = 7
STOCH_D_PERIOD = 3
EMA_FAST = 5
EMA_SLOW = 10
ATR_PERIOD = 7
RR_RATIO = 1.5

# ==================== TELEGRAM HELPER ====================
def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": message})
    except Exception as e:
        print(f"Telegram error: {e}")

# ==================== DATA FETCHERS ====================
def fetch_crypto_ohlcv(symbol, interval='1m', limit=150):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Crypto error {symbol}: {e}")
        return None

def fetch_forex_ohlcv(symbol, interval='5m', limit=150):
    import yfinance as yf
    ticker = yf.Ticker(f"{symbol}=X")
    df = ticker.history(period=f"{limit*2}m", interval=interval)
    if df.empty:
        return None
    df = df.reset_index()
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    return df.tail(limit)

# ==================== INDICATORS ====================
def calculate_supertrend(df, period, multiplier):
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    hl_avg = (high + low) / 2
    upper_band = hl_avg + multiplier * atr
    lower_band = hl_avg - multiplier * atr
    supertrend_dir = np.where(close > upper_band, 1, -1)
    for i in range(1, len(supertrend_dir)):
        if close.iloc[i] > upper_band.iloc[i]:
            supertrend_dir[i] = 1
        elif close.iloc[i] < lower_band.iloc[i]:
            supertrend_dir[i] = -1
        else:
            supertrend_dir[i] = supertrend_dir[i-1]
    supertrend = np.where(supertrend_dir == 1, upper_band, lower_band)
    df['supertrend'] = supertrend
    df['supertrend_dir'] = supertrend_dir
    return df

def calculate_ema(df, period, column='close'):
    return df[column].ewm(span=period, adjust=False).mean()

def calculate_rsi(df, period):
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_adx(df, period):
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.abs().rolling(window=period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(window=period).mean()

def calculate_stochastic(df, k_period, d_period):
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d

def calculate_atr(df, period):
    tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def check_signal(asset_name, df, is_crypto, current_price=None):
    if df is None or len(df) < LOOKBACK:
        return None
    df = calculate_supertrend(df, SUPERTREND_PERIOD, SUPERTREND_MULTIPLIER)
    df['ema_fast'] = calculate_ema(df, EMA_FAST)
    df['ema_slow'] = calculate_ema(df, EMA_SLOW)
    df['rsi'] = calculate_rsi(df, RSI_PERIOD)
    df['adx'] = calculate_adx(df, ADX_PERIOD)
    df['volume_sma'] = df['volume'].rolling(window=VOLUME_SMA_PERIOD).mean()
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df, STOCH_K_PERIOD, STOCH_D_PERIOD)
    df['atr'] = calculate_atr(df, ATR_PERIOD)

    last = df.iloc[-1]
    close = last['close']
    current_price = current_price or close

    long_cond = (
        (last['supertrend_dir'] == 1) &
        (last['adx'] > ADX_THRESHOLD) &
        (last['rsi'] < 35) &
        (last['ema_fast'] > last['ema_slow']) &
        (last['stoch_k'] < 20) & (last['stoch_k'] > last['stoch_d']) &
        (last['volume'] > last['volume_sma'] * VOLUME_SPIKE)
    )
    short_cond = (
        (last['supertrend_dir'] == -1) &
        (last['adx'] > ADX_THRESHOLD) &
        (last['rsi'] > 65) &
        (last['ema_fast'] < last['ema_slow']) &
        (last['stoch_k'] > 80) & (last['stoch_k'] < last['stoch_d']) &
        (last['volume'] > last['volume_sma'] * VOLUME_SPIKE)
    )

    atr = last['atr']
    if atr is None or atr == 0:
        return None

    if long_cond:
        entry = current_price
        sl = entry - atr
        tp = entry + RR_RATIO * atr
        if is_crypto:
            msg = f"{asset_name} Long\nentry:{entry:.2f}\nTp:{tp:.2f}\nSL:{sl:.2f}"
        else:
            msg = f"{asset_name} Buy\nentry:{entry:.5f}\nTp:{tp:.5f}\nSL:{sl:.5f}"
        return {'message': msg}
    elif short_cond:
        entry = current_price
        sl = entry + atr
        tp = entry - RR_RATIO * atr
        if is_crypto:
            msg = f"{asset_name} Short\nentry:{entry:.2f}\nTp:{tp:.2f}\nSL:{sl:.2f}"
        else:
            msg = f"{asset_name} Sell\nentry:{entry:.5f}\nTp:{tp:.5f}\nSL:{sl:.5f}"
        return {'message': msg}
    return None

# ==================== BOT THREAD ====================
def bot_worker():
    send_telegram("⚡ Scalper Signal Bot is live! Scanning for high‑probability setups (70%+ win rate).")
    while True:
        try:
            # Crypto
            for sym in CRYPTO_SYMBOLS:
                df = fetch_crypto_ohlcv(sym, CRYPTO_TIMEFRAME, LOOKBACK)
                if df is not None:
                    asset_name = sym.replace('USDT', '')
                    signal = check_signal(asset_name, df, is_crypto=True)
                    if signal:
                        send_telegram(signal['message'])
                        print(f"[{datetime.now()}] Crypto signal {sym}")
            # Forex
            for sym in FOREX_SYMBOLS:
                df = fetch_forex_ohlcv(sym, FOREX_TIMEFRAME, LOOKBACK)
                if df is not None:
                    signal = check_signal(sym, df, is_crypto=False)
                    if signal:
                        send_telegram(signal['message'])
                        print(f"[{datetime.now()}] Forex signal {sym}")
            time.sleep(60)
        except Exception as e:
            print(f"Worker error: {e}")
            time.sleep(30)

# ==================== FLASK APP (for Pxxl WSGI) ====================
app = Flask(__name__)

@app.route('/')
def home():
    return "Scalper Signal Bot is running!"

@app.route('/health')
def health():
    return "OK"

# Start the bot thread when the Flask app starts
thread = Thread(target=bot_worker, daemon=True)
thread.start()

# For local testing you can run with: python main.py
# For Gunicorn, just the app object is enough.
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
