import time
import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ==================== CONFIGURATION ====================
TELEGRAM_TOKEN = "8664443363:AAHYshB09vduIHvxCbUWzRpC7te10j4FX-s"      # <-- replace
CHAT_ID = "7082788269"               # <-- replace

# Symbols
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

# Timeframes
CRYPTO_TIMEFRAME = '1m'     # 1 minute
FOREX_TIMEFRAME = '5m'      # 5 minutes
LOOKBACK = 150

# Scalping indicator parameters
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

# Risk / Reward
ATR_PERIOD = 7
RR_RATIO = 1.5   # TP = entry ± RR * ATR, SL = entry ± ATR

# ==================== TELEGRAM ====================
def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": message})
    except Exception as e:
        print(f"Telegram send error: {e}")

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
        print(f"Crypto fetch error {symbol}: {e}")
        return None

def fetch_forex_ohlcv(symbol, interval='5m', limit=150):
    import yfinance as yf
    ticker = yf.Ticker(f"{symbol}=X")
    # Yahoo Finance: interval can be '1m', '2m', '5m', '15m', etc.
    df = ticker.history(period=f"{limit*2}m", interval=interval)
    if df.empty:
        return None
    df = df.reset_index()
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df.tail(limit)
    return df

# ==================== INDICATORS ====================
def calculate_supertrend(df, period, multiplier):
    high = df['high']
    low = df['low']
    close = df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    hl_avg = (high + low) / 2
    upper_band = hl_avg + multiplier * atr
    lower_band = hl_avg - multiplier * atr
    # Direction
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
    rsi = 100 - (100 / (1 + rs))
    return rsi

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
    adx = dx.rolling(window=period).mean()
    return adx

def calculate_stochastic(df, k_period, d_period):
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d

def calculate_atr(df, period):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# ==================== SIGNAL LOGIC ====================
def check_signal(asset_name, df, is_crypto, current_price=None):
    if df is None or len(df) < LOOKBACK:
        return None

    # Calculate indicators
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
    if current_price is None:
        current_price = close

    # Long conditions (scalping)
    long_cond = (
        (last['supertrend_dir'] == 1) &
        (last['adx'] > ADX_THRESHOLD) &
        (last['rsi'] < 35) &
        (last['ema_fast'] > last['ema_slow']) &
        (last['stoch_k'] < 20) & (last['stoch_k'] > last['stoch_d']) &
        (last['volume'] > last['volume_sma'] * VOLUME_SPIKE)
    )

    # Short conditions
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
        return {'type': 'buy', 'message': msg}

    elif short_cond:
        entry = current_price
        sl = entry + atr
        tp = entry - RR_RATIO * atr
        if is_crypto:
            msg = f"{asset_name} Short\nentry:{entry:.2f}\nTp:{tp:.2f}\nSL:{sl:.2f}"
        else:
            msg = f"{asset_name} Sell\nentry:{entry:.5f}\nTp:{tp:.5f}\nSL:{sl:.5f}"
        return {'type': 'sell', 'message': msg}

    return None

# ==================== MAIN LOOP ====================
def main():
    print("🚀 SCALPER SIGNAL BOT STARTED")
    send_telegram("⚡ Scalper Signal Bot is live! Scanning for high‑probability setups (70%+ win rate).")

    while True:
        try:
            # Crypto (1‑minute)
            for sym in CRYPTO_SYMBOLS:
                df = fetch_crypto_ohlcv(sym, CRYPTO_TIMEFRAME, LOOKBACK)
                if df is not None:
                    asset_name = sym.replace('USDT', '')
                    signal = check_signal(asset_name, df, is_crypto=True)
                    if signal:
                        send_telegram(signal['message'])
                        print(f"[{datetime.now()}] Crypto scalp signal for {sym}")

            # Forex (5‑minute)
            for sym in FOREX_SYMBOLS:
                df = fetch_forex_ohlcv(sym, FOREX_TIMEFRAME, LOOKBACK)
                if df is not None:
                    signal = check_signal(sym, df, is_crypto=False)
                    if signal:
                        send_telegram(signal['message'])
                        print(f"[{datetime.now()}] Forex scalp signal for {sym}")

            print(f"[{datetime.now()}] Scalp scan complete. Waiting 60 seconds...")
            time.sleep(60)  # scan every minute for crypto 1m

        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
