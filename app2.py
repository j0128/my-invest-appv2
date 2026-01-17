import streamlit as st
import feedparser
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import urllib.parse
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(page_title="App 10.0 天眼指揮官", layout="wide")

st.title("🦅 App 10.0: 天眼指揮官 (Macro-Regime Integrated)")
st.markdown("""
**突破關鍵：**
1.  **加入環境濾網 (Macro Filter)**：監控 **美元(DXY)**、**殖利率(TNX)**、**風險債(HYG)**。
2.  **順勢而為**：當「潮水退去」(Risk-Off) 時，強制過濾掉所有買進訊號，避免接刀。
""")

# ==========================================
# 1. 天眼系統：總體環境掃描 (Macro Scanner)
# ==========================================
@st.cache_data(ttl=3600*4)
def fetch_macro_context():
    """
    抓取 DXY, TNX, HYG, VIX 來判斷目前是 Risk-On 還是 Risk-Off
    """
    tickers = ['DX-Y.NYB', '^TNX', 'HYG', '^VIX']
    data = yf.download(tickers, period="1y", progress=False)['Close']
    
    # 計算趨勢 (簡單均線與斜率)
    status = {}
    
    # 1. 美元指數 (DXY) - 資金抽水機
    dxy = data['DX-Y.NYB']
    dxy_ma20 = dxy.rolling(20).mean().iloc[-1]
    dxy_trend = "⬆️ 強勢吸金" if dxy.iloc[-1] > dxy_ma20 else "⬇️ 弱勢(利多)"
    
    # 2. 10年期殖利率 (TNX) - 估值殺手
    tnx = data['^TNX']
    tnx_ma20 = tnx.rolling(20).mean().iloc[-1]
    tnx_trend = "⬆️ 殺估值" if tnx.iloc[-1] > tnx_ma20 else "⬇️ 穩定"
    
    # 3. 高收益債 (HYG) - 風險胃納 (聰明錢)
    hyg = data['HYG']
    hyg_ma20 = hyg.rolling(20).mean().iloc[-1]
    # HYG 漲代表資金願意冒險 (Risk-On)
    risk_appetite = "🦁 Risk-On (冒險)" if hyg.iloc[-1] > hyg_ma20 else "🐻 Risk-Off (避險)"
    
    # 4. 恐慌指數 (VIX)
    vix = data['^VIX'].iloc[-1]
    vix_status = "😨 恐慌" if vix > 20 else "😌 平靜"
    
    # 綜合判定環境分 (Macro Score)
    # 分數越高越適合做多
    macro_score = 0
    if dxy.iloc[-1] < dxy_ma20: macro_score += 1 # 美元弱，好
    if tnx.iloc[-1] < tnx_ma20: macro_score += 1 # 利率降，好
    if hyg.iloc[-1] > hyg_ma20: macro_score += 1 # 聰明錢買債，好
    if vix < 20: macro_score += 1                # 不恐慌，好
    
    regime = "🔴 紅燈 (現金為王)"
    if macro_score >= 3: regime = "🟢 綠燈 (積極進攻)"
    elif macro_score == 2: regime = "🟡 黃燈 (選股不選市)"
    
    return {
        'Regime': regime,
        'Score': macro_score,
        'DXY': dxy_trend,
        'TNX': tnx_trend,
        'HYG': risk_appetite,
        'VIX': vix_status,
        'Raw_Data': data # 回測用
    }

# ==========================================
# 2. 新聞爬蟲 (保留上一版功能)
# ==========================================
TICKER_MAP = {
    'TSM': {'TW': '台積電', 'JP': 'TSMC', 'EU': 'TSMC'},
    'NVDA': {'TW': '輝達', 'JP': 'NVIDIA', 'EU': 'Nvidia'},
    'AMD': {'TW': '超微', 'JP': 'AMD', 'EU': 'AMD'},
    'URA': {'TW': '鈾礦', 'JP': 'ウラン', 'EU': 'Uranium'},
    'SOXL': {'TW': '半導體', 'JP': '半導体', 'EU': 'Semiconductor'},
    'BTC-USD': {'TW': '比特幣', 'JP': 'ビットコイン', 'EU': 'Bitcoin'}
}

MULTILINGUAL_DICT = {
    'ZH': {'UP': ['大漲','漲停','創高','利多','爆發','擴產','急單'], 'DOWN': ['大跌','跌停','重挫','利空','砍單','衰退']},
    'JA': {'UP': ['上昇','急騰','最高値','好調','増益'], 'DOWN': ['下落','急落','最安値','不調','減益']},
    'DE': {'UP': ['anstieg','rekord','gewinn','kaufen'], 'DOWN': ['verlust','fallen','krise','verkaufen']}
}

@st.cache_data(ttl=3600*24)
def fetch_global_news_12m(ticker):
    news_history = []
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=12) 
    map_info = TICKER_MAP.get(ticker, {})
    term_us = f"{ticker}+stock" if len(ticker) <= 4 else ticker
    term_tw = urllib.parse.quote(map_info.get('TW', ticker))
    term_jp = urllib.parse.quote(map_info.get('JP', ticker))
    term_eu = urllib.parse.quote(map_info.get('EU', ticker))

    current = start_date
    while current < end_date:
        next_month = current + relativedelta(months=1)
        d_after = current.strftime('%Y-%m-%d')
        d_before = next_month.strftime('%Y-%m-%d')
        
        urls = [
            (f"https://news.google.com/rss/search?q={term_us}+after:{d_after}+before:{d_before}&hl=en-US&gl=US&ceid=US:en", 'US'),
            (f"https://news.google.com/rss/search?q={term_us}+after:{d_after}+before:{d_before}&hl=en-GB&gl=GB&ceid=GB:en", 'EU_UK')
        ]
        if ticker in ['TSM', 'NVDA', 'AMD', '0050.TW', 'CLS', 'SOXL']:
            urls.append((f"https://news.google.com/rss/search?q={term_tw}+after:{d_after}+before:{d_before}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant", 'TW'))
        if ticker in ['TSM', 'NVDA', 'SOXL', 'URA', 'BTC-USD']:
            urls.append((f"https://news.google.com/rss/search?q={term_jp}+after:{d_after}+before:{d_before}&hl=ja&gl=JP&ceid=JP:ja", 'JP'))
        if ticker in ['URA', 'SOXL', 'CLS', 'AMD']:
            urls.append((f"https://news.google.com/rss/search?q={term_eu}+after:{d_after}+before:{d_before}&hl=de&gl=DE&ceid=DE:de", 'EU_DE'))

        for url, region in urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:2]: 
                    title = entry.title
                    pub_date = pd.to_datetime(entry.published).date() if hasattr(entry, 'published') else current.date()
                    score = 0
                    if region in ['US', 'EU_UK']:
                        score = TextBlob(title).sentiment.polarity
                        if any(x in title.lower() for x in ['beat', 'surge', 'record']): score += 0.3
                    elif region == 'TW':
                        for k in MULTILINGUAL_DICT['ZH']['UP']: 
                            if k in title: score += 0.5
                    elif region == 'JP':
                        for k in MULTILINGUAL_DICT['JA']['UP']: 
                            if k in title: score += 0.5
                    elif region == 'EU_DE':
                        for k in MULTILINGUAL_DICT['DE']['UP']: 
                            if k in title.lower(): score += 0.5
                    if score != 0:
                        news_history.append({'Ticker': ticker, 'Date': pub_date, 'Region': region, 'Title': title, 'Score': score})
            except: pass
        current = next_month
        time.sleep(0.05)
    return pd.DataFrame(news_history)

# ==========================================
# 3. 定價層 (Quant Engine)
# ==========================================
def train_rf_model(df, ticker):
    try:
        data = df[['Close']].copy()
        data['Ret'] = data['Close'].pct_change()
        data['Vol'] = data['Ret'].rolling(20).std()
        data['SMA'] = data['Close'].rolling(20).mean()
        data['Target'] = data['Close'].shift(-22)
        data = data.dropna()
        if len(data) < 60: return None
        X = data[['Ret', 'Vol', 'SMA']]
        y = data['Target']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        last_row = data.iloc[[-1]][['Ret', 'Vol', 'SMA']]
        return model.predict(last_row)[0]
    except: return None

def calc_4d_target(ticker, df_price):
    current = df_price['Close'].iloc[-1]
    tr = df_price['High'] - df_price['Low']
    atr = tr.rolling(14).mean().iloc[-1]
    t_atr = current + (atr * np.sqrt(22))
    recent = df_price['Close'].iloc[-60:]
    t_fib = recent.max() + (recent.max() - recent.min()) * 0.618
    mu = df_price['Close'].pct_change().mean()
    t_mc = current * ((1 + mu) ** 22)
    t_rf = train_rf_model(df_price, ticker)
    if t_rf is None: t_rf = t_mc
    avg_target = (t_atr + t_fib + t_mc + t_rf) / 4
    return avg_target, {'ATR': t_atr, 'Fib': t_fib, 'MC': t_mc, 'RF': t_rf}

# ==========================================
# 4. 回測層：天眼回測 (God View Backtest)
# ==========================================
def run_historical_validation(df_price, df_news_ticker, macro_data):
    """
    加入 Macro Filter 的回測
    只有在 Macro Score >= 2 (黃燈或綠燈) 時，才允許狙擊
    """
    df = df_price.copy()
    
    # 1. 整合新聞
    if not df_news_ticker.empty:
        df_news_ticker['Weight'] = df_news_ticker['Region'].apply(lambda x: 1.2 if x != 'US' else 1.0)
        df_news_ticker['W_Score'] = df_news_ticker['Score'] * df_news_ticker['Weight']
        daily_score = df_news_ticker.groupby('Date')['W_Score'].mean()
        df = df.join(daily_score, how='left').fillna(0)
        df['News_Roll'] = df['W_Score'].rolling(3).mean()
    else:
        df['News_Roll'] = 0
        
    # 2. 整合宏觀 (Macro)
    # 將 HYG, DXY 等數據對齊到個股日期
    macro_aligned = macro_data.reindex(df.index).ffill()
    
    # 計算 Macro Condition (歷史上的每天)
    # 條件：HYG > HYG_MA20 (Risk On) AND DXY < DXY_MA20 (Dollar Weak) -> 簡化版 Risk On
    macro_aligned['HYG_MA'] = macro_aligned['HYG'].rolling(20).mean()
    macro_aligned['Risk_On'] = macro_aligned['HYG'] > macro_aligned['HYG_MA']
    
    df = df.join(macro_aligned[['Risk_On']], how='left').fillna(False)
    
    # 3. 技術特徵
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_Slope'] = df['OBV'].diff(5)
    vol_mean = df['Volume'].rolling(20).mean()
    vol_std = df['Volume'].rolling(20).std()
    df['Vol_Z'] = (df['Volume'] - vol_mean) / (vol_std + 1e-9)
    
    # 4. 未來回報
    df['Ret_1M'] = df['Close'].shift(-22) / df['Close'] - 1
    
    # --- 回測: 天眼狙擊手 ---
    # 條件: News>0.1 & OBV>0 & Vol>1.5 & **Risk_On==True**
    sniper_mask = (df['News_Roll'] > 0.1) & (df['OBV_Slope'] > 0) & (df['Vol_Z'] > 1.5) & (df['Risk_On'] == True)
    
    sniper_opportunities = df[sniper_mask].dropna(subset=['Ret_1M'])
    
    if len(sniper_opportunities) > 0:
        sniper_wins = sniper_opportunities[sniper_opportunities['Ret_1M'] > 0]
        sniper_win_rate = len(sniper_wins) / len(sniper_opportunities)
        sniper_count = len(sniper_opportunities)
        avg_return = sniper_opportunities['Ret_1M'].mean()
    else:
        sniper_win_rate = 0.0
        sniper_count = 0
        avg_return = 0.0
        
    return sniper_win_rate, sniper_count, avg_return

# ==========================================
# 5. 主程式
# ==========================================
st.sidebar.title("控制台")
data_mode = st.sidebar.radio("數據來源", ["1. 即時爬取", "2. 上傳 CSV"])
default_tickers = ["TSM", "NVDA", "AMD", "SOXL", "URA", "CLS"]
user_tickers = st.sidebar.text_area("代號", ", ".join(default_tickers))
ticker_list = [t.strip().upper() for t in user_tickers.split(',')]

# 1. 先抓宏觀數據
macro_info = fetch_macro_context()
st.subheader(f"🌍 天眼環境掃描: {macro_info['Regime']} (分數: {macro_info['Score']}/4)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("美元 (DXY)", macro_info['DXY'])
c2.metric("殖利率 (TNX)", macro_info['TNX'])
c3.metric("風險胃納 (HYG)", macro_info['HYG'])
c4.metric("恐慌 (VIX)", macro_info['VIX'])
st.divider()

news_df = pd.DataFrame()
run = False

if data_mode.startswith("1"):
    if st.sidebar.button("🚀 啟動天眼"):
        all_news = []
        bar = st.sidebar.progress(0)
        for i, t in enumerate(ticker_list):
            df = fetch_global_news_12m(t)
            if not df.empty: all_news.append(df)
            bar.progress((i+1)/len(ticker_list))
        if all_news:
            news_df = pd.concat(all_news, ignore_index=True)
            run = True
else:
    up = st.sidebar.file_uploader("上傳 news.csv", type=['csv'])
    if up:
        news_df = pd.read_csv(up)
        news_df['Date'] = pd.to_datetime(news_df['Date'])
        run = st.sidebar.button("🚀 執行")

if run:
    st.subheader("📊 天眼狙擊報告 (含 Risk-On/Off 濾網驗證)")
    
    results = []
    for t in ticker_list:
        df_price = yf.download(t, period="2y", progress=False, auto_adjust=True)
        if isinstance(df_price.columns, pd.MultiIndex):
            temp = df_price['Close'][[t]].copy(); temp.columns = ['Close']
            temp['Volume'] = df_price['Volume'][t]
            temp['High'] = df_price['High'][t]
            temp['Low'] = df_price['Low'][t]
            df_price = temp
        else:
            df_price = df_price[['Close', 'Volume', 'High', 'Low']]
            
        df_news_t = news_df[news_df['Ticker'] == t].copy() if not news_df.empty else pd.DataFrame()
        
        # 執行天眼回測
        win_rate, count, avg_ret = run_historical_validation(df_price, df_news_t, macro_info['Raw_Data'])
        target, _ = calc_4d_target(t, df_price)
        
        # 判斷當下狀態 (結合 Macro)
        # 只有在 Macro 綠燈/黃燈時才給建議
        can_trade = macro_info['Score'] >= 2
        status = "🛑 環境紅燈 (禁止操作)" if not can_trade else "⬜ 觀望"
        action = "Cash" if not can_trade else "Hold"
        
        if can_trade:
            # 這裡簡化判斷，實際可加入新聞邏輯
            pass 
            
        results.append({
            'Ticker': t,
            'Current': df_price['Close'].iloc[-1],
            'Target': target,
            'Sniper_Win': win_rate,
            'Sniper_Count': count,
            'Avg_Return': avg_ret,
            'Macro_Filter': "PASS" if can_trade else "BLOCK"
        })
        
    res_df = pd.DataFrame(results)
    
    show = res_df.copy()
    show['Sniper_Win'] = show['Sniper_Win'].apply(lambda x: f"{x:.0%}")
    show['Avg_Return'] = show['Avg_Return'].apply(lambda x: f"{x:+.1%}")
    show['Current'] = show['Current'].apply(lambda x: f"${x:.2f}")
    show['Target'] = show['Target'].apply(lambda x: f"${x:.2f}")

    st.dataframe(show)