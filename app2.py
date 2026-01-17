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
st.set_page_config(page_title="App 9.1 狙擊手回測版", layout="wide")

st.title("🦅 App 9.1: 狙擊手指揮官 (含真實勝率回測)")
st.markdown("""
**回測機制升級：**
1.  **狙擊勝率 (Sniper Win Rate)**：統計過去一年，當「新聞+OBV+爆量」三燈全亮時，進場持有 1 個月(22交易日)的勝率。
2.  **方向準確度 (Dir Acc)**：綜合評分對於「下個月漲跌」判斷的長期準確度。
""")

# ==========================================
# 1. 數據層：全球新聞爬蟲
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
# 2. 定價層：四維定價
# ==========================================
def train_rf_model(df, ticker):
    try:
        data = df[['Close']].copy()
        data['Ret'] = data['Close'].pct_change()
        data['Vol'] = data['Ret'].rolling(20).std()
        data['SMA'] = data['Close'].rolling(20).mean()
        data['Target'] = data['Close'].shift(-22) # 預測22天(一個月)後
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
    t_atr = current + (atr * np.sqrt(22)) # 調整為22天
    recent = df_price['Close'].iloc[-60:]
    t_fib = recent.max() + (recent.max() - recent.min()) * 0.618
    mu = df_price['Close'].pct_change().mean()
    t_mc = current * ((1 + mu) ** 22)
    t_rf = train_rf_model(df_price, ticker)
    if t_rf is None: t_rf = t_mc
    avg_target = (t_atr + t_fib + t_mc + t_rf) / 4
    return avg_target, {'ATR': t_atr, 'Fib': t_fib, 'MC': t_mc, 'RF': t_rf}

# ==========================================
# 3. 回測層：時光機驗證 (Historical Validation)
# ==========================================
def run_historical_validation(df_price, df_news_ticker):
    """
    對過去一年進行逐日回測
    目標：預測 22 天後的漲跌 (Month-Over-Month)
    """
    df = df_price.copy()
    
    # 1. 準備新聞特徵 (歷史對齊)
    if not df_news_ticker.empty:
        df_news_ticker['Weight'] = df_news_ticker['Region'].apply(lambda x: 1.2 if x != 'US' else 1.0)
        df_news_ticker['W_Score'] = df_news_ticker['Score'] * df_news_ticker['Weight']
        daily_score = df_news_ticker.groupby('Date')['W_Score'].mean()
        df = df.join(daily_score, how='left').fillna(0)
        df['News_Roll'] = df['W_Score'].rolling(3).mean()
    else:
        df['News_Roll'] = 0
        
    # 2. 準備技術特徵 (歷史對齊)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_Slope'] = df['OBV'].diff(5) # 5日 OBV 趨勢
    
    vol_mean = df['Volume'].rolling(20).mean()
    vol_std = df['Volume'].rolling(20).std()
    df['Vol_Z'] = (df['Volume'] - vol_mean) / (vol_std + 1e-9)
    
    # 3. 計算 Alpha Score (簡化版，為了回測速度)
    # 結合: News + Tech(均線) + OBV
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Score_Tech'] = np.where(df['Close'] > df['MA20'], 1, -1)
    df['Alpha_Signal'] = (df['News_Roll'] * 0.4) + (df['Score_Tech'] * 0.4) + (np.sign(df['OBV_Slope']) * 0.2)
    
    # 4. 定義 "未來真實回報" (22天後)
    df['Ret_1M'] = df['Close'].shift(-22) / df['Close'] - 1
    
    # --- 回測 A: 方向準確度 (Directional Accuracy) ---
    # 預測看多(Alpha>0) 且 實際漲 > 0
    valid_rows = df.dropna(subset=['Ret_1M'])
    if len(valid_rows) > 0:
        correct_dir = np.sign(valid_rows['Alpha_Signal']) == np.sign(valid_rows['Ret_1M'])
        dir_acc = correct_dir.mean()
    else:
        dir_acc = 0.5
        
    # --- 回測 B: 狙擊手勝率 (Sniper Win Rate) ---
    # 條件: News>0.1 & OBV>0 & Vol_Z>1.5
    sniper_mask = (df['News_Roll'] > 0.1) & (df['OBV_Slope'] > 0) & (df['Vol_Z'] > 1.5)
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
        
    return dir_acc, sniper_win_rate, sniper_count, avg_return, df

# ==========================================
# 4. 決策層
# ==========================================
def analyze_sniper_full(ticker, df_price, df_news_ticker):
    # 執行回測
    dir_acc, sniper_rate, sniper_count, sniper_ret, df_processed = run_historical_validation(df_price, df_news_ticker)
    
    # 計算當下狀態
    target, details = calc_4d_target(ticker, df_price)
    current_row = df_processed.iloc[-1]
    
    # 狙擊判斷
    status = "⬜ 觀望"
    action = "Hold"
    
    is_news = current_row['News_Roll'] > 0.1
    is_obv = current_row['OBV_Slope'] > 0
    is_vol = current_row['Vol_Z'] > 1.5
    
    if is_news and is_obv and is_vol:
        status = "🎯 狙擊訊號 (Sniper)"
        action = "Strong Buy"
    elif is_news and not is_obv:
        status = "⚠️ 假突破 (Fakeout)"
        action = "Avoid"
    elif not is_news and is_obv:
        status = "🥷 潛伏 (Stealth)"
        action = "Buy"
        
    latest_news = "無新聞"
    if not df_news_ticker.empty:
        latest = df_news_ticker.sort_values('Date').iloc[-1]
        latest_news = f"[{latest['Region']}] {latest['Title']}"

    return {
        'Ticker': ticker,
        'Current': current_row['Close'],
        'Target_1M': target,
        'Upside': (target - current_row['Close']) / current_row['Close'],
        'Dir_Acc': dir_acc,          # 回測指標 1
        'Sniper_Win': sniper_rate,   # 回測指標 2
        'Sniper_Count': sniper_count,# 樣本數
        'Sniper_AvgRet': sniper_ret, # 平均獲利
        'Status': status,
        'Action': action,
        'Latest_News': latest_news
    }

# ==========================================
# 5. 主程式流程
# ==========================================
st.sidebar.title("控制台")
data_mode = st.sidebar.radio("數據來源", ["1. 即時爬取 (Live)", "2. 上傳 CSV"])
default_tickers = ["TSM", "NVDA", "AMD", "SOXL", "URA", "CLS"]
user_tickers = st.sidebar.text_area("代號", ", ".join(default_tickers))
ticker_list = [t.strip().upper() for t in user_tickers.split(',')]

news_df = pd.DataFrame()
run = False

if data_mode.startswith("1"):
    if st.sidebar.button("🚀 啟動回測"):
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
    # CSV 下載
    st.sidebar.download_button("📥 下載本次新聞數據", news_df.to_csv(index=False).encode('utf-8'), "news_data.csv", "text/csv")
    
    st.subheader("📊 狙擊手戰略報告 (含 12 個月回測驗證)")
    
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
        res = analyze_sniper_full(t, df_price, df_news_t)
        results.append(res)
        
    res_df = pd.DataFrame(results)
    
    # 格式化
    show = res_df.copy()
    show['Dir_Acc'] = show['Dir_Acc'].apply(lambda x: f"{x:.0%}")
    show['Sniper_Win'] = show['Sniper_Win'].apply(lambda x: f"{x:.0%}")
    show['Sniper_AvgRet'] = show['Sniper_AvgRet'].apply(lambda x: f"{x:+.1%}")
    for c in ['Current', 'Target_1M']: show[c] = show[c].apply(lambda x: f"${x:.2f}")
    show['Upside'] = show['Upside'].apply(lambda x: f"{x:+.1%}")
    
    # 顯示主表
    st.dataframe(show[['Ticker', 'Status', 'Action', 'Dir_Acc', 'Sniper_Win', 'Sniper_Count', 'Sniper_AvgRet', 'Current', 'Target_1M', 'Latest_News']].style.map(
        lambda x: 'background-color: #00FF7F; color: black' if '狙擊' in str(x) else '', subset=['Status']
    ))
    
    # 驗證散佈圖
    fig = go.Figure()
    for i, row in res_df.iterrows():
        # X軸: 方向準確度 (代表模型多懂這支股票)
        # Y軸: 狙擊勝率 (代表爆發訊號多準)
        size = np.log(row['Sniper_Count'] + 1) * 15 # 樣本數越多泡泡越大
        color = '#00FF7F' if row['Sniper_Win'] > 0.6 else '#FF4B4B'
        
        fig.add_trace(go.Scatter(
            x=[row['Dir_Acc']], y=[row['Sniper_Win']],
            mode='markers+text', text=[row['Ticker']],
            textposition="top center", marker=dict(size=size, color=color),
            name=row['Ticker'],
            hovertemplate="<b>%{text}</b><br>方向準確度: %{x:.0%}<br>狙擊勝率: %{y:.0%}<br>樣本數: " + str(row['Sniper_Count'])
        ))
        
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="<b>模型可信度矩陣</b> (右上角=聖杯區)",
        xaxis_title="長期方向準確度 (12M)",
        yaxis_title="狙擊訊號勝率 (1M return > 0)",
        template="plotly_dark", height=500
    )
    st.plotly_chart(fig, use_container_width=True)