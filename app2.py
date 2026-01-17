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
import os # 引入 OS 模組進行本機檔案操作
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 0. 頁面設定與本機檔案檢查
# ==========================================
st.set_page_config(page_title="App 11.1 反脆弱指揮官", layout="wide")

LOCAL_NEWS_FILE = "news_data_local.csv"

# 初始化：如果本機有檔案，直接載入
if 'news_data' not in st.session_state:
    if os.path.exists(LOCAL_NEWS_FILE):
        try:
            df_local = pd.read_csv(LOCAL_NEWS_FILE)
            if 'Date' in df_local.columns:
                df_local['Date'] = pd.to_datetime(df_local['Date'])
            st.session_state['news_data'] = df_local
            st.toast(f"✅ 已自動載入本機存檔：{len(df_local)} 筆新聞", icon="📂")
        except:
            st.session_state['news_data'] = pd.DataFrame()
    else:
        st.session_state['news_data'] = pd.DataFrame()

st.title("🦅 App 11.1: 反脆弱指揮官 (本機存檔增強版)")
st.markdown("""
**核心升級：**
1.  **本機持久化**：新聞抓取後直接寫入硬碟 `news_data_local.csv`，重整網頁資料不遺失。
2.  **自動載入**：程式啟動時會優先讀取本機舊檔，節省爬蟲時間。
""")

# ==========================================
# 1. 新聞爬蟲 (含寫入硬碟功能)
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
# 2. 定價層 (四維定價)
# ==========================================
def train_rf_model(df, ticker):
    try:
        data = df[['Close']].copy()
        data['Ret'] = data['Close'].pct_change()
        data['Vol'] = data['Ret'].rolling(20).std()
        data['SMA'] = data['Close'].rolling(20).mean()
        data['Target'] = data['Close'].shift(-5) 
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
    t_atr = current + (atr * np.sqrt(5))
    recent = df_price['Close'].iloc[-60:]
    t_fib = recent.max() + (recent.max() - recent.min()) * 0.618
    mu = df_price['Close'].pct_change().mean()
    t_mc = current * ((1 + mu) ** 5)
    t_rf = train_rf_model(df_price, ticker)
    if t_rf is None: t_rf = t_mc
    avg_target = (t_atr + t_fib + t_mc + t_rf) / 4
    return avg_target

# ==========================================
# 3. 反脆弱回測 (5日)
# ==========================================
def run_antifragile_backtest(df_price, df_news_ticker):
    df = df_price.copy()
    
    if not df_news_ticker.empty:
        if not pd.api.types.is_datetime64_any_dtype(df_news_ticker['Date']):
             df_news_ticker['Date'] = pd.to_datetime(df_news_ticker['Date'])
        df_news_ticker['Weight'] = df_news_ticker['Region'].apply(lambda x: 1.2 if x != 'US' else 1.0)
        df_news_ticker['W_Score'] = df_news_ticker['Score'] * df_news_ticker['Weight']
        daily_score = df_news_ticker.groupby('Date')['W_Score'].mean()
        df = df.join(daily_score, how='left').fillna(0)
        df['News_Roll'] = df['W_Score'].rolling(3).mean()
    else:
        df['News_Roll'] = 0
        
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Bias'] = (df['Close'] - df['MA20']) / df['MA20']
    
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    df['RSI'] = 100 - (100 / (1 + up.ewm(com=13).mean() / down.ewm(com=13).mean()))
    
    df['Ret_5D'] = df['Close'].shift(-5) / df['Close'] - 1
    
    # 策略 1: 逆勢抄底
    buy_mask = (df['News_Roll'] < -0.1) & ((df['Bias'] < -0.03) | (df['RSI'] < 35))
    buy_opps = df[buy_mask].dropna(subset=['Ret_5D'])
    
    if len(buy_opps) > 0:
        win_rate = len(buy_opps[buy_opps['Ret_5D'] > 0]) / len(buy_opps)
        count = len(buy_opps)
        avg_ret = buy_opps['Ret_5D'].mean()
    else:
        win_rate = 0.0; count = 0; avg_ret = 0.0
        
    # 策略 2: 順勢追高 (對照組)
    mom_mask = (df['News_Roll'] > 0.1) & (df['Bias'] > 0.03)
    mom_opps = df[mom_mask].dropna(subset=['Ret_5D'])
    if len(mom_opps) > 0:
        mom_win = len(mom_opps[mom_opps['Ret_5D'] > 0]) / len(mom_opps)
    else:
        mom_win = 0.0

    return win_rate, count, avg_ret, mom_win

# ==========================================
# 4. 主程式
# ==========================================
st.sidebar.title("控制台")
data_mode = st.sidebar.radio("數據來源", ["1. 優先使用本機/記憶體", "2. 強制重新抓取 (Live)", "3. 上傳 CSV"])

default_tickers = ["TSM", "NVDA", "AMD", "SOXL", "URA", "CLS"]
user_tickers = st.sidebar.text_area("代號", ", ".join(default_tickers))
ticker_list = [t.strip().upper() for t in user_tickers.split(',')]

# 狀態顯示
if not st.session_state['news_data'].empty:
    st.sidebar.success(f"目前資料庫：{len(st.session_state['news_data'])} 筆")
else:
    st.sidebar.warning("目前資料庫為空")

# 邏輯處理
if data_mode.startswith("2"): # 強制重抓
    if st.sidebar.button("🚀 啟動爬蟲 (覆蓋舊檔)"):
        all_news = []
        bar = st.sidebar.progress(0)
        for i, t in enumerate(ticker_list):
            df = fetch_global_news_12m(t)
            if not df.empty: all_news.append(df)
            bar.progress((i+1)/len(ticker_list))
            
        if all_news:
            news_df = pd.concat(all_news, ignore_index=True)
            # 1. 存入 Session
            st.session_state['news_data'] = news_df
            # 2. 寫入本機硬碟 (關鍵！)
            news_df.to_csv(LOCAL_NEWS_FILE, index=False)
            st.sidebar.success(f"已更新並寫入 {LOCAL_NEWS_FILE}")
            
elif data_mode.startswith("3"): # 上傳
    up = st.sidebar.file_uploader("上傳 news.csv", type=['csv'])
    if up:
        try:
            temp = pd.read_csv(up)
            temp['Date'] = pd.to_datetime(temp['Date'])
            st.session_state['news_data'] = temp
            # 也要寫入本機，方便下次使用
            temp.to_csv(LOCAL_NEWS_FILE, index=False) 
            st.sidebar.success("讀取並存檔成功")
        except: st.error("讀檔失敗")

# 分析執行
if st.button("🚀 執行反脆弱分析"):
    if st.session_state['news_data'].empty:
        st.error("請先取得數據！")
    else:
        st.subheader("📊 反脆弱戰略報告 (本機存檔版)")
        news_df = st.session_state['news_data']
        results = []
        
        for t in ticker_list:
            df_price = yf.download(t, period="2y", progress=False, auto_adjust=True)
            if isinstance(df_price.columns, pd.MultiIndex):
                temp = df_price['Close'][[t]].copy(); temp.columns = ['Close']
                temp['High'] = df_price['High'][t]
                temp['Low'] = df_price['Low'][t]
                df_price = temp
            else:
                df_price = df_price[['Close', 'High', 'Low']]
            
            df_news_t = news_df[news_df['Ticker'] == t].copy()
            
            win_rate, count, avg_ret, mom_win = run_antifragile_backtest(df_price, df_news_t)
            target = calc_4d_target(t, df_price)
            
            current_close = df_price['Close'].iloc[-1]
            ma20 = df_price['Close'].rolling(20).mean().iloc[-1]
            bias = (current_close - ma20) / ma20
            
            latest_news_score = 0
            if not df_news_t.empty:
                df_news_t['Date'] = pd.to_datetime(df_news_t['Date'])
                last_news = df_news_t.sort_values('Date').iloc[-1]
                latest_news_score = last_news['Score']
            
            signal = "⬜ 觀望"
            if latest_news_score < -0.1 and bias < -0.03:
                signal = "💎 逆勢抄底"
            elif latest_news_score > 0.3 and bias > 0.05:
                signal = "⚠️ 過熱警戒"

            results.append({
                'Ticker': t,
                'Anti_Win': win_rate,
                'Mom_Win': mom_win,
                'Count': count,
                'Avg_Ret_5D': avg_ret,
                'Current': current_close,
                'Target_5D': target,
                'Signal': signal,
                'News_Score': latest_news_score,
                'Bias': bias
            })
            
        res_df = pd.DataFrame(results)
        
        show = res_df.copy()
        show['Anti_Win'] = show['Anti_Win'].apply(lambda x: f"{x:.0%}")
        show['Mom_Win'] = show['Mom_Win'].apply(lambda x: f"{x:.0%}")
        show['Avg_Ret_5D'] = show['Avg_Ret_5D'].apply(lambda x: f"{x:+.1%}")
        show['Current'] = show['Current'].apply(lambda x: f"${x:.2f}")
        show['Target_5D'] = show['Target_5D'].apply(lambda x: f"${x:.2f}")
        show['News_Score'] = show['News_Score'].apply(lambda x: f"{x:.2f}")
        show['Bias'] = show['Bias'].apply(lambda x: f"{x:+.1%}")
        
        st.dataframe(show[['Ticker', 'Signal', 'Anti_Win', 'Mom_Win', 'Avg_Ret_5D', 'Current', 'Target_5D', 'News_Score', 'Bias']].style.map(
            lambda x: 'background-color: #00FF7F; color: black' if '抄底' in str(x) else '', subset=['Signal']
        ))