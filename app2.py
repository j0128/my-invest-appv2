import streamlit as st
import feedparser
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import urllib.parse
import os
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 0. 頁面設定與本機檔案
# ==========================================
st.set_page_config(page_title="App 14.0 智能定投指揮官", layout="wide")
LOCAL_NEWS_FILE = "news_data_local.csv"

# 初始化 Session State (新聞資料庫)
if 'news_data' not in st.session_state:
    if os.path.exists(LOCAL_NEWS_FILE):
        try:
            df_local = pd.read_csv(LOCAL_NEWS_FILE)
            if 'Date' in df_local.columns:
                df_local['Date'] = pd.to_datetime(df_local['Date'])
            st.session_state['news_data'] = df_local
        except: st.session_state['news_data'] = pd.DataFrame()
    else: st.session_state['news_data'] = pd.DataFrame()

st.title("🦅 App 14.0: 智能定投指揮官 (Smart DCA Backtest)")
st.markdown("""
**定投對決實驗：**
* **情境**：初始資金 $10,000，每月月初加碼 $10,000。
* **🔴 無腦定投 (Blind DCA)**：每月 1 號拿到錢直接買，不賣出。
* **🟢 智能定投 (Smart DCA)**：拿到錢先**存現金**，直到出現「趨勢回調訊號」才買進；若趨勢破壞則**賣出避險**。
""")

# ==========================================
# 1. 核心工具：宏觀 & VWAP
# ==========================================
@st.cache_data(ttl=3600*4)
def fetch_macro_context():
    tickers = ['DX-Y.NYB', '^TNX', 'HYG', '^VIX']
    try:
        data = yf.download(tickers, period="2y", progress=False)['Close']
        # Risk-On 定義: HYG 趨勢向上 OR DXY 趨勢向下
        hyg = data['HYG']
        hyg_ma = hyg.rolling(20).mean()
        dxy = data['DX-Y.NYB']
        dxy_ma = dxy.rolling(20).mean()
        
        # 用 True/False 序列代表每一天是否適合做多
        risk_on_series = (hyg > hyg_ma) | (dxy < dxy_ma)
        return risk_on_series
    except:
        return pd.Series(True, index=pd.date_range(end=datetime.now(), periods=500))

def calculate_vwap(df, window=20):
    v = df['Volume']
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * v).rolling(window).sum() / v.rolling(window).sum()

# ==========================================
# 2. 新聞爬蟲 (維持 App 13.0 功能)
# ==========================================
TICKER_MAP = {
    'TSM': {'TW': '台積電', 'JP': 'TSMC', 'EU': 'TSMC'},
    'NVDA': {'TW': '輝達', 'JP': 'NVIDIA', 'EU': 'Nvidia'},
    'AMD': {'TW': '超微', 'JP': 'AMD', 'EU': 'AMD'},
    'URA': {'TW': '鈾礦', 'JP': 'ウラン', 'EU': 'Uranium'},
    'SOXL': {'TW': '半導體', 'JP': '半導体', 'EU': 'Semiconductor'},
    'BTC-USD': {'TW': '比特幣', 'JP': 'ビットコイン', 'EU': 'Bitcoin'}
}

def fetch_global_news_12m(ticker):
    # (此函式與前版相同，省略重複代碼，確保完整性請複製 App 13.0 的這部分)
    # 這裡僅示意，實際執行需包含完整爬蟲邏輯
    news_history = []
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=12) 
    map_info = TICKER_MAP.get(ticker, {})
    term_us = f"{ticker}+stock" if len(ticker) <= 4 else ticker
    term_tw = urllib.parse.quote(map_info.get('TW', ticker))
    
    current = start_date
    while current < end_date:
        next_month = current + relativedelta(months=1)
        d_after = current.strftime('%Y-%m-%d')
        d_before = next_month.strftime('%Y-%m-%d')
        url = f"https://news.google.com/rss/search?q={term_us}+after:{d_after}+before:{d_before}&hl=en-US&gl=US&ceid=US:en"
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:2]:
                title = entry.title
                score = TextBlob(title).sentiment.polarity
                if score != 0:
                    news_history.append({'Ticker': ticker, 'Date': pd.to_datetime(entry.published).date(), 'Region': 'US', 'Title': title, 'Score': score})
        except: pass
        current = next_month
        time.sleep(0.05)
    return pd.DataFrame(news_history)

# ==========================================
# 3. 智能定投回測引擎 (Smart DCA Engine)
# ==========================================
def run_smart_dca_simulation(ticker, df_price, df_news, macro_series):
    df = df_price.copy()
    
    # 1. 整合特徵
    if not df_news.empty:
        if not pd.api.types.is_datetime64_any_dtype(df_news['Date']):
             df_news['Date'] = pd.to_datetime(df_news['Date'])
        daily_score = df_news.groupby('Date')['Score'].mean()
        df = df.join(daily_score, how='left').fillna(0)
        df['News_Roll'] = df['Score'].rolling(3).mean()
    else:
        df['News_Roll'] = 0
        
    # 宏觀與技術
    macro_aligned = macro_series.reindex(df.index).ffill().fillna(True)
    df['Risk_On'] = macro_aligned
    df['MA60'] = df['Close'].rolling(60).mean()
    df['VWAP'] = calculate_vwap(df, 20)
    df['Dev_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']
    
    # 2. 回測變數
    # Smart Strategy
    smart_cash = 10000.0
    smart_shares = 0.0
    
    # Blind DCA
    dca_shares = 0.0
    dca_cash_flow = 10000.0 # 每月進來的錢
    
    total_invested = 10000.0
    history = []
    last_month = -1
    
    # 從第 60 天開始 (讓 MA60 算出來)
    start_idx = 60
    
    # 訊號向量化
    # 買入: 趨勢向上 & 價格在 VWAP 附近 (±5%) & 宏觀好
    cond_buy = (df['Close'] > df['MA60']) & (df['Dev_VWAP'].abs() < 0.05) & (df['Risk_On'])
    # 賣出: 趨勢跌破 MA60 OR 乖離過熱 > 10%
    cond_sell = (df['Close'] < df['MA60']) | (df['Dev_VWAP'] > 0.1)
    
    for i in range(start_idx, len(df)):
        date = df.index[i]
        price = df['Close'].iloc[i]
        
        # --- A. 發薪日 (Monthly Contribution) ---
        if date.month != last_month:
            if last_month != -1: # 排除第一個月
                income = 10000.0
                smart_cash += income
                total_invested += income
                
                # Blind DCA: 拿到錢直接買
                dca_shares += income / price
                
            last_month = date.month
            
        # --- B. 智能交易 (Smart Strategy) ---
        is_buy = cond_buy.iloc[i]
        is_sell = cond_sell.iloc[i]
        
        # 優先檢查賣出 (保命)
        if smart_shares > 0 and is_sell_signal:
            smart_cash += smart_shares * price
            smart_shares = 0
            
        # 檢查買入 (有現金才買)
        elif smart_cash > 0 and is_buy_signal:
            smart_shares += smart_cash / price
            smart_cash = 0
            
        # --- C. 資產結算 ---
        smart_val = smart_cash + (smart_shares * price)
        dca_val = (dca_shares * price) # DCA 沒有現金，全在股票裡 (除了剛發薪水那一刻，簡化計算)
        
        history.append({
            'Date': date,
            'Smart_Val': smart_val,
            'DCA_Val': dca_val,
            'Invested': total_invested
        })
        
    res_df = pd.DataFrame(history)
    if res_df.empty: return 0, 0, 0, pd.DataFrame()
    
    # 結果計算
    final_smart = res_df['Smart_Val'].iloc[-1]
    final_dca = res_df['DCA_Val'].iloc[-1]
    tot_inv = res_df['Invested'].iloc[-1]
    
    smart_roi = (final_smart - tot_inv) / tot_inv
    dca_roi = (final_dca - tot_inv) / tot_inv
    
    return smart_roi, dca_roi, tot_inv, res_df

# ==========================================
# 4. 主程式
# ==========================================
st.sidebar.title("控制台")
# 數據模式
data_mode = st.sidebar.radio("數據來源", ["1. 使用記憶體/本機", "2. 強制重抓", "3. 上傳 CSV"])
default_tickers = ["TSM", "NVDA", "AMD", "SOXL", "URA", "CLS", "0050.TW"]
user_tickers = st.sidebar.text_area("代號", ", ".join(default_tickers))
ticker_list = [t.strip().upper() for t in user_tickers.split(',')]

# 宏觀數據
risk_on_series = fetch_macro_context()

# (此處省略爬蟲/上傳邏輯代碼，請保留 App 13.0 的那部分)
# ... [Insert News Fetching Logic Here] ...

if st.button("🚀 執行定投對決"):
    if st.session_state['news_data'].empty:
        st.warning("⚠️ 無新聞數據，僅使用技術與宏觀指標")
    
    news_df = st.session_state['news_data']
    results = []
    
    st.subheader("📊 智能定投戰果 (Smart vs Blind)")
    
    for t in ticker_list:
        df_price = yf.download(t, period="2y", progress=False, auto_adjust=True)
        # 格式整理
        if isinstance(df_price.columns, pd.MultiIndex):
            temp = df_price['Close'][[t]].copy(); temp.columns = ['Close']
            temp['Volume'] = df_price['Volume'][t]
            temp['High'] = df_price['High'][t]
            temp['Low'] = df_price['Low'][t]
            df_price = temp
        else:
            df_price = df_price[['Close', 'Volume', 'High', 'Low']]
            
        df_news_t = news_df[news_df['Ticker'] == t].copy() if not news_df.empty else pd.DataFrame()
        
        # 執行回測
        smart_roi, dca_roi, inv, history = run_smart_dca_simulation(t, df_price, df_news_t, risk_on_series)
        
        # 計算 Alpha (超額報酬)
        alpha = smart_roi - dca_roi
        
        results.append({
            'Ticker': t,
            'Invested': inv,
            'Smart_ROI': smart_roi,
            'DCA_ROI': dca_roi,
            'Alpha': alpha,
            'Smart_Final': inv * (1+smart_roi)
        })
        
        # 畫圖：只畫 Alpha 最大的前兩名，避免洗版
        if abs(alpha) > 0.05:
            with st.expander(f"📈 {t} 資金曲線 (Alpha: {alpha:+.1%})"):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=history['Date'], y=history['Smart_Val'], name='智能定投', line=dict(color='#00FF7F', width=2)))
                fig.add_trace(go.Scatter(x=history['Date'], y=history['DCA_Val'], name='無腦定投', line=dict(color='#FF4B4B', width=2, dash='dot')))
                fig.add_trace(go.Scatter(x=history['Date'], y=history['Invested'], name='總投入本金', line=dict(color='gray', dash='dash')))
                fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

    res_df = pd.DataFrame(results)
    
    # 格式化
    show = res_df.copy()
    show['Invested'] = show['Invested'].apply(lambda x: f"${x:,.0f}")
    show['Smart_ROI'] = show['Smart_ROI'].apply(lambda x: f"{x:+.1%}")
    show['DCA_ROI'] = show['DCA_ROI'].apply(lambda x: f"{x:+.1%}")
    show['Alpha'] = show['Alpha'].apply(lambda x: f"{x:+.1%}")
    show['Smart_Final'] = show['Smart_Final'].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(show.style.map(
        lambda x: 'color: #00FF7F' if '+' in str(x) and float(str(x).strip('%+')) > 0 else 'color: white',
        subset=['Alpha']
    ))