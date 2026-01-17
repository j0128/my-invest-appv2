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
st.set_page_config(page_title="App 10.3 天眼指揮官 (最終穩定版)", layout="wide")

st.title("🦅 App 10.3: 天眼指揮官 (Macro + 雙重回測 + 穩定修復)")
st.markdown("""
**修復報告：**
1.  **解決 TypeError**：修正 `np.sign` 與空值 (NaN) 衝突導致的崩潰，改用「積數判斷法」計算準確度。
2.  **完整功能**：包含天眼 (Macro)、新聞爬蟲、以及數據保全功能。
""")

# ==========================================
# 1. 天眼系統 (Macro)
# ==========================================
@st.cache_data(ttl=3600*4)
def fetch_macro_context():
    tickers = ['DX-Y.NYB', '^TNX', 'HYG', '^VIX']
    data = yf.download(tickers, period="1y", progress=False)['Close']
    
    # 趨勢判斷
    dxy = data['DX-Y.NYB']
    dxy_trend = "⬆️ 強勢" if dxy.iloc[-1] > dxy.rolling(20).mean().iloc[-1] else "⬇️ 弱勢"
    
    tnx = data['^TNX']
    tnx_trend = "⬆️ 升息" if tnx.iloc[-1] > tnx.rolling(20).mean().iloc[-1] else "⬇️ 降息"
    
    hyg = data['HYG']
    risk_appetite = "🦁 Risk-On" if hyg.iloc[-1] > hyg.rolling(20).mean().iloc[-1] else "🐻 Risk-Off"
    
    vix = data['^VIX'].iloc[-1]
    
    # 評分
    score = 0
    if dxy.iloc[-1] < dxy.rolling(20).mean().iloc[-1]: score += 1
    if tnx.iloc[-1] < tnx.rolling(20).mean().iloc[-1]: score += 1
    if hyg.iloc[-1] > hyg.rolling(20).mean().iloc[-1]: score += 1
    if vix < 20: score += 1
    
    regime = "🟢 綠燈 (積極)" if score >= 3 else ("🟡 黃燈 (謹慎)" if score == 2 else "🔴 紅燈 (現金)")
    
    return {'Regime': regime, 'Score': score, 'DXY': dxy_trend, 'TNX': tnx_trend, 'HYG': risk_appetite, 'Raw': data}

# ==========================================
# 2. 新聞爬蟲
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
# 3. 定價層
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
# 4. 回測層 (關鍵修正)
# ==========================================
def run_historical_validation(df_price, df_news_ticker, macro_data):
    df = df_price.copy()
    
    # A. 新聞對齊
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
        
    # B. 宏觀對齊
    macro_aligned = macro_data.reindex(df.index).ffill()
    macro_aligned['HYG_MA'] = macro_aligned['HYG'].rolling(20).mean()
    macro_aligned['Risk_On'] = macro_aligned['HYG'] > macro_aligned['HYG_MA']
    df = df.join(macro_aligned[['Risk_On']], how='left').fillna(False)
    
    # C. 技術指標
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_Slope'] = df['OBV'].diff(5)
    df['MA20'] = df['Close'].rolling(20).mean()
    vol_mean = df['Volume'].rolling(20).mean()
    vol_std = df['Volume'].rolling(20).std()
    df['Vol_Z'] = (df['Volume'] - vol_mean) / (vol_std + 1e-9)
    
    # D. 目標變數 (22天後漲跌)
    df['Ret_1M'] = df['Close'].shift(-22) / df['Close'] - 1
    
    # E. Alpha 訊號計算
    # 技術得分: 收盤 > 均線 = 1, 否則 -1
    tech_score = np.where(df['Close'] > df['MA20'], 1, -1)
    obv_score = np.sign(df['OBV_Slope']).fillna(0) # 修正: 填補 NaN
    
    df['Alpha_Raw'] = (df['News_Roll'] * 0.4) + (obv_score * 0.3) + (tech_score * 0.3)
    
    # --- 關鍵修正區：穩定計算方向準確度 ---
    # 1. 移除 NaN (前幾筆沒有 OBV 或是後幾筆沒有 Ret_1M 的)
    valid_rows = df.dropna(subset=['Ret_1M', 'Alpha_Raw'])
    
    if len(valid_rows) > 0:
        # 2. 不用 np.sign 比較，改用「相乘 > 0」邏輯 (同號相乘為正)
        # 避免浮點數與 NaN 的問題
        correct = (valid_rows['Alpha_Raw'] * valid_rows['Ret_1M']) > 0
        dir_acc = correct.mean()
    else:
        dir_acc = 0.5
    
    # --- Sniper Win Rate ---
    sniper_mask = (df['News_Roll'] > 0.1) & (df['OBV_Slope'] > 0) & (df['Vol_Z'] > 1.5) & (df['Risk_On'] == True)
    sniper_opps = df[sniper_mask].dropna(subset=['Ret_1M'])
    
    if len(sniper_opps) > 0:
        sniper_wins = sniper_opps[sniper_opps['Ret_1M'] > 0]
        sniper_win_rate = len(sniper_wins) / len(sniper_opps)
        sniper_count = len(sniper_opps)
        avg_ret = sniper_opps['Ret_1M'].mean()
    else:
        sniper_win_rate = 0.0
        sniper_count = 0
        avg_ret = 0.0
        
    return dir_acc, sniper_win_rate, sniper_count, avg_ret

# ==========================================
# 5. 主程式
# ==========================================
st.sidebar.title("控制台")
data_mode = st.sidebar.radio("數據來源", ["1. 即時爬取 (Live)", "2. 上傳 CSV"])
default_tickers = ["TSM", "NVDA", "AMD", "SOXL", "URA", "CLS"]
user_tickers = st.sidebar.text_area("代號", ", ".join(default_tickers))
ticker_list = [t.strip().upper() for t in user_tickers.split(',')]

# Macro
macro_info = fetch_macro_context()
st.subheader(f"🌍 天眼環境掃描: {macro_info['Regime']} (分數: {macro_info['Score']}/4)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("美元", macro_info['DXY'])
c2.metric("利率", macro_info['TNX'])
c3.metric("風險", macro_info['HYG'])
c4.metric("建議", "積極" if macro_info['Score']>=3 else "保守")
st.divider()

news_df = pd.DataFrame()
run = False

if data_mode.startswith("1"):
    if st.sidebar.button("🚀 啟動全系統"):
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
    up = st.sidebar.file_uploader("上傳 news_data.csv", type=['csv'])
    if up:
        try:
            temp = pd.read_csv(up)
            if 'Date' in temp.columns and 'Score' in temp.columns:
                news_df = temp
                news_df['Date'] = pd.to_datetime(news_df['Date'])
                run = st.sidebar.button("🚀 執行")
            else:
                st.error("❌ 檔案格式錯誤，請確認包含 Date, Score 欄位")
        except: st.error("讀檔失敗")

if run:
    st.sidebar.markdown("### 💾 數據保全")
    st.sidebar.download_button("📥 下載新聞 CSV", news_df.to_csv(index=False).encode('utf-8'), "news_data.csv")
    
    st.subheader("📊 天眼戰略報告")
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
        
        dir_acc, win_rate, count, avg_ret = run_historical_validation(df_price, df_news_t, macro_info['Raw'])
        target, _ = calc_4d_target(t, df_price)
        
        can_trade = macro_info['Score'] >= 2
        
        results.append({
            'Ticker': t,
            'Dir_Acc': dir_acc,
            'Sniper_Win': win_rate,
            'Sniper_Count': count,
            'Avg_Return': avg_ret,
            'Current': df_price['Close'].iloc[-1],
            'Target': target,
            'Upside': (target - df_price['Close'].iloc[-1]) / df_price['Close'].iloc[-1],
            'Regime': "PASS" if can_trade else "BLOCK"
        })
        
    res_df = pd.DataFrame(results)
    
    show = res_df.copy()
    show['Dir_Acc'] = show['Dir_Acc'].apply(lambda x: f"{x:.0%}")
    show['Sniper_Win'] = show['Sniper_Win'].apply(lambda x: f"{x:.0%}")
    show['Avg_Return'] = show['Avg_Return'].apply(lambda x: f"{x:+.1%}")
    show['Current'] = show['Current'].apply(lambda x: f"${x:.2f}")
    show['Target'] = show['Target'].apply(lambda x: f"${x:.2f}")
    show['Upside'] = show['Upside'].apply(lambda x: f"{x:+.1%}")

    st.dataframe(show[['Ticker', 'Dir_Acc', 'Sniper_Win', 'Sniper_Count', 'Avg_Return', 'Current', 'Target', 'Upside', 'Regime']].style.map(
        lambda x: 'background-color: #00FF7F; color: black' if 'PASS' in str(x) else 'background-color: #FF4B4B; color: white', subset=['Regime']
    ))
    
    fig = go.Figure()
    for i, row in res_df.iterrows():
        color = '#00FF7F' if row['Sniper_Win'] > 0.6 else '#FF4B4B'
        size = np.log(row['Sniper_Count'] + 1) * 15
        fig.add_trace(go.Scatter(
            x=[row['Dir_Acc']], y=[row['Sniper_Win']],
            mode='markers+text', text=[row['Ticker']],
            textposition="top center", marker=dict(size=size, color=color),
            name=row['Ticker'],
            hovertemplate="<b>%{text}</b><br>Dir Acc: %{x:.0%}<br>Win Rate: %{y:.0%}"
        ))
    fig.update_layout(title="模型效能矩陣", xaxis_title="方向準確度", yaxis_title="狙擊勝率", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)