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
import os
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 0. 頁面設定與本機檔案
# ==========================================
st.set_page_config(page_title="App 12.0 終極指揮官", layout="wide")
LOCAL_NEWS_FILE = "news_data_local.csv"

# 初始化 Session State
if 'news_data' not in st.session_state:
    if os.path.exists(LOCAL_NEWS_FILE):
        try:
            df_local = pd.read_csv(LOCAL_NEWS_FILE)
            if 'Date' in df_local.columns:
                df_local['Date'] = pd.to_datetime(df_local['Date'])
            st.session_state['news_data'] = df_local
        except: st.session_state['news_data'] = pd.DataFrame()
    else: st.session_state['news_data'] = pd.DataFrame()

st.title("🦅 App 12.0: 終極指揮官 (Macro + Fund + Quant + News)")
st.markdown("""
**集大成之作：**
1.  **宏觀天眼 (Macro)**：DXY/TNX/HYG 決定紅綠燈。
2.  **基本面修正 (Fund)**：財報數據修正目標價 (Scalar)。
3.  **雙重戰術回測**：同時驗證 **「順勢狙擊」** 與 **「逆勢抄底」** 勝率。
4.  **方向準確度 (Dir_Acc)**：檢驗模型長期預測能力。
""")

# ==========================================
# 1. 第一層：宏觀天眼 (Macro Regime)
# ==========================================
@st.cache_data(ttl=3600*4)
def fetch_macro_context():
    tickers = ['DX-Y.NYB', '^TNX', 'HYG', '^VIX']
    data = yf.download(tickers, period="1y", progress=False)['Close']
    
    # 趨勢
    dxy = data['DX-Y.NYB']
    dxy_ma = dxy.rolling(20).mean().iloc[-1]
    
    tnx = data['^TNX']
    tnx_ma = tnx.rolling(20).mean().iloc[-1]
    
    hyg = data['HYG']
    hyg_ma = hyg.rolling(20).mean().iloc[-1]
    
    vix = data['^VIX'].iloc[-1]
    
    # 評分 (Risk-On Score)
    score = 0
    if dxy.iloc[-1] < dxy_ma: score += 1      # 美元弱 -> 加分
    if tnx.iloc[-1] < tnx_ma: score += 1      # 利率降 -> 加分
    if hyg.iloc[-1] > hyg_ma: score += 1      # 聰明錢買債 -> 加分
    if vix < 20: score += 1                   # 不恐慌 -> 加分
    
    regime = "🟢 綠燈 (積極)" if score >= 3 else ("🟡 黃燈 (謹慎)" if score == 2 else "🔴 紅燈 (保守)")
    
    return {'Regime': regime, 'Score': score, 'Raw': data}

# ==========================================
# 2. 第二層：基本面純量 (Fundamental Scalar)
# ==========================================
@st.cache_data(ttl=3600*24)
def get_fundamental_scalar(ticker):
    """
    從 App 3.0 移植：根據財報修正目標價 (0.85 ~ 1.15)
    """
    try:
        if ticker in ['BTC-USD', 'URA', 'TLT', '0050.TW']: return 1.0, "ETF/Crypto" # 非個股跳過
        
        stock = yf.Ticker(ticker)
        info = stock.info
        fins = stock.quarterly_financials
        if fins.empty: return 1.0, "No Data"

        score = 0
        
        # A. 營收成長
        if 'Total Revenue' in fins.index and len(fins.columns) >= 2:
            r_now = fins.loc['Total Revenue'].iloc[0]
            r_prev = fins.loc['Total Revenue'].iloc[1]
            growth = (r_now - r_prev) / r_prev if r_prev != 0 else 0
            if growth > 0.10: score += 1
            elif growth < -0.05: score -= 1
            
        # B. 獲利能力
        if 'Net Income' in fins.index:
            ni = fins.loc['Net Income'].iloc[0]
            if ni > 0: score += 1
            else: score -= 1
            
        # C. P/E 檢查
        pe = info.get('trailingPE')
        if pe:
            if pe > 60: score -= 1 # 過熱
            elif pe < 15 and pe > 0: score += 1 # 價值
            
        scalar = 1.0 + (score * 0.05)
        return max(0.85, min(1.15, scalar)), f"Score: {score}"
        
    except: return 1.0, "Error"

# ==========================================
# 3. 數據層：全球新聞 (含本機存檔)
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
                    elif region == 'TW' and 'TW' in map_info:
                        if '漲' in title or '高' in title: score += 0.5
                    # ... 簡化邏輯以節省代碼空間
                    
                    if score != 0:
                        news_history.append({'Ticker': ticker, 'Date': pub_date, 'Region': region, 'Title': title, 'Score': score})
            except: pass
        current = next_month
        time.sleep(0.05)
    return pd.DataFrame(news_history)

# ==========================================
# 4. 第二層：四維定價 (Quant Engine)
# ==========================================
def train_rf_model(df, ticker):
    try:
        data = df[['Close']].copy()
        data['Ret'] = data['Close'].pct_change()
        data['Vol'] = data['Ret'].rolling(20).std()
        data['SMA'] = data['Close'].rolling(20).mean()
        data['Target'] = data['Close'].shift(-22) # 22天預測 (配合 Dir_Acc)
        data = data.dropna()
        if len(data) < 60: return None
        X = data[['Ret', 'Vol', 'SMA']]
        y = data['Target']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        last_row = data.iloc[[-1]][['Ret', 'Vol', 'SMA']]
        return model.predict(last_row)[0]
    except: return None

def calc_4d_target(ticker, df_price, scalar):
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
    # 套用基本面修正 (Scalar)
    final_target = avg_target * scalar
    return final_target

# ==========================================
# 5. 第三層：雙重回測 (Dir_Acc + Strategy)
# ==========================================
def run_dual_backtest(df_price, df_news_ticker, macro_data):
    df = df_price.copy()
    
    # 1. 整合新聞
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
        
    # 2. 整合宏觀 (Risk-On)
    macro_aligned = macro_data.reindex(df.index).ffill()
    macro_aligned['HYG_MA'] = macro_aligned['HYG'].rolling(20).mean()
    macro_aligned['Risk_On'] = macro_aligned['HYG'] > macro_aligned['HYG_MA']
    df = df.join(macro_aligned[['Risk_On']], how='left').fillna(False)

    # 3. 技術指標 (OBV, Vol_Z, Bias)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_Slope'] = df['OBV'].diff(5)
    
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Bias'] = (df['Close'] - df['MA20']) / df['MA20']
    
    vol_mean = df['Volume'].rolling(20).mean()
    vol_std = df['Volume'].rolling(20).std()
    df['Vol_Z'] = (df['Volume'] - vol_mean) / (vol_std + 1e-9)
    
    # 4. 目標變數 (22天回報，用於 Dir_Acc)
    df['Ret_1M'] = df['Close'].shift(-22) / df['Close'] - 1
    # 目標變數 (5天回報，用於戰術回測)
    df['Ret_5D'] = df['Close'].shift(-5) / df['Close'] - 1
    
    # --- Metric 1: Dir_Acc (長期方向) ---
    # 結合 News + OBV + Tech
    df['Alpha_Raw'] = (df['News_Roll'] * 0.4) + (np.sign(df['OBV_Slope']).fillna(0) * 0.3) + (np.where(df['Bias']>0, 1, -1) * 0.3)
    valid_dir = df.dropna(subset=['Ret_1M', 'Alpha_Raw'])
    if len(valid_dir) > 0:
        correct = (valid_dir['Alpha_Raw'] * valid_dir['Ret_1M']) > 0
        dir_acc = correct.mean()
    else: dir_acc = 0.5
    
    # --- Metric 2: Sniper Win (順勢狙擊) ---
    # 條件: News>0.1 & OBV>0 & Vol_Z>1.5 & Risk_On
    # 預測: 5天後漲
    sniper_mask = (df['News_Roll'] > 0.1) & (df['OBV_Slope'] > 0) & (df['Vol_Z'] > 1.5) & (df['Risk_On'] == True)
    sniper_opps = df[sniper_mask].dropna(subset=['Ret_5D'])
    if len(sniper_opps) > 0:
        sniper_win = len(sniper_opps[sniper_opps['Ret_5D'] > 0]) / len(sniper_opps)
    else: sniper_win = 0.0
    
    # --- Metric 3: Antifragile Win (逆勢抄底) ---
    # 條件: News<-0.1 (壞消息) & Bias<-0.05 (超賣)
    # 預測: 5天後漲 (反彈)
    anti_mask = (df['News_Roll'] < -0.1) & (df['Bias'] < -0.05)
    anti_opps = df[anti_mask].dropna(subset=['Ret_5D'])
    if len(anti_opps) > 0:
        anti_win = len(anti_opps[anti_opps['Ret_5D'] > 0]) / len(anti_opps)
    else: anti_win = 0.0
    
    # 回傳最後一天的指標供訊號判斷
    last_row = df.iloc[-1]
    last_metrics = {
        'News': last_row['News_Roll'],
        'OBV': last_row['OBV_Slope'],
        'Vol_Z': last_row['Vol_Z'],
        'Bias': last_row['Bias'],
        'Risk_On': last_row['Risk_On']
    }
    
    return dir_acc, sniper_win, anti_win, len(sniper_opps), len(anti_opps), last_metrics

# ==========================================
# 6. 主程式
# ==========================================
st.sidebar.title("控制台")
data_mode = st.sidebar.radio("數據來源", ["1. 優先使用本機/記憶體", "2. 強制重抓", "3. 上傳 CSV"])
default_tickers = ["TSM", "NVDA", "AMD", "SOXL", "URA", "CLS", "0050.TW"]
user_tickers = st.sidebar.text_area("代號", ", ".join(default_tickers))
ticker_list = [t.strip().upper() for t in user_tickers.split(',')]

# 宏觀看板
macro_info = fetch_macro_context()
st.subheader(f"🌍 宏觀天眼: {macro_info['Regime']} (Score: {macro_info['Score']})")
c1, c2, c3, c4 = st.columns(4)
c1.metric("美元 (Risk-Off)", f"{macro_info['Raw']['DX-Y.NYB'].iloc[-1]:.2f}")
c2.metric("殖利率 (Valuation)", f"{macro_info['Raw']['^TNX'].iloc[-1]:.2f}")
c3.metric("風險債 (Risk-On)", f"{macro_info['Raw']['HYG'].iloc[-1]:.2f}")
c4.metric("基本面修正", "啟動")
st.divider()

# 資料處理
if data_mode.startswith("2"):
    if st.sidebar.button("🚀 啟動爬蟲"):
        all_news = []
        bar = st.sidebar.progress(0)
        for i, t in enumerate(ticker_list):
            df = fetch_global_news_12m(t)
            if not df.empty: all_news.append(df)
            bar.progress((i+1)/len(ticker_list))
        if all_news:
            news_df = pd.concat(all_news, ignore_index=True)
            st.session_state['news_data'] = news_df
            news_df.to_csv(LOCAL_NEWS_FILE, index=False)
            st.sidebar.success("更新成功")

elif data_mode.startswith("3"):
    up = st.sidebar.file_uploader("上傳", type=['csv'])
    if up:
        try:
            temp = pd.read_csv(up)
            temp['Date'] = pd.to_datetime(temp['Date'])
            st.session_state['news_data'] = temp
            temp.to_csv(LOCAL_NEWS_FILE, index=False)
        except: st.error("失敗")

# 分析執行
if st.button("🚀 執行終極分析"):
    if st.session_state['news_data'].empty:
        st.error("無數據")
    else:
        st.sidebar.download_button("📥 下載新聞", st.session_state['news_data'].to_csv(index=False).encode('utf-8'), "news.csv")
        st.subheader("📊 終極戰略報告")
        
        news_df = st.session_state['news_data']
        results = []
        
        for t in ticker_list:
            # 1. 抓股價
            df_price = yf.download(t, period="2y", progress=False, auto_adjust=True)
            if isinstance(df_price.columns, pd.MultiIndex):
                temp = df_price['Close'][[t]].copy(); temp.columns = ['Close']
                temp['Volume'] = df_price['Volume'][t]
                temp['High'] = df_price['High'][t]
                temp['Low'] = df_price['Low'][t]
                df_price = temp
            else:
                df_price = df_price[['Close', 'Volume', 'High', 'Low']]
            
            # 2. 抓個股新聞
            df_news_t = news_df[news_df['Ticker'] == t].copy()
            
            # 3. 雙重回測
            dir_acc, sn_win, an_win, sn_cnt, an_cnt, metrics = run_dual_backtest(df_price, df_news_t, macro_info['Raw'])
            
            # 4. 基本面 Scalar
            scalar, fund_note = get_fundamental_scalar(t)
            
            # 5. 四維定價
            target = calc_4d_target(t, df_price, scalar)
            current = df_price['Close'].iloc[-1]
            
            # 6. 訊號生成 (整合 Macro)
            signal = "⬜ 觀望"
            # 必須 Risk-On 才能做狙擊
            if metrics['Risk_On'] and metrics['News'] > 0.1 and metrics['OBV'] > 0 and metrics['Vol_Z'] > 1.5:
                signal = "🎯 順勢狙擊"
            # 逆勢抄底不一定需要 Risk-On (因為是搶反彈)
            elif metrics['News'] < -0.1 and metrics['Bias'] < -0.05:
                signal = "💎 逆勢抄底"
            
            results.append({
                'Ticker': t,
                'Dir_Acc': dir_acc,
                'Sniper_Win': sn_win,
                'Anti_Win': an_win,
                'Current': current,
                'Target': target,
                'Upside': (target-current)/current,
                'Fund_Scalar': scalar,
                'Signal': signal,
                'Risk_On': "YES" if metrics['Risk_On'] else "NO"
            })
            
        res_df = pd.DataFrame(results)
        
        # 顯示
        show = res_df.copy()
        show['Dir_Acc'] = show['Dir_Acc'].apply(lambda x: f"{x:.0%}")
        show['Sniper_Win'] = show['Sniper_Win'].apply(lambda x: f"{x:.0%}")
        show['Anti_Win'] = show['Anti_Win'].apply(lambda x: f"{x:.0%}")
        show['Upside'] = show['Upside'].apply(lambda x: f"{x:+.1%}")
        show['Current'] = show['Current'].apply(lambda x: f"${x:.2f}")
        show['Target'] = show['Target'].apply(lambda x: f"${x:.2f}")
        show['Fund_Scalar'] = show['Fund_Scalar'].apply(lambda x: f"x{x:.2f}")

        st.dataframe(show[['Ticker', 'Signal', 'Dir_Acc', 'Sniper_Win', 'Anti_Win', 'Upside', 'Target', 'Fund_Scalar', 'Risk_On']].style.map(
            lambda x: 'background-color: #00FF7F; color: black' if '狙擊' in str(x) else ('background-color: #00BFFF; color: black' if '抄底' in str(x) else ''), 
            subset=['Signal']
        ))
        
        st.info("💡 Dir_Acc (方向準確度) 代表長期體質；Sniper/Anti Win 代表特定戰術勝率。請根據資產性格選擇戰術。")