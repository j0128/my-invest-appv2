import streamlit as st
import feedparser
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import random
import plotly.graph_objects as go

# ==========================================
# 0. 頁面設定 & 工具函數
# ==========================================
st.set_page_config(page_title="App 7.2 全能讀檔指揮官", layout="wide")

st.title("🦅 App 7.2: 全自動真實回測指揮官 (讀檔修復版)")
st.markdown("""
**修復說明：**
1. **強化 CSV 解析**：自動辨識 Ticker 欄位，兼容各種欄位格式。
2. **容錯機制**：若找不到金額欄位，將自動預設為 10 萬台幣，確保分析不中斷。
""")

# 獲取即時匯率 (USDTWD)
@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        df = yf.download("USDTWD=X", period="1d", progress=False)
        if not df.empty:
            return df['Close'].iloc[-1].item()
    except: pass
    return 32.5 # 預設備援

EXCHANGE_RATE = get_exchange_rate()
st.sidebar.metric("目前匯率 (USDTWD)", f"{EXCHANGE_RATE:.2f}")

# ==========================================
# 1. 檔案上傳與解析 (強化版)
# ==========================================
st.sidebar.header("📂 匯入資產")
uploaded_file = st.sidebar.file_uploader("上傳 CSV (需包含代號)", type=["csv"])

default_data = [
    {"Ticker": "TSM", "Value_NTD": 100000},
    {"Ticker": "NVDA", "Value_NTD": 100000},
    {"Ticker": "AMD", "Value_NTD": 100000}
]

MY_PORTFOLIO = []

if uploaded_file is not None:
    try:
        # 1. 先嘗試用 Header 讀取
        df_upload = pd.read_csv(uploaded_file)
        
        # 2. 欄位標準化 (轉大寫、去空白)
        df_upload.columns = [str(c).upper().strip() for c in df_upload.columns]
        
        # 3. 尋找代號欄位 (Ticker Column)
        ticker_col = None
        possible_names = ['TICKER', 'SYMBOL', 'CODE', 'STOCK', '代號', '股票']
        for col in df_upload.columns:
            if any(name in col for name in possible_names):
                ticker_col = col
                break
        
        # 如果找不到 Header，嘗試讀第一欄當作 Ticker
        if ticker_col is None:
            # 重讀，不設 header
            uploaded_file.seek(0)
            df_upload = pd.read_csv(uploaded_file, header=None)
            # 假設第一欄是代號，第二欄是金額
            df_upload.columns = ['TICKER_AUTO', 'VALUE_AUTO'] + [f'COL_{i}' for i in range(2, len(df_upload.columns))]
            ticker_col = 'TICKER_AUTO'
            
        # 4. 尋找金額欄位 (Value Column)
        value_col = None
        possible_values = ['VALUE', 'AMOUNT', 'COST', 'NTD', 'TWD', '市值', '金額', 'VALUE_NTD']
        for col in df_upload.columns:
            if any(name in col for name in possible_values):
                value_col = col
                break
                
        # 5. 建構資料
        clean_data = []
        for index, row in df_upload.iterrows():
            try:
                # 抓代號
                t = str(row[ticker_col]).upper().strip()
                # 排除無效代號 (如 index number, nan)
                if t == 'NAN' or t == '' or t.isdigit():
                    continue
                    
                # 抓金額
                v = 100000.0 # 預設值
                if value_col:
                    try:
                        raw_v = str(row[value_col]).replace(',', '').replace('$', '')
                        v = float(raw_v)
                    except: pass
                
                clean_data.append({"Ticker": t, "Value_NTD": v})
            except: continue
            
        if len(clean_data) > 0:
            MY_PORTFOLIO = clean_data
            st.sidebar.success(f"✅ 成功解析 {len(MY_PORTFOLIO)} 檔資產")
            if not value_col:
                st.sidebar.warning("⚠️ 未偵測到金額欄位，已預設為 10 萬台幣。")
        else:
            st.sidebar.error("❌ 無法辨識有效代號，請檢查 CSV。")
            MY_PORTFOLIO = default_data

    except Exception as e:
        st.sidebar.error(f"讀取失敗: {e}，使用預設資料。")
        MY_PORTFOLIO = default_data
else:
    st.sidebar.info("尚未上傳，使用預設範例。")
    MY_PORTFOLIO = default_data

# 顯示目前持倉預覽
with st.expander("查看目前持倉清單", expanded=True):
    preview_df = pd.DataFrame(MY_PORTFOLIO)
    preview_df['Est_Value_USD'] = preview_df['Value_NTD'] / EXCHANGE_RATE
    st.dataframe(preview_df)

# ==========================================
# 2. 真實歷史挖掘 (保留原功能)
# ==========================================
@st.cache_data(ttl=3600*12) 
def fetch_true_history(ticker, months=12):
    news_history = []
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=months)
    
    KEYWORDS = {
        'UP': ['beat', 'record', 'deal', 'partnership', 'approval', 'hike', 'surge', 'jump', 'buy', 'upgrade', 'bull'],
        'DOWN': ['miss', 'ban', 'restriction', 'probe', 'fraud', 'plunge', 'drop', 'cut', 'sell', 'downgrade', 'bear']
    }

    search_term = ticker
    if len(ticker) <= 4: search_term = f"{ticker} stock"

    current = start_date
    while current < end_date:
        next_month = current + relativedelta(months=1)
        d_after = current.strftime('%Y-%m-%d')
        d_before = next_month.strftime('%Y-%m-%d')
        rss_url = f"https://news.google.com/rss/search?q={search_term}+after:{d_after}+before:{d_before}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:5]: 
                title = entry.title
                score = TextBlob(title).sentiment.polarity
                t_lower = title.lower()
                for k in KEYWORDS['UP']: 
                    if k in t_lower: score += 0.4
                for k in KEYWORDS['DOWN']: 
                    if k in t_lower: score -= 0.4
                news_history.append({'Date': pd.to_datetime(entry.published).date(), 'Score': np.clip(score, -1, 1), 'Title': title})
        except: pass
        current = next_month
        time.sleep(0.05)
        
    if not news_history: return pd.DataFrame(columns=['Date', 'Score', 'Title'])
    df = pd.DataFrame(news_history)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# ==========================================
# 3. 戰略引擎 (保留原功能)
# ==========================================
STRATEGY_DB = {
    'TSM': {'Type': '機構型', 'W': {'Fund': 0.2, 'Tech': 0.2, 'News': 0.6}},
    'CLS': {'Type': '機構型', 'W': {'Fund': 0.5, 'Tech': 0.2, 'News': 0.3}},
    'NVDA': {'Type': '信仰型', 'W': {'Fund': 0.1, 'Tech': 0.7, 'News': 0.2}},
    'BTC-USD': {'Type': '信仰型', 'W': {'Fund': 0.0, 'Tech': 0.6, 'News': 0.4}},
    'SOXL': {'Type': '投機型', 'W': {'Fund': 0.1, 'Tech': 0.5, 'News': 0.4}},
    'AMD':  {'Type': '成長型', 'W': {'Fund': 0.3, 'Tech': 0.4, 'News': 0.3}},
    'DEFAULT': {'Type': '一般型', 'W': {'Fund': 0.33, 'Tech': 0.33, 'News': 0.33}}
}

def analyze_ticker(ticker, value_ntd):
    df_price = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    if isinstance(df_price.columns, pd.MultiIndex):
        temp = df_price['Close'][[ticker]].copy(); temp.columns = ['Close']
        df_price = temp
    else:
        df_price = df_price[['Close']]
    
    if df_price.empty: return None

    df_news = fetch_true_history(ticker, months=12)
    
    if not df_news.empty:
        daily_news = df_news.groupby('Date')['Score'].mean()
        df_price = df_price.join(daily_news, how='left').fillna(0)
        df_price['News_Factor'] = df_price['Score'].rolling(3).mean()
    else:
        df_price['News_Factor'] = 0
        
    df_price['MA200'] = df_price['Close'].rolling(200).mean()
    df_price['Bias'] = (df_price['Close'] - df_price['MA200']) / df_price['MA200']
    df_price['Score_F'] = -np.clip(df_price['Bias'] * 2, -1, 1) 
    
    df_price['MA20'] = df_price['Close'].rolling(20).mean()
    df_price['Score_T'] = np.where(df_price['Close'] > df_price['MA20'], 0.8, -0.8)
    
    strategy = STRATEGY_DB.get(ticker, STRATEGY_DB['DEFAULT'])
    w = strategy['W']
    
    df_price['Alpha_Score'] = (df_price['Score_F'] * w['Fund']) + (df_price['Score_T'] * w['Tech']) + (df_price['News_Factor'] * w['News'])
                              
    future_ret = df_price['Close'].shift(-20) - df_price['Close']
    valid_mask = (df_price.index > (datetime.now() - timedelta(days=365))) & (future_ret.notna())
    check_df = df_price[valid_mask]
    
    if not check_df.empty:
        hits = np.sign(check_df['Alpha_Score']) == np.sign(check_df['Close'].shift(-20) - check_df['Close'])
        dir_acc = hits.mean()
    else:
        dir_acc = 0.5
        
    current_price = df_price['Close'].iloc[-1]
    current_alpha = df_price['Alpha_Score'].iloc[-1]
    vol = df_price['Close'].pct_change().rolling(30).std().iloc[-1] * np.sqrt(30)
    
    target = current_price * (1 + current_alpha * 0.05)
    buy_zone = target * (1 - vol * 1.5)
    sell_zone = target * (1 + vol * 1.5)
    
    latest_news = df_news.iloc[-1]['Title'] if not df_news.empty else "無重大新聞"
    
    return {
        '代號': ticker, '類型': strategy['Type'], '方向準確度': dir_acc,
        '現價': current_price, '建議買點': buy_zone, '建議賣點': sell_zone,
        '最新情報': latest_news, 'Alpha值': current_alpha, '市值(NTD)': value_ntd
    }

# ==========================================
# 4. 執行按鈕
# ==========================================
if st.button("🚀 開始真實回測", type="primary"):
    results = []
    progress_bar = st.progress(0)
    status = st.empty()
    total = len(MY_PORTFOLIO)
    
    for i, item in enumerate(MY_PORTFOLIO):
        ticker = item['Ticker']
        val = item['Value_NTD']
        status.text(f"正在分析 {ticker} ... ({i+1}/{total})")
        try:
            res = analyze_ticker(ticker, val)
            if res: results.append(res)
        except Exception as e:
            st.error(f"{ticker} 失敗: {e}")
        progress_bar.progress((i+1)/total)
        
    status.text("✅ 全部分析完成")
    
    if results:
        df_res = pd.DataFrame(results)
        st.subheader("📊 戰略回測報告")
        
        show_df = df_res.copy()
        show_df['方向準確度'] = show_df['方向準確度'].apply(lambda x: f"{x:.0%}")
        show_df['現價'] = show_df['現價'].apply(lambda x: f"${x:.2f}")
        show_df['建議買點'] = show_df['建議買點'].apply(lambda x: f"${x:.2f}")
        show_df['建議賣點'] = show_df['建議賣點'].apply(lambda x: f"${x:.2f}")
        show_df['Alpha值'] = show_df['Alpha值'].apply(lambda x: f"{x:+.2f}")
        
        st.dataframe(show_df[['代號', '類型', '方向準確度', 'Alpha值', '現價', '建議買點', '建議賣點', '最新情報']].style.map(
            lambda x: 'background-color: #1f77b4; color: white' if isinstance(x, str) and '%' in x and int(x.strip('%')) > 60 else '',
            subset=['方向準確度']
        ))
        
        fig = go.Figure()
        for i, row in df_res.iterrows():
            upside = (row['建議賣點'] - row['現價']) / row['現價']
            acc = row['方向準確度']
            color = '#00FF7F' if acc > 0.6 else '#FF4B4B'
            size = np.log(row['市值(NTD)'] + 1) * 3
            fig.add_trace(go.Scatter(
                x=[acc], y=[upside], mode='markers+text', text=[row['代號']],
                textposition="top center", marker=dict(size=size, color=color, opacity=0.8),
                name=row['代號'], hovertemplate="<b>%{text}</b><br>勝率: %{x:.0%}<br>潛在漲幅: %{y:.1%}"
            ))
        fig.update_layout(
            title="<b>資產戰略矩陣</b>", xaxis_title="方向準確度", yaxis_title="潛在漲幅",
            template="plotly_dark", showlegend=False, height=500
        )
        fig.add_vline(x=0.6, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)