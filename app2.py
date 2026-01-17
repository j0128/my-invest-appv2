import streamlit as st
import feedparser
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from datetime import datetime, timedelta  # <--- 修正點：補上 timedelta
from dateutil.relativedelta import relativedelta
import time
import random
import plotly.graph_objects as go

# ==========================================
# 0. 頁面設定 & 工具函數
# ==========================================
st.set_page_config(page_title="App 7.1 真實回測指揮官", layout="wide")

st.title("🦅 App 7.1: 全自動真實回測指揮官 (修復版)")
st.markdown("""
**修復說明：**
1. **修正 timedelta 錯誤**：解決了 CLS 等股票運算失敗的問題。
2. **真實爬蟲核心**：程式會逐月挖掘過去 12 個月的 Google News RSS，絕無模擬。
3. **CSV 匯入**：支援上傳資產表 (代號 + 台幣市值)。
""")

# 獲取即時匯率 (USDTWD)
@st.cache_data(ttl=3600)
def get_exchange_rate():
    try:
        df = yf.download("USDTWD=X", period="1d", progress=False)
        return df['Close'].iloc[-1].item()
    except:
        return 32.5 # 預設備援

EXCHANGE_RATE = get_exchange_rate()
st.sidebar.metric("目前匯率 (USDTWD)", f"{EXCHANGE_RATE:.2f}")

# ==========================================
# 1. 檔案上傳與解析
# ==========================================
st.sidebar.header("📂 匯入資產")
uploaded_file = st.sidebar.file_uploader("上傳 CSV (A欄:代號, B欄:台幣市值)", type=["csv"])

default_data = [
    {"Ticker": "TSM", "Value_NTD": 100000},
    {"Ticker": "NVDA", "Value_NTD": 100000},
    {"Ticker": "AMD", "Value_NTD": 100000}
]

MY_PORTFOLIO = default_data # 預設

if uploaded_file is not None:
    try:
        # 嘗試讀取 CSV (假設無 header)
        df_upload = pd.read_csv(uploaded_file, header=None)
        
        # 簡單判斷：如果第一列看起來像標題，就重讀
        first_val = str(df_upload.iloc[0, 0])
        if len(first_val) > 0 and not first_val[0].isdigit():
            df_upload = pd.read_csv(uploaded_file)
            # 強制重新命名欄位，避免使用者標題打錯
            df_upload.columns = ["Ticker", "Value_NTD"]
        else:
            df_upload.columns = ["Ticker", "Value_NTD"]
            
        # 清理數據
        df_upload['Ticker'] = df_upload['Ticker'].astype(str).str.upper().str.strip()
        # 處理金額 (移除逗號等)
        df_upload['Value_NTD'] = pd.to_numeric(df_upload['Value_NTD'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        
        MY_PORTFOLIO = df_upload.to_dict('records')
        st.sidebar.success(f"✅ 成功讀取 {len(MY_PORTFOLIO)} 檔資產")
        
    except Exception as e:
        st.sidebar.error(f"讀取失敗: {e}，將使用預設資料。")

# 顯示目前持倉預覽
with st.expander("查看目前持倉清單", expanded=True):
    preview_df = pd.DataFrame(MY_PORTFOLIO)
    # 換算 USD 估值
    preview_df['Est_Value_USD'] = preview_df['Value_NTD'] / EXCHANGE_RATE
    st.dataframe(preview_df)

# ==========================================
# 2. 核心：真實歷史新聞挖掘 (RSS Hacker)
# ==========================================
@st.cache_data(ttl=3600*12) 
def fetch_true_history(ticker, months=12):
    """
    真正的回測核心：逐月爬取歷史新聞
    """
    news_history = []
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=months)
    
    # 強力關鍵字 (NLP 修正)
    KEYWORDS = {
        'UP': ['beat', 'record', 'deal', 'partnership', 'approval', 'hike', 'surge', 'jump', 'buy', 'upgrade', 'bull', 'growth'],
        'DOWN': ['miss', 'ban', 'restriction', 'probe', 'fraud', 'plunge', 'drop', 'cut', 'sell', 'downgrade', 'bear', 'warn']
    }

    # 針對短代碼優化
    search_term = ticker
    if len(ticker) <= 4: search_term = f"{ticker} stock"

    current = start_date
    
    # 為了不讓使用者等太久，我們這裡稍微優化一下間隔，但保持真實請求
    while current < end_date:
        next_month = current + relativedelta(months=1)
        d_after = current.strftime('%Y-%m-%d')
        d_before = next_month.strftime('%Y-%m-%d')
        
        rss_url = f"https://news.google.com/rss/search?q={search_term}+after:{d_after}+before:{d_before}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(rss_url)
            # 取該月前 5 條
            for entry in feed.entries[:5]: 
                title = entry.title
                
                score = TextBlob(title).sentiment.polarity
                
                t_lower = title.lower()
                for k in KEYWORDS['UP']: 
                    if k in t_lower: score += 0.4
                for k in KEYWORDS['DOWN']: 
                    if k in t_lower: score -= 0.4
                
                news_history.append({
                    'Date': pd.to_datetime(entry.published).date(),
                    'Score': np.clip(score, -1, 1),
                    'Title': title
                })
        except: pass
        
        current = next_month
        time.sleep(0.05) # 微小延遲
        
    if not news_history:
        return pd.DataFrame(columns=['Date', 'Score', 'Title'])
        
    df = pd.DataFrame(news_history)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# ==========================================
# 3. 戰略引擎 (Alpha 32)
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
    # 1. 抓股價
    df_price = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    if isinstance(df_price.columns, pd.MultiIndex):
        temp = df_price['Close'][[ticker]].copy(); temp.columns = ['Close']
        df_price = temp
    else:
        df_price = df_price[['Close']]
    
    if df_price.empty: return None

    # 2. 抓真實歷史新聞
    df_news = fetch_true_history(ticker, months=12)
    
    # 3. 合併數據
    if not df_news.empty:
        daily_news = df_news.groupby('Date')['Score'].mean()
        df_price = df_price.join(daily_news, how='left').fillna(0)
        df_price['News_Factor'] = df_price['Score'].rolling(3).mean()
    else:
        df_price['News_Factor'] = 0
        
    # 4. 計算因子
    df_price['MA200'] = df_price['Close'].rolling(200).mean()
    df_price['Bias'] = (df_price['Close'] - df_price['MA200']) / df_price['MA200']
    df_price['Score_F'] = -np.clip(df_price['Bias'] * 2, -1, 1) 
    
    df_price['MA20'] = df_price['Close'].rolling(20).mean()
    df_price['Score_T'] = np.where(df_price['Close'] > df_price['MA20'], 0.8, -0.8)
    
    # 5. 加權
    strategy = STRATEGY_DB.get(ticker, STRATEGY_DB['DEFAULT'])
    w = strategy['W']
    
    df_price['Alpha_Score'] = (df_price['Score_F'] * w['Fund']) + \
                              (df_price['Score_T'] * w['Tech']) + \
                              (df_price['News_Factor'] * w['News'])
                              
    # 6. 計算方向準確度 (這是真實回測的關鍵)
    # 我們看過去一年，每一天的 Alpha Score 是否正確預測了未來 20 天的漲跌
    future_ret = df_price['Close'].shift(-20) - df_price['Close']
    
    # 這裡用到了 timedelta，之前報錯就是這裡
    valid_mask = (df_price.index > (datetime.now() - timedelta(days=365))) & (future_ret.notna())
    check_df = df_price[valid_mask]
    
    if not check_df.empty:
        hits = np.sign(check_df['Alpha_Score']) == np.sign(check_df['Close'].shift(-20) - check_df['Close'])
        dir_acc = hits.mean()
    else:
        dir_acc = 0.5
        
    # 7. 生成預測
    current_price = df_price['Close'].iloc[-1]
    current_alpha = df_price['Alpha_Score'].iloc[-1]
    vol = df_price['Close'].pct_change().rolling(30).std().iloc[-1] * np.sqrt(30)
    
    target = current_price * (1 + current_alpha * 0.05)
    buy_zone = target * (1 - vol * 1.5)
    sell_zone = target * (1 + vol * 1.5)
    
    latest_news = df_news.iloc[-1]['Title'] if not df_news.empty else "無重大新聞"
    
    return {
        '代號': ticker,
        '類型': strategy['Type'],
        '方向準確度': dir_acc,
        '現價': current_price,
        '建議買點': buy_zone,
        '建議賣點': sell_zone,
        '最新情報': latest_news,
        'Alpha值': current_alpha,
        '市值(NTD)': value_ntd
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
        status.text(f"正在深入挖掘 {ticker} 過去 12 個月的所有新聞... ({i+1}/{total})")
        
        try:
            res = analyze_ticker(ticker, val)
            if res: results.append(res)
        except Exception as e:
            st.error(f"{ticker} 失敗: {e}")
            
        progress_bar.progress((i+1)/total)
        
    status.text("✅ 全部分析完成")
    
    if results:
        df_res = pd.DataFrame(results)
        
        # 1. 核心報表
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
        
        # 2. 戰略氣泡圖
        fig = go.Figure()
        
        for i, row in df_res.iterrows():
            upside = (row['建議賣點'] - row['現價']) / row['現價']
            acc = row['方向準確度']
            
            color = '#00FF7F' if acc > 0.6 else '#FF4B4B'
            size = np.log(row['市值(NTD)'] + 1) * 3 # 氣泡大小根據持倉
            
            fig.add_trace(go.Scatter(
                x=[acc], y=[upside],
                mode='markers+text',
                text=[row['代號']],
                textposition="top center",
                marker=dict(size=size, color=color, opacity=0.8),
                name=row['代號'],
                hovertemplate="<b>%{text}</b><br>勝率: %{x:.0%}<br>潛在漲幅: %{y:.1%}"
            ))
            
        fig.update_layout(
            title="<b>資產戰略矩陣</b> (右上=高勝率高潛力)",
            xaxis_title="方向準確度 (歷史勝率)",
            yaxis_title="潛在漲幅 (到賣點的距離)",
            template="plotly_dark",
            showlegend=False,
            height=500
        )
        fig.add_vline(x=0.6, line_dash="dash", annotation_text="及格線 (60%)")
        st.plotly_chart(fig, use_container_width=True)