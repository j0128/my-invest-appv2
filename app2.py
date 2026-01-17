import streamlit as st
import feedparser
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from scipy import stats
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import random
import plotly.graph_objects as go

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(page_title="App 5.0 全自動情報指揮官", layout="wide")

st.title("🦅 App 5.0: 全自動真實情報指揮官 (12個月深度版)")
st.markdown("""
**功能說明：**
1. **深度考古**：現場挖掘過去 12 個月的真實新聞 (Google RSS)。
2. **智能權重**：針對個股性格套用 Alpha 32 最佳權重。
3. **安全啟動**：請在下方編輯持倉，確認無誤後點擊按鈕開始。
""")

# ==========================================
# 1. 您的資產輸入區 (互動式表格)
# ==========================================
st.subheader("1. 持倉設定 (請直接修改表格)")

# 預設持倉數據
default_data = pd.DataFrame([
    {"Ticker": "TSM", "Cost": 145.0},
    {"Ticker": "NVDA", "Cost": 120.0},
    {"Ticker": "AMD", "Cost": 160.0},
    {"Ticker": "SOXL", "Cost": 35.0},
    {"Ticker": "CLS", "Cost": 60.0},
    {"Ticker": "BTC-USD", "Cost": 65000.0}
])

# 讓使用者在網頁上編輯
edited_df = st.data_editor(default_data, num_rows="dynamic")

# 轉換回字典格式供程式使用
MY_PORTFOLIO = dict(zip(edited_df['Ticker'], edited_df['Cost']))
HISTORY_MONTHS = 12 

# ==========================================
# 2. 核心功能函式庫
# ==========================================

@st.cache_data(ttl=3600) # 加入快取，避免重複抓取浪費時間
def hack_historical_news(ticker, months):
    """RSS 歷史新聞駭客"""
    # st.write(f"  ⛏️ [RSS Hacker] 正在挖掘 {ticker}...")
    news_pool = []
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=months)
    
    KEYWORDS = {
        'BOOST': ['beat', 'record', 'deal', 'partnership', 'approval', 'hike', 'surge', 'jump', 'buy', 'upgrade'],
        'DRAG':  ['miss', 'ban', 'restriction', 'probe', 'fraud', 'plunge', 'drop', 'cut', 'sell', 'downgrade']
    }

    current = start_date
    progress_text = f"正在掃描 {ticker} 歷史新聞..."
    # 這裡不顯示進度條以免畫面太亂，改用後台執行
    
    while current < end_date:
        next_month = current + relativedelta(months=1)
        d_after = current.strftime('%Y-%m-%d')
        d_before = next_month.strftime('%Y-%m-%d')
        
        rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+after:{d_after}+before:{d_before}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                title = entry.title
                base_score = TextBlob(title).sentiment.polarity
                
                boost = 0
                t_lower = title.lower()
                for k in KEYWORDS['BOOST']: 
                    if k in t_lower: boost += 0.3
                for k in KEYWORDS['DRAG']: 
                    if k in t_lower: boost -= 0.3
                
                final_score = np.clip(base_score + boost, -1, 1)
                
                news_pool.append({
                    'Date': pd.to_datetime(entry.published).date(),
                    'Title': title,
                    'Score': final_score
                })
        except: pass
        current = next_month
        # 在 Streamlit 中稍微減少延遲，因為會並行處理
        time.sleep(0.5) 
    
    if not news_pool:
        return pd.DataFrame(columns=['Date', 'Title', 'Score'])
    
    df = pd.DataFrame(news_pool)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

STRATEGY_DB = {
    'TSM': {'Type': '機構型', 'W': {'Fund': 0.2, 'Tech': 0.2, 'News': 0.6}},
    'CLS': {'Type': '機構型', 'W': {'Fund': 0.5, 'Tech': 0.2, 'News': 0.3}},
    'NVDA': {'Type': '信仰型', 'W': {'Fund': 0.1, 'Tech': 0.7, 'News': 0.2}},
    'BTC-USD': {'Type': '信仰型', 'W': {'Fund': 0.0, 'Tech': 0.6, 'News': 0.4}},
    'SOXL': {'Type': '投機型', 'W': {'Fund': 0.1, 'Tech': 0.5, 'News': 0.4}},
    'AMD':  {'Type': '成長型', 'W': {'Fund': 0.3, 'Tech': 0.4, 'News': 0.3}},
    'DEFAULT': {'Type': '一般型', 'W': {'Fund': 0.33, 'Tech': 0.33, 'News': 0.33}}
}

def analyze_asset_full_auto(ticker, cost_basis):
    # 下載股價
    df_price = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    if isinstance(df_price.columns, pd.MultiIndex):
        temp = df_price['Close'][[ticker]].copy(); temp.columns = ['Close']
        df_price = temp
    else:
        df_price = df_price[['Close']]
    
    # 抓取新聞
    df_news = hack_historical_news(ticker, HISTORY_MONTHS)
    
    if not df_news.empty:
        daily_news = df_news.groupby('Date')['Score'].mean()
        df_price = df_price.join(daily_news, how='left').fillna(0)
        df_price['News_Factor'] = df_price['Score'].rolling(3).mean()
    else:
        df_price['News_Factor'] = 0
    
    # 計算因子
    df_price['MA200'] = df_price['Close'].rolling(200).mean()
    df_price['Bias'] = (df_price['Close'] - df_price['MA200']) / df_price['MA200']
    df_price['Score_F'] = -np.clip(df_price['Bias'] * 2, -1, 1) 
    
    df_price['MA20'] = df_price['Close'].rolling(20).mean()
    df_price['Score_T'] = np.where(df_price['Close'] > df_price['MA20'], 0.8, -0.8)
    
    # 權重
    strategy = STRATEGY_DB.get(ticker, STRATEGY_DB['DEFAULT'])
    w = strategy['W']
    
    df_price['Alpha_Score'] = (df_price['Score_F'] * w['Fund']) + \
                              (df_price['Score_T'] * w['Tech']) + \
                              (df_price['News_Factor'] * w['News'])
                              
    # 回測誤差
    df_price['Pred_Target'] = df_price['Close'] * (1 + df_price['Alpha_Score'] * 0.05)
    valid_data = df_price.dropna()
    if len(valid_data) > 60:
        real_future = valid_data['Close']
        past_pred = valid_data['Pred_Target'].shift(30)
        error = (abs(real_future - past_pred) / real_future).tail(120).mean()
    else:
        error = 0.20
        
    current_price = df_price['Close'].iloc[-1]
    current_alpha = df_price['Alpha_Score'].iloc[-1]
    vol = df_price['Close'].pct_change().rolling(30).std().iloc[-1] * np.sqrt(30)
    
    target_price = current_price * (1 + current_alpha * 0.05)
    box_high = target_price * (1 + vol * 1.5)
    box_low = target_price * (1 - vol * 1.5)
    pnl_pct = (current_price - cost_basis) / cost_basis
    
    latest_news = df_news.iloc[-1]['Title'] if not df_news.empty else "無近期新聞"
    
    return {
        'Ticker': ticker, 'Type': strategy['Type'], 'Cost': cost_basis,
        'Current': current_price, 'PnL%': pnl_pct, 'Model_Error': error,
        'Latest_News': latest_news,
        'Score': current_alpha, 'Target': target_price,
        'Buy_Zone': box_low, 'Sell_Zone': box_high,
        'Action': '加碼' if current_price < box_low else ('獲利了結' if current_price > box_high else '續抱')
    }

# ==========================================
# 3. 執行介面
# ==========================================
st.subheader("2. 啟動指揮官")
st.info("⚠️ 注意：程式將連線 Google 抓取大量數據，每檔股票約需 5-10 秒，請耐心等待。")

if st.button("🚀 開始全域掃描", type="primary"):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tickers = len(MY_PORTFOLIO)
    
    for i, (ticker, cost) in enumerate(MY_PORTFOLIO.items()):
        status_text.text(f"正在分析 {ticker} ({i+1}/{total_tickers})...")
        try:
            data = analyze_asset_full_auto(ticker, cost)
            results.append(data)
        except Exception as e:
            st.error(f"❌ {ticker} 分析失敗: {e}")
        
        progress_bar.progress((i + 1) / total_tickers)
        
    status_text.text("✅ 分析完成！")
    
    # ==========================================
    # 4. 結果展示
    # ==========================================
    if results:
        df_res = pd.DataFrame(results)
        
        st.subheader("📊 Alpha 32 真實戰略地圖")
        
        # 顯示主要表格
        display_df = df_res[['Ticker', 'Type', 'Model_Error', 'Current', 'Target', 'Buy_Zone', 'Sell_Zone', 'Action']].copy()
        
        # 格式化顯示
        for col in ['Current', 'Target', 'Buy_Zone', 'Sell_Zone']:
            display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
        display_df['Model_Error'] = display_df['Model_Error'].apply(lambda x: f"{x:.1%}")
        
        st.table(display_df)
        
        # 繪製 Plotly 圖表
        fig = go.Figure()
        for i, row in df_res.iterrows():
            color = 'cyan' if row['PnL%'] > 0 else 'red'
            
            # 戰略箱體
            fig.add_trace(go.Box(
                y=[row['Buy_Zone'], row['Target'], row['Target'], row['Sell_Zone']],
                name=f"{row['Ticker']}",
                marker_color=color,
                boxpoints=False,
                hoverinfo='y+name'
            ))
            
            # 成本線
            fig.add_trace(go.Scatter(
                x=[row['Ticker']], y=[row['Cost']],
                mode='markers', marker=dict(symbol='line-ew', size=50, color='white', line=dict(width=3)),
                name='成本價'
            ))
            
            # 現價
            fig.add_trace(go.Scatter(
                x=[row['Ticker']], y=[row['Current']],
                mode='markers', marker=dict(symbol='diamond', size=12, color='yellow'),
                name='現價'
            ))

        fig.update_layout(
            title="戰略區間分佈 (箱體=預測 | 白線=成本 | 黃鑽=現價)",
            template="plotly_dark",
            yaxis_title="價格 (USD)",
            height=600,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 顯示詳細新聞資訊
        with st.expander("📰 查看最新情報來源"):
            for res in results:
                st.markdown(f"**{res['Ticker']}**: {res['Latest_News']}")