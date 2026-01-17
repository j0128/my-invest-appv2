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
st.set_page_config(page_title="App 6.0 羅盤指揮官", layout="wide")

st.title("🦅 App 6.0: 全自動羅盤指揮官 (含方向準確度)")
st.markdown("""
**升級重點：**
1. **方向準確度 (Dir. Acc)**：顯示模型預測漲跌的勝率。模糊的正確 > 精確的錯誤。
2. **短代碼修復**：修正 URA 等短代碼抓錯新聞的問題。
3. **戰略地圖**：結合價格誤差與方向勝率的綜合評估。
""")

# ==========================================
# 1. 您的資產輸入區
# ==========================================
st.subheader("1. 持倉設定")

default_data = pd.DataFrame([
    {"Ticker": "TSM", "Cost": 145.0},
    {"Ticker": "NVDA", "Cost": 120.0},
    {"Ticker": "AMD", "Cost": 160.0},
    {"Ticker": "SOXL", "Cost": 35.0},
    {"Ticker": "CLS", "Cost": 60.0},
    {"Ticker": "BTC-USD", "Cost": 65000.0},
    {"Ticker": "URA", "Cost": 30.0},
    {"Ticker": "META", "Cost": 580.0},
    {"Ticker": "TLT", "Cost": 95.0}
])

edited_df = st.data_editor(default_data, num_rows="dynamic")
MY_PORTFOLIO = dict(zip(edited_df['Ticker'], edited_df['Cost']))
HISTORY_MONTHS = 12 

# ==========================================
# 2. 核心功能
# ==========================================

@st.cache_data(ttl=3600)
def hack_historical_news(ticker, months):
    news_pool = []
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=months)
    
    KEYWORDS = {
        'BOOST': ['beat', 'record', 'deal', 'partnership', 'approval', 'hike', 'surge', 'jump', 'buy', 'upgrade'],
        'DRAG':  ['miss', 'ban', 'restriction', 'probe', 'fraud', 'plunge', 'drop', 'cut', 'sell', 'downgrade']
    }

    # 修復短代碼問題 (如 URA, CLS)
    search_ticker = ticker
    if len(ticker) <= 4 and "-" not in ticker:
        search_ticker = f"{ticker} stock" # 強制加上 stock 關鍵字

    current = start_date
    
    while current < end_date:
        next_month = current + relativedelta(months=1)
        d_after = current.strftime('%Y-%m-%d')
        d_before = next_month.strftime('%Y-%m-%d')
        
        rss_url = f"https://news.google.com/rss/search?q={search_ticker}+after:{d_after}+before:{d_before}&hl=en-US&gl=US&ceid=US:en"
        
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
        time.sleep(0.3) 
    
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

def analyze_asset_compass(ticker, cost_basis):
    # 下載數據 (2年)
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
                              
    # --- 回測核心：方向準確度 (Directional Accuracy) ---
    # 預測 30 天後的漲跌
    df_price['Pred_Target'] = df_price['Close'] * (1 + df_price['Alpha_Score'] * 0.05)
    
    valid_data = df_price.dropna()
    
    if len(valid_data) > 60:
        # 真實的 30 天變動 (未來 - 現在)
        real_move = valid_data['Close'] - valid_data['Close'].shift(30)
        # 預測的 30 天變動 (預測 - 現在)
        pred_move = valid_data['Pred_Target'].shift(30) - valid_data['Close'].shift(30)
        
        # 1. 計算方向準確度 (同號為 True)
        # 用最近 120 天 (半年) 的數據來算勝率
        matches = np.sign(real_move) == np.sign(pred_move)
        dir_acc = matches.tail(120).mean()
        
        # 2. 計算價格誤差 (MAPE)
        real_future = valid_data['Close']
        past_pred = valid_data['Pred_Target'].shift(30)
        error = (abs(real_future - past_pred) / real_future).tail(120).mean()
    else:
        dir_acc = 0.5 # 資料不足，跟丟銅板一樣
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
        'Current': current_price, 'PnL%': pnl_pct, 
        'Model_Error': error, 'Dir_Acc': dir_acc, # 新增指標
        'Latest_News': latest_news,
        'Score': current_alpha, 'Target': target_price,
        'Buy_Zone': box_low, 'Sell_Zone': box_high,
        'Action': '加碼' if current_price < box_low else ('獲利了結' if current_price > box_high else '續抱')
    }

# ==========================================
# 3. 執行介面
# ==========================================
st.subheader("2. 啟動指揮官")

if st.button("🚀 啟動羅盤掃描", type="primary"):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tickers = len(MY_PORTFOLIO)
    
    for i, (ticker, cost) in enumerate(MY_PORTFOLIO.items()):
        status_text.text(f"正在分析 {ticker} ({i+1}/{total_tickers}) - 掃描方向性中...")
        try:
            data = analyze_asset_compass(ticker, cost)
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
        
        st.subheader("📊 Alpha 32 戰略地圖 (含方向勝率)")
        
        # 顯示主要表格
        display_df = df_res[['Ticker', 'Type', 'Dir_Acc', 'Model_Error', 'Current', 'Buy_Zone', 'Target', 'Sell_Zone', 'Action']].copy()
        
        # 格式化顯示
        display_df['Dir_Acc'] = display_df['Dir_Acc'].apply(lambda x: f"{x:.0%}") # 勝率
        display_df['Model_Error'] = display_df['Model_Error'].apply(lambda x: f"{x:.1%}")
        
        for col in ['Current', 'Target', 'Buy_Zone', 'Sell_Zone']:
            display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
            
        # 使用顏色標記勝率
        st.dataframe(display_df.style.map(
            lambda x: 'color: lightgreen' if isinstance(x, str) and '%' in x and float(x.strip('%')) > 60 else '', 
            subset=['Dir_Acc']
        ))
        
        st.markdown("""
        **指標解讀：**
        * **方向勝率 (Dir_Acc)**：越高越好。**>60%** 代表模型對這檔股票的漲跌判斷很有優勢。
        * **價格誤差 (Model_Error)**：越低越好。代表預測點位精準。
        """)
        
        # 繪製 Plotly 圖表 (氣泡圖：勝率 vs 誤差)
        fig_bubble = go.Figure()
        
        for i, row in df_res.iterrows():
            # 氣泡顏色：綠色=高勝率，紅色=低勝率
            color = '#00FF7F' if row['Dir_Acc'] > 0.6 else '#FF4B4B'
            
            fig_bubble.add_trace(go.Scatter(
                x=[row['Model_Error']], 
                y=[row['Dir_Acc']],
                mode='markers+text',
                text=[row['Ticker']],
                textposition="top center",
                marker=dict(size=30, color=color, opacity=0.7),
                name=row['Ticker'],
                hovertemplate="<b>%{text}</b><br>方向勝率: %{y:.0%}<br>價格誤差: %{x:.1%}"
            ))

        fig_bubble.update_layout(
            title="<b>模型可信度矩陣</b> (右上角=危險, 左上角=聖杯)",
            xaxis_title="價格誤差 (越左越好)",
            yaxis_title="方向勝率 (越上越好)",
            xaxis=dict(autorange="reversed"), # 讓誤差小的在右邊 (或維持原樣，越左越小) -> 這裡我讓越左越小比較直觀
            template="plotly_dark",
            showlegend=False,
            height=400
        )
        # 畫十字線 (60% 勝率 / 15% 誤差)
        fig_bubble.add_hline(y=0.6, line_dash="dash", line_color="gray", annotation_text="及格線 (60%)")
        fig_bubble.add_vline(x=0.15, line_dash="dash", line_color="gray", annotation_text="精準線 (15%)")
        
        st.plotly_chart(fig_bubble, use_container_width=True)
        
        # 顯示詳細新聞
        with st.expander("📰 查看新聞來源"):
            for res in results:
                st.markdown(f"**{res['Ticker']}**: {res['Latest_News']}")