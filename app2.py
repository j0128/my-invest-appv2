# @title 🦅 App 5.0: 全自動真實情報指揮官 (12個月深度版)
# @markdown **功能升級：**<br>1. **深度考古**：鎖定挖掘過去 12 個月的真實新聞。<br>2. **安全啟動**：等待您確認持倉後才開始執行。<br>3. **真實回測**：用一整年的數據驗證 Alpha 32 準確度。

# ==========================================
# 1. 您的資產輸入區 (請在此修改)
# ==========================================
MY_PORTFOLIO = {
    # 格式: '股票代號': 您的成本價
    'TSM':  145.0,  
    'NVDA': 120.0,
    'AMD':  160.0,
    'SOXL': 35.0,
    'CLS':  60.0,
    'BTC-USD': 65000.0
}

# 設定回測新聞長度 (月)
HISTORY_MONTHS = 12 

# ==========================================
# (以下為系統核心，無需修改)
# ==========================================

# 0. 環境準備與安全啟動
try:
    import feedparser
    import textblob
    import tabulate
except ImportError:
    print("正在安裝組件...")
    !pip install feedparser textblob tabulate -q

import feedparser
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from scipy import stats
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import time
import random
import plotly.graph_objects as go

# 等待使用者確認
print(f"📋 目前設定的持倉清單: {list(MY_PORTFOLIO.keys())}")
print(f"🕒 預計抓取新聞長度: {HISTORY_MONTHS} 個月")
input("⚠️ 請確認上方 `MY_PORTFOLIO` 已修改完畢。準備好後，請點擊此處並按 [Enter] 鍵開始執行...")

# 1. RSS 歷史駭客
def hack_historical_news(ticker, months):
    print(f"  ⛏️ [RSS Hacker] 正在挖掘 {ticker} 過去 {months} 個月的真實新聞...", end=" ")
    news_pool = []
    
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=months)
    
    # 關鍵字加權字典
    KEYWORDS = {
        'BOOST': ['beat', 'record', 'deal', 'partnership', 'approval', 'hike', 'surge', 'jump', 'buy', 'upgrade'],
        'DRAG':  ['miss', 'ban', 'restriction', 'probe', 'fraud', 'plunge', 'drop', 'cut', 'sell', 'downgrade']
    }

    current = start_date
    count = 0
    
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
                count += 1
        except: pass
        
        current = next_month
        # 隨機延遲 1.5 ~ 3 秒，避免跑 12 個月被 Google 封鎖
        time.sleep(random.uniform(1.5, 3.0))
    
    print(f"✅ 捕獲 {count} 條。")
    if not news_pool:
        return pd.DataFrame(columns=['Date', 'Title', 'Score'])
    
    df = pd.DataFrame(news_pool)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# 2. Alpha 32 權重配置
STRATEGY_DB = {
    'TSM': {'Type': '機構型', 'W': {'Fund': 0.2, 'Tech': 0.2, 'News': 0.6}},
    'CLS': {'Type': '機構型', 'W': {'Fund': 0.5, 'Tech': 0.2, 'News': 0.3}},
    'NVDA': {'Type': '信仰型', 'W': {'Fund': 0.1, 'Tech': 0.7, 'News': 0.2}},
    'BTC-USD': {'Type': '信仰型', 'W': {'Fund': 0.0, 'Tech': 0.6, 'News': 0.4}},
    'SOXL': {'Type': '投機型', 'W': {'Fund': 0.1, 'Tech': 0.5, 'News': 0.4}},
    'AMD':  {'Type': '成長型', 'W': {'Fund': 0.3, 'Tech': 0.4, 'News': 0.3}},
    'DEFAULT': {'Type': '一般型', 'W': {'Fund': 0.33, 'Tech': 0.33, 'News': 0.33}}
}

# 3. 核心運算引擎
def analyze_asset_full_auto(ticker, cost_basis):
    # 下載股價 (包含過去 18 個月以配合 12 個月新聞 + 指標運算)
    df_price = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    if isinstance(df_price.columns, pd.MultiIndex):
        temp = df_price['Close'][[ticker]].copy(); temp.columns = ['Close']
        df_price = temp
    else:
        df_price = df_price[['Close']]
    
    # 現場抓取真實歷史新聞
    df_news = hack_historical_news(ticker, HISTORY_MONTHS)
    
    if not df_news.empty:
        daily_news = df_news.groupby('Date')['Score'].mean()
        df_price = df_price.join(daily_news, how='left').fillna(0)
        df_price['News_Factor'] = df_price['Score'].rolling(3).mean()
    else:
        df_price['News_Factor'] = 0
    
    # F: 基本面
    df_price['MA200'] = df_price['Close'].rolling(200).mean()
    df_price['Bias'] = (df_price['Close'] - df_price['MA200']) / df_price['MA200']
    df_price['Score_F'] = -np.clip(df_price['Bias'] * 2, -1, 1) 
    
    # T: 技術面
    df_price['MA20'] = df_price['Close'].rolling(20).mean()
    df_price['Score_T'] = np.where(df_price['Close'] > df_price['MA20'], 0.8, -0.8)
    
    # 套用權重
    strategy = STRATEGY_DB.get(ticker, STRATEGY_DB['DEFAULT'])
    w = strategy['W']
    
    df_price['Alpha_Score'] = (df_price['Score_F'] * w['Fund']) + \
                              (df_price['Score_T'] * w['Tech']) + \
                              (df_price['News_Factor'] * w['News'])
                              
    # 回測誤差 (使用過去 12 個月的數據)
    df_price['Pred_Target'] = df_price['Close'] * (1 + df_price['Alpha_Score'] * 0.05)
    
    valid_data = df_price.dropna()
    if len(valid_data) > 60:
        real_future = valid_data['Close']
        past_pred = valid_data['Pred_Target'].shift(30)
        # 計算最近 6 個月的平均誤差
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
        'Latest_News': latest_news[:30] + "...",
        'Score': current_alpha, 'Target': target_price,
        'Buy_Zone': box_low, 'Sell_Zone': box_high,
        'Action': '加碼' if current_price < box_low else ('獲利了結' if current_price > box_high else '續抱')
    }

# 4. 執行
print("\n🦅 App 5.0: 啟動全自動真實情報掃描...")
print("---------------------------------------------------------------")
portfolio_results = []

for t, c in MY_PORTFOLIO.items():
    try:
        data = analyze_asset_full_auto(t, c)
        portfolio_results.append(data)
    except Exception as e:
        print(f"❌ {t} 失敗: {e}")

# 5. 儀表板
if portfolio_results:
    df_res = pd.DataFrame(portfolio_results)
    
    print("\n📊 === Alpha 32 真實戰略地圖 (12個月回測版) ===")
    fmt_df = df_res.copy()
    for col in ['Current', 'Cost', 'Target', 'Buy_Zone', 'Sell_Zone']:
        fmt_df[col] = fmt_df[col].apply(lambda x: f"${x:.2f}")
    fmt_df['PnL%'] = fmt_df['PnL%'].apply(lambda x: f"{x:+.2%}")
    fmt_df['Model_Error'] = fmt_df['Model_Error'].apply(lambda x: f"{x:.1%}")
    
    cols = ['Ticker', 'Type', 'Model_Error', 'Current', 'Target', 'Buy_Zone', 'Action']
    print(fmt_df[cols].to_markdown(index=False))
    
    fig = go.Figure()
    for i, row in df_res.iterrows():
        color = 'cyan' if row['PnL%'] > 0 else 'red'
        fig.add_trace(go.Box(
            y=[row['Buy_Zone'], row['Target'], row['Target'], row['Sell_Zone']],
            name=f"{row['Ticker']} (Err {row['Model_Error']})",
            marker_color=color, boxpoints=False
        ))
        fig.add_trace(go.Scatter(
            x=[f"{row['Ticker']} (Err {row['Model_Error']})"], y=[row['Cost']],
            mode='markers+text', marker=dict(symbol='line-ew', size=50, color='white', line=dict(width=3)),
            name='成本'
        ))
        fig.add_trace(go.Scatter(
            x=[f"{row['Ticker']} (Err {row['Model_Error']})"], y=[row['Current']],
            mode='markers', marker=dict(symbol='diamond', size=12, color='yellow'),
            name='現價'
        ))

    fig.update_layout(title="App 5.0 資產戰略圖 (12個月新聞回測)", template="plotly_dark", yaxis_title="價格 (USD)", showlegend=False, height=500)
    fig.show()