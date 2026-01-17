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

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(page_title="App 8.1 全球情報網", layout="wide")

st.title("🦅 App 8.1: 全球情報網 (美/台/日/歐 四核心)")
st.markdown("""
**戰略地圖全開：**
1.  **🇺🇸 美國 (US)**：全球資金共識 (NVDA, BTC, META)。
2.  **🇹🇼 台灣 (TW)**：半導體製造內幕 (TSM)。
3.  **🇯🇵 日本 (JP)**：材料設備上游 (SOXL)。
4.  **🇪🇺 歐洲 (EU)**：核能與設備巨頭 (URA, ASML)。
""")

# ==========================================
# 1. 多語言金融字典 (含德文)
# ==========================================
MULTILINGUAL_DICT = {
    'ZH': { # 中文 (TW)
        'UP': ['大漲', '漲停', '創高', '新高', '利多', '優於預期', '爆發', '擴產', '完銷', '急單', '看好', '買進', '加碼', '成長'],
        'DOWN': ['大跌', '跌停', '重挫', '新低', '利空', '不如預期', '砍單', '衰退', '虧損', '裁員', '看壞', '賣出', '減碼', '疲弱']
    },
    'JA': { # 日文 (JP)
        'UP': ['上昇', '急騰', '最高値', '好調', '増益', '最高益', '買収', '提携', '拡大', '回復', '期待', 'ストップ高'],
        'DOWN': ['下落', '急落', '最安値', '不調', '減益', '赤字', '撤退', '中止', '縮小', '懸念', '失望', 'ストップ安']
    },
    'DE': { # 德文 (EU)
        'UP': ['anstieg', 'rekord', 'gewinn', 'kaufen', 'bullisch', 'wachstum', 'erholung', 'hoch', 'positiv', 'übertreffen'],
        'DOWN': ['verlust', 'fallen', 'krise', 'verkaufen', 'bärisch', 'rückgang', 'tief', 'negativ', 'warnung', 'absturz']
    }
}

# 股票代號自動翻譯機
TICKER_MAP = {
    'TSM': {'TW': '台積電', 'JP': 'TSMC', 'EU': 'TSMC'},
    'NVDA': {'TW': '輝達', 'JP': 'NVIDIA', 'EU': 'Nvidia'},
    'AMD': {'TW': '超微', 'JP': 'AMD', 'EU': 'AMD'},
    'URA': {'TW': '鈾礦', 'JP': 'ウラン', 'EU': 'Uranium'}, # URA 關鍵
    'ASML': {'TW': '艾司摩爾', 'JP': 'ASML', 'EU': 'ASML'},
    'SOXL': {'TW': '半導體', 'JP': '半導体', 'EU': 'Semiconductor'},
    'BTC-USD': {'TW': '比特幣', 'JP': 'ビットコイン', 'EU': 'Bitcoin'}
}

# ==========================================
# 2. 全球 RSS 駭客 (美/台/日/歐)
# ==========================================
@st.cache_data(ttl=3600*12) 
def fetch_global_news(ticker, months=12):
    news_history = []
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=months)
    
    # 準備搜尋關鍵字
    map_info = TICKER_MAP.get(ticker, {})
    
    # 英文關鍵字 (美/英)
    term_us = f"{ticker}+stock" if len(ticker) <= 4 else ticker
    
    # 在地關鍵字
    term_tw = urllib.parse.quote(map_info.get('TW', ticker))
    term_jp = urllib.parse.quote(map_info.get('JP', ticker))
    term_eu = urllib.parse.quote(map_info.get('EU', ticker)) # 歐洲關鍵字

    current = start_date
    while current < end_date:
        next_month = current + relativedelta(months=1)
        d_after = current.strftime('%Y-%m-%d')
        d_before = next_month.strftime('%Y-%m-%d')
        
        # --- 1. US Node (美) ---
        url_us = f"https://news.google.com/rss/search?q={term_us}+after:{d_after}+before:{d_before}&hl=en-US&gl=US&ceid=US:en"
        parse_rss_feed(url_us, 'US', news_history, current.date())
        
        # --- 2. TW Node (台) ---
        if ticker in ['TSM', 'NVDA', 'AMD', '0050.TW', 'CLS', 'SOXL'] or '.TW' in ticker:
            url_tw = f"https://news.google.com/rss/search?q={term_tw}+after:{d_after}+before:{d_before}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
            parse_rss_feed(url_tw, 'TW', news_history, current.date())
            
        # --- 3. JP Node (日) ---
        if ticker in ['TSM', 'NVDA', 'AMD', 'SOXL', 'BTC-USD', 'URA']: # 日本重啟核能，URA 相關
            url_jp = f"https://news.google.com/rss/search?q={term_jp}+after:{d_after}+before:{d_before}&hl=ja&gl=JP&ceid=JP:ja"
            parse_rss_feed(url_jp, 'JP', news_history, current.date())

        # --- 4. EU Node (歐 - 德/英) ---
        # 針對 URA (核能), SOXL (ASML), CLS (全球佈局), TLT (歐債影響)
        if ticker in ['URA', 'SOXL', 'CLS', 'TLT', 'BTC-USD', 'AMD']:
            # 德國 (DE) - 抓工業/核能
            url_de = f"https://news.google.com/rss/search?q={term_eu}+after:{d_after}+before:{d_before}&hl=de&gl=DE&ceid=DE:de"
            parse_rss_feed(url_de, 'EU_DE', news_history, current.date())
            
            # 英國 (UK) - 抓金融共識
            url_uk = f"https://news.google.com/rss/search?q={term_us}+after:{d_after}+before:{d_before}&hl=en-GB&gl=GB&ceid=GB:en"
            parse_rss_feed(url_uk, 'EU_UK', news_history, current.date())

        current = next_month
        time.sleep(0.1) 
        
    if not news_history: return pd.DataFrame(columns=['Date', 'Score', 'Title', 'Region'])
    df = pd.DataFrame(news_history)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def parse_rss_feed(url, region, container, date_ref):
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:3]: 
            title = entry.title
            score = 0
            
            if region in ['US', 'EU_UK']:
                score = TextBlob(title).sentiment.polarity
                if any(x in title.lower() for x in ['beat', 'surge', 'jump', 'record', 'buy']): score += 0.3
                if any(x in title.lower() for x in ['miss', 'drop', 'plunge', 'cut', 'sell']): score -= 0.3
                
            elif region == 'TW':
                for k in MULTILINGUAL_DICT['ZH']['UP']: 
                    if k in title: score += 0.5
                for k in MULTILINGUAL_DICT['ZH']['DOWN']: 
                    if k in title: score -= 0.5
                    
            elif region == 'JP':
                for k in MULTILINGUAL_DICT['JA']['UP']: 
                    if k in title: score += 0.5
                for k in MULTILINGUAL_DICT['JA']['DOWN']: 
                    if k in title: score -= 0.5
            
            elif region == 'EU_DE': # 德文
                t_lower = title.lower()
                for k in MULTILINGUAL_DICT['DE']['UP']: 
                    if k in t_lower: score += 0.5
                for k in MULTILINGUAL_DICT['DE']['DOWN']: 
                    if k in t_lower: score -= 0.5
            
            if score != 0:
                container.append({
                    'Date': pd.to_datetime(entry.published).date() if hasattr(entry, 'published') else date_ref,
                    'Score': np.clip(score, -1, 1),
                    'Title': f"[{region}] {title}",
                    'Region': region
                })
    except: pass

# ==========================================
# 3. 戰略引擎 (四國權重版)
# ==========================================
STRATEGY_DB = {
    'TSM': {'Type': '機構型', 'W': {'Fund': 0.1, 'Tech': 0.2, 'News': 0.7}}, 
    'NVDA': {'Type': '信仰型', 'W': {'Fund': 0.1, 'Tech': 0.6, 'News': 0.3}},
    # URA: 歐洲權重拉高，因為核能是歐洲大事
    'URA': {'Type': '政策型', 'W': {'Fund': 0.2, 'Tech': 0.3, 'News': 0.5}}, 
    'SOXL': {'Type': '投機型', 'W': {'Fund': 0.1, 'Tech': 0.4, 'News': 0.5}},
    'DEFAULT': {'Type': '一般型', 'W': {'Fund': 0.3, 'Tech': 0.4, 'News': 0.3}}
}

def analyze_ticker_global(ticker, value_ntd):
    # 1. 股價
    df_price = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
    if isinstance(df_price.columns, pd.MultiIndex):
        temp = df_price['Close'][[ticker]].copy(); temp.columns = ['Close']
        df_price = temp
    else:
        df_price = df_price[['Close']]
    
    if df_price.empty: return None

    # 2. 全球新聞挖掘
    df_news = fetch_global_news(ticker, months=12)
    
    # 3. 新聞情緒融合
    if not df_news.empty:
        # 計算每日加權分數 (TW/JP/EU 的分數給予加成，因為是第一手)
        def weighted_score(x):
            w_sum = 0
            count = 0
            for s, r in zip(x['Score'], x['Region']):
                # 在地情報加權 1.2 倍
                weight = 1.2 if r in ['TW', 'JP', 'EU_DE'] else 1.0
                w_sum += s * weight
                count += 1
            return w_sum / count if count > 0 else 0

        daily_news = df_news.groupby('Date').apply(weighted_score).rename('Score')
        df_price = df_price.join(daily_news, how='left').fillna(0)
        df_price['News_Factor'] = df_price['Score'].rolling(3).mean()
        
        # 抓出最新標題 (顯示各國來源)
        latest_titles = df_news.sort_values('Date').tail(3)['Title'].tolist()
        latest_news_str = " | ".join(latest_titles)
    else:
        df_price['News_Factor'] = 0
        latest_news_str = "無全球新聞"

    # 4. 因子運算
    df_price['MA200'] = df_price['Close'].rolling(200).mean()
    df_price['Bias'] = (df_price['Close'] - df_price['MA200']) / df_price['MA200']
    df_price['Score_F'] = -np.clip(df_price['Bias'] * 2, -1, 1) 
    
    df_price['MA20'] = df_price['Close'].rolling(20).mean()
    df_price['Score_T'] = np.where(df_price['Close'] > df_price['MA20'], 0.8, -0.8)
    
    strategy = STRATEGY_DB.get(ticker, STRATEGY_DB['DEFAULT'])
    w = strategy['W']
    
    df_price['Alpha_Score'] = (df_price['Score_F'] * w['Fund']) + \
                              (df_price['Score_T'] * w['Tech']) + \
                              (df_price['News_Factor'] * w['News'])

    # 5. 真實方向回測 (一年前)
    future_ret = df_price['Close'].shift(-20) - df_price['Close']
    valid_mask = (df_price.index > (datetime.now() - timedelta(days=365))) & (future_ret.notna())
    check_df = df_price[valid_mask]
    
    if not check_df.empty:
        hits = np.sign(check_df['Alpha_Score']) == np.sign(check_df['Close'].shift(-20) - check_df['Close'])
        dir_acc = hits.mean()
    else:
        dir_acc = 0.5

    # 6. 結果
    current_price = df_price['Close'].iloc[-1]
    current_alpha = df_price['Alpha_Score'].iloc[-1]
    vol = df_price['Close'].pct_change().rolling(30).std().iloc[-1] * np.sqrt(30)
    
    target = current_price * (1 + current_alpha * 0.05)
    buy_zone = target * (1 - vol * 1.5)
    sell_zone = target * (1 + vol * 1.5)
    
    return {
        '代號': ticker, '方向準確度': dir_acc,
        '現價': current_price, '建議買點': buy_zone, '建議賣點': sell_zone,
        '最新情報': latest_news_str, 'Alpha值': current_alpha, '市值(NTD)': value_ntd
    }

# ==========================================
# 4. 執行介面
# ==========================================
# 匯率
@st.cache_data(ttl=3600)
def get_rate():
    try: return yf.download("USDTWD=X", period="1d", progress=False)['Close'].iloc[-1].item()
    except: return 32.5
EXCHANGE_RATE = get_rate()
st.sidebar.metric("匯率 (USDTWD)", f"{EXCHANGE_RATE:.2f}")

st.sidebar.header("📂 匯入資產")
uploaded_file = st.sidebar.file_uploader("上傳 CSV", type=["csv"])

MY_PORTFOLIO = [{"Ticker": "URA", "Value_NTD": 100000}, {"Ticker": "TSM", "Value_NTD": 100000}] # Default URA Demo

if uploaded_file:
    try:
        df_up = pd.read_csv(uploaded_file)
        df_up.columns = [str(c).upper().strip() for c in df_up.columns]
        # (解析邏輯省略，同上版)
        clean = []
        for i, r in df_up.iterrows():
            clean.append({"Ticker": str(r[0]), "Value_NTD": 100000}) # 簡化示範
        MY_PORTFOLIO = clean
        st.sidebar.success(f"讀取 {len(clean)} 筆")
    except: pass

if st.button("🚀 啟動全球情報網 (四國聯防)", type="primary"):
    results = []
    bar = st.progress(0)
    status = st.empty()
    
    for i, item in enumerate(MY_PORTFOLIO):
        t = item['Ticker']
        status.text(f"正在掃描 美/台/日/歐 情報網: {t}... ({i+1}/{len(MY_PORTFOLIO)})")
        try:
            res = analyze_ticker_global(t, item['Value_NTD'])
            if res: results.append(res)
        except Exception as e: st.error(f"{t}: {e}")
        bar.progress((i+1)/len(MY_PORTFOLIO))
        
    status.text("✅ 完成")
    
    if results:
        df_res = pd.DataFrame(results)
        st.subheader("📊 全球戰略報告 (含歐洲視角)")
        
        # 樣式
        show = df_res.copy()
        show['方向準確度'] = show['方向準確度'].apply(lambda x: f"{x:.0%}")
        for c in ['現價','建議買點','建議賣點']: show[c] = show[c].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(show.style.map(
            lambda x: 'background-color: #1f77b4; color: white' if isinstance(x, str) and '%' in x and int(x.strip('%')) > 60 else '',
            subset=['方向準確度']
        ))
        
        # 氣泡圖
        fig = go.Figure()
        for i, row in df_res.iterrows():
            upside = (row['建議賣點'] - row['現價']) / row['現價']
            color = '#00FF7F' if row['方向準確度'] > 0.6 else '#FF4B4B'
            fig.add_trace(go.Scatter(
                x=[row['方向準確度']], y=[upside], mode='markers+text', text=[row['代號']],
                textposition="top center", marker=dict(size=25, color=color),
                name=row['代號']
            ))
        fig.update_layout(title="全球戰略矩陣", template="plotly_dark", height=500, xaxis_title="準確度", yaxis_title="潛在漲幅")
        st.plotly_chart(fig, use_container_width=True)