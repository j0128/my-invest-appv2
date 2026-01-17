import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(page_title="App 21.0 十年全景指揮官", layout="wide")
LOCAL_NEWS_FILE = "news_data_local.csv"

if 'news_data' not in st.session_state:
    if os.path.exists(LOCAL_NEWS_FILE):
        try:
            df_local = pd.read_csv(LOCAL_NEWS_FILE)
            if 'Date' in df_local.columns:
                df_local['Date'] = pd.to_datetime(df_local['Date'])
            st.session_state['news_data'] = df_local
        except: st.session_state['news_data'] = pd.DataFrame()
    else: st.session_state['news_data'] = pd.DataFrame()

st.title("🦅 App 21.0: 十年全景指揮官 (Decade-Scale Probability)")
st.markdown("""
**數據升級：**
* **時間跨度**：從 2 年擴展到 **10 年 (2015-2025)**。
* **包含週期**：涵蓋 2022 升息崩盤、2020 熔斷、2018 貿易戰。
* **目的**：讓模型學會「熊市」的樣子，避免在牛市末期過度樂觀。
""")

# ==========================================
# 1. 核心工具：10年宏觀數據
# ==========================================
@st.cache_data(ttl=3600*4)
def fetch_long_term_data(tickers, period="10y"):
    try:
        data = yf.download(tickers, period=period, progress=False)['Close']
        return data
    except: return pd.DataFrame()

# ==========================================
# 2. 歷史機率引擎 (10年版)
# ==========================================
def analyze_decade_probability(ticker, df_price, lookahead=30):
    df = df_price.copy()
    
    # 1. 定義狀態 (與 App 20.0 相同，但樣本變多)
    # A. 趨勢: Price vs MA200 (牛熊分界線)
    df['MA200'] = df['Close'].rolling(200).mean()
    df['Trend'] = np.where(df['Close'] > df['MA200'], 'Bull', 'Bear')
    
    # B. 乖離: Price vs MA60 (中期乖離)
    df['MA60'] = df['Close'].rolling(60).mean()
    df['Bias_60'] = (df['Close'] - df['MA60']) / df['MA60']
    
    # 定義乖離狀態
    # 這裡用統計分位數 (Quantile) 來定義何謂「過熱」
    # 因為 10 年的數據分佈比較準
    bias_high = df['Bias_60'].quantile(0.8) # 前 20% 高
    bias_low = df['Bias_60'].quantile(0.2)  # 前 20% 低
    
    conditions = [
        (df['Bias_60'] > bias_high),
        (df['Bias_60'] < bias_low),
        (df['Bias_60'] >= bias_low) & (df['Bias_60'] <= bias_high)
    ]
    choices = ['Overheated', 'Oversold', 'Normal']
    df['Bias_State'] = np.select(conditions, choices, default='Normal')
    
    # C. 波動率狀態 (VIX Proxy)
    # 用自身的波動率替代 VIX (因為個股股性不同)
    df['Vol_20'] = df['Close'].pct_change().rolling(20).std()
    vol_high = df['Vol_20'].quantile(0.7)
    df['Vol_State'] = np.where(df['Vol_20'] > vol_high, 'High_Vol', 'Low_Vol')
    
    # 組合簽名
    df['Signature'] = df['Trend'] + "_" + df['Bias_State'] + "_" + df['Vol_State']
    
    # 2. 計算未來回報
    df['Future_Ret'] = df['Close'].shift(-lookahead) / df['Close'] - 1
    
    # 3. 獲取當前狀態
    current_sig = df['Signature'].iloc[-1]
    
    # 4. 歷史搜尋 (10年數據)
    # 排除最近 30 天
    history = df.iloc[:-lookahead]
    matches = history[history['Signature'] == current_sig]
    
    # 5. 統計
    if len(matches) < 5: # 樣本不足，放寬條件
        fallback_sig = df['Trend'].iloc[-1] + "_" + df['Bias_State'].iloc[-1]
        df['Simple_Sig'] = df['Trend'] + "_" + df['Bias_State']
        matches = history[history['Simple_Sig'] == fallback_sig]
        note = "模糊比對 (10年樣本仍少)"
    else:
        note = "精確比對"
        
    if len(matches) > 0:
        win_rate = len(matches[matches['Future_Ret'] > 0]) / len(matches)
        exp_ret = matches['Future_Ret'].mean()
        avg_loss = matches[matches['Future_Ret'] < 0]['Future_Ret'].mean() if len(matches[matches['Future_Ret'] < 0]) > 0 else 0
        
        # 預測價格
        pred_price = df['Close'].iloc[-1] * (1 + exp_ret)
    else:
        win_rate = 0.5; exp_ret = 0.0; pred_price = df['Close'].iloc[-1]
        avg_loss = 0.0; note = "無歷史樣本"
        
    return {
        'State': current_sig,
        'Count': len(matches),
        'Note': note,
        'Win_Rate': win_rate,
        'Exp_Return': exp_ret,
        'Avg_Loss': avg_loss,
        'Pred_Price': pred_price,
        'Current_Bias': df['Bias_60'].iloc[-1],
        'High_Bias_Threshold': bias_high
    }

# ==========================================
# 3. 主程式
# ==========================================
st.sidebar.title("控制台")
default_tickers = ["TSM", "NVDA", "AMD", "SOXL", "URA", "0050.TW", "SPY"]
user_tickers = st.sidebar.text_area("代號", ", ".join(default_tickers))
ticker_list = [t.strip().upper() for t in user_tickers.split(',')]

st.info("💡 資料庫已切換為 **10年期 (2015-2025)**。這能捕捉到 2022 熊市與 2020 崩盤的特徵，讓預測更保守且真實。")

if st.button("🚀 執行十年機率預測"):
    results = []
    
    for t in ticker_list:
        # 下載 10 年數據
        df_price = yf.download(t, period="10y", progress=False, auto_adjust=True)
        if isinstance(df_price.columns, pd.MultiIndex):
            temp = df_price['Close'][[t]].copy(); temp.columns = ['Close']
            df_price = temp
        else:
            df_price = df_price[['Close']]
            
        if len(df_price) < 250: # 新股保護
            st.warning(f"{t} 上市時間不足 10 年，將使用現有數據。")
            
        # 執行分析
        data = analyze_decade_probability(t, df_price, lookahead=30)
        
        # 判斷方向
        if data['Win_Rate'] > 0.6: 
            direction = "↗️ 看漲"
            color = "#00FF7F"
        elif data['Win_Rate'] < 0.4: 
            direction = "↘️ 看跌"
            color = "#FF4B4B"
        else: 
            direction = "➡️ 震盪"
            color = "gray"
            
        # 判斷是否過熱 (跟自己的 10 年歷史比)
        bias_status = "正常"
        if data['Current_Bias'] > data['High_Bias_Threshold']:
            bias_status = "⚠️ 歷史高點過熱"
        elif data['Current_Bias'] < -0.1: # 簡單定義
            bias_status = "🥶 歷史低檔"
            
        results.append({
            'Ticker': t,
            'Current': df_price['Close'].iloc[-1],
            'Pred_30D': data['Pred_Price'],
            'Direction': direction,
            'Win_Rate': data['Win_Rate'],
            'Exp_Ret': data['Exp_Return'],
            'Max_Risk': data['Avg_Loss'],
            'State': data['State'],
            'Bias_Status': bias_status,
            'Samples': data['Count']
        })
        
        # Expander
        with st.expander(f"{t}: {direction} (勝率 {data['Win_Rate']:.0%}) | {bias_status}"):
            c1, c2 = st.columns(2)
            c1.markdown("#### 當前狀態 (10年尺度)")
            c1.write(f"狀態簽名: `{data['State']}`")
            c1.write(f"歷史出現次數: {data['Count']} 次 ({data['Note']})")
            c1.metric("乖離水位", f"{data['Current_Bias']:.1%}", f"歷史高標: {data['High_Bias_Threshold']:.1%}")
            
            c2.markdown("#### 30天後劇本")
            c2.write(f"期望回報: **{data['Exp_Return']:+.1%}**")
            c2.write(f"平均下行風險: **{data['Avg_Loss']:.1%}**")
            
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = data['Win_Rate'] * 100,
                title = {'text': "10年歷史勝率"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': color}}
            ))
            fig.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20))
            st.plotly_chart(fig, use_container_width=True)

    res_df = pd.DataFrame(results)
    
    st.markdown("### 🏆 十年全景報告")
    show = res_df.copy()
    show['Current'] = show['Current'].apply(lambda x: f"${x:.2f}")
    show['Pred_30D'] = show['Pred_30D'].apply(lambda x: f"${x:.2f}")
    show['Win_Rate'] = show['Win_Rate'].apply(lambda x: f"{x:.0%}")
    show['Exp_Ret'] = show['Exp_Ret'].apply(lambda x: f"{x:+.1%}")
    
    st.dataframe(show[['Ticker', 'Direction', 'Win_Rate', 'Exp_Ret', 'Current', 'Pred_30D', 'Bias_Status', 'Samples']].style.map(
        lambda x: 'color: #FF4B4B' if '過熱' in str(x) or '看跌' in str(x) else ('color: #00FF7F' if '看漲' in str(x) else ''),
        subset=['Direction', 'Bias_Status']
    ))