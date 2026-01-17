import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 0. 頁面設定
# ==========================================
st.set_page_config(page_title="App 19.0 全景指揮官 (誤差校正版)", layout="wide")
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

st.title("🦅 App 19.0: 全景指揮官 (真實誤差校正版)")
st.markdown("""
**新增維度：**
1.  **方向準確度 (Dir_Acc)**：模型判斷漲跌的長期勝率。
2.  **預測誤差 (MAPE)**：回測過去每一天的「預測價 vs 真實價」，計算平均誤差率。
""")

# ==========================================
# 1. 核心工具
# ==========================================
@st.cache_data(ttl=3600*4)
def fetch_market_vitals():
    try:
        data = yf.download(['SPY', '^VIX'], period="2y", progress=False)['Close']
        if isinstance(data, pd.DataFrame) and 'SPY' in data.columns:
            spy = data['SPY']
            vix = data['^VIX']
        else: return pd.DataFrame(), pd.Series(), pd.Series()

        spy_ma200 = spy.rolling(200).mean()
        cond_green = (spy > spy_ma200) & (vix < 25)
        cond_red = (spy < spy_ma200) & (vix > 30)
        
        vitals = pd.DataFrame(index=data.index)
        vitals['Green'] = cond_green
        vitals['Red'] = cond_red
        vitals['Yellow'] = (~cond_green) & (~cond_red)
        return vitals, spy, vix
    except: return pd.DataFrame(), pd.Series(), pd.Series()

def calculate_vwap(df, window=20):
    v = df['Volume']
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * v).rolling(window).sum() / v.rolling(window).sum()

# ==========================================
# 2. 歷史誤差回測引擎 (Historical Error Engine)
# ==========================================
def calc_rolling_forecast_stats(df, days=30):
    """
    計算過去每一天的預測值，並與 N 天後的真實股價比較
    """
    d = df.copy()
    
    # 1. 建立滾動特徵 (模擬當時能看到的數據)
    # ATR Target
    tr = d['High'] - d['Low']
    atr = tr.rolling(14).mean()
    d['Target_ATR'] = d['Close'] + (atr * np.sqrt(days))
    
    # Fibonacci Target (Rolling Max)
    roll_max = d['Close'].rolling(60).max()
    roll_min = d['Close'].rolling(60).min()
    d['Target_Fib'] = roll_max + (roll_max - roll_min) * 0.618
    
    # Monte Carlo (Simple Drift)
    # 這裡用簡單的 20日平均漲幅推算
    avg_ret = d['Close'].pct_change().rolling(60).mean()
    d['Target_MC'] = d['Close'] * ((1 + avg_ret) ** days)
    
    # 綜合預測 (歷史回測不跑 RF 以免超時，僅用統計模型)
    d['Pred_Price'] = (d['Target_ATR'] * 0.3) + (d['Target_Fib'] * 0.3) + (d['Target_MC'] * 0.4)
    
    # 2. 對答案 (Future Close)
    d['Actual_Future'] = d['Close'].shift(-days)
    
    # 3. 計算誤差
    # Error % = |Pred - Actual| / Actual
    d['Error_Pct'] = (d['Pred_Price'] - d['Actual_Future']).abs() / d['Actual_Future']
    
    # 排除還沒發生未來的資料
    valid = d.dropna(subset=['Actual_Future', 'Error_Pct'])
    
    if len(valid) == 0: return 0.0, 0.0
    
    mape = valid['Error_Pct'].mean() # 平均誤差
    last_pred = d['Pred_Price'].iloc[-1] # 最新的預測值
    
    return mape, last_pred

# ==========================================
# 3. 綜合回測 (Smart DCA + Dir_Acc)
# ==========================================
def run_comprehensive_backtest(ticker, df_price, df_news, vitals):
    df = df_price.copy()
    
    # --- A. 數據整合 ---
    if not df_news.empty:
        if not pd.api.types.is_datetime64_any_dtype(df_news['Date']):
             df_news['Date'] = pd.to_datetime(df_news['Date'])
        daily_score = df_news.groupby('Date')['Score'].mean()
        df = df.join(daily_score, how='left').fillna(0)
        df['News_Roll'] = df['Score'].rolling(3).mean()
    else: df['News_Roll'] = 0
        
    vitals_aligned = vitals.reindex(df.index).ffill().fillna(False)
    df = df.join(vitals_aligned)
    
    df['MA60'] = df['Close'].rolling(60).mean()
    df['VWAP'] = calculate_vwap(df, 20)
    df['Dev_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']
    
    # --- B. 方向準確度 (Dir_Acc) ---
    # 預測 N 天後漲跌
    df['Ret_30D'] = df['Close'].shift(-30) / df['Close'] - 1
    
    # 簡單 Alpha 模型: News + Trend + VWAP
    # 如果新聞好 且 趨勢向上 且 在 VWAP 之上 -> 看多
    df['Alpha_Score'] = (df['News_Roll'] * 0.3) + (np.where(df['Close']>df['MA60'], 1, -1) * 0.4) + (np.where(df['Dev_VWAP']>0, 1, -1) * 0.3)
    
    valid_dir = df.dropna(subset=['Ret_30D'])
    if len(valid_dir) > 0:
        # 同號相乘 > 0 代表方向預測正確
        correct = (valid_dir['Alpha_Score'] * valid_dir['Ret_30D']) > 0
        dir_acc = correct.mean()
    else: dir_acc = 0.5
    
    # --- C. Smart DCA 回測 ---
    # 策略: 黃燈時才啟用智能 (趨勢向上+回調)，綠燈無腦買，紅燈不買
    cash = 10000.0; shares = 0.0; dca_shares = 0.0
    total_inv = 10000.0
    last_month = -1
    start_idx = 200
    
    cond_smart = (df['Close'] > df['MA60']) & (df['Dev_VWAP'].abs() < 0.05)
    
    for i in range(start_idx, len(df)):
        price = df['Close'].iloc[i]
        date = df.index[i]
        
        is_green = df['Green'].iloc[i] if 'Green' in df.columns else True
        is_yellow = df['Yellow'].iloc[i] if 'Yellow' in df.columns else False
        
        if date.month != last_month:
            if last_month != -1:
                income = 10000.0
                total_inv += income
                cash += income
                dca_shares += income / price
            last_month = date.month
            
        if is_green: # 綠燈無腦買
            if cash > 0:
                shares += cash / price
                cash = 0
        elif is_yellow: # 黃燈智能買
            if cash > 0 and cond_smart.iloc[i]:
                shares += cash / price
                cash = 0
                
    val_smart = cash + shares * df['Close'].iloc[-1]
    val_dca = dca_shares * df['Close'].iloc[-1]
    
    roi_smart = (val_smart - total_inv) / total_inv
    roi_dca = (val_dca - total_inv) / total_inv
    
    return dir_acc, roi_smart, roi_dca

# ==========================================
# 4. 主程式
# ==========================================
st.sidebar.title("控制台")
default_tickers = ["TSM", "NVDA", "AMD", "SOXL", "URA", "0050.TW"]
user_tickers = st.sidebar.text_area("代號", ", ".join(default_tickers))
ticker_list = [t.strip().upper() for t in user_tickers.split(',')]

vitals_df, _, _ = fetch_market_vitals()
if not vitals_df.empty:
    last = vitals_df.iloc[-1]
    status = "🟢 牛市健康" if last['Green'] else ("🔴 牛市休克" if last['Red'] else "🟡 牛市回檔")
    st.subheader(f"🏥 市場生命徵象: {status}")
    st.divider()

if st.button("🚀 執行全維度分析"):
    st.subheader("📊 全景分析報告")
    results = []
    
    news_df = st.session_state.get('news_data', pd.DataFrame())
    
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
        
        # 1. 執行綜合回測 (Dir_Acc, ROI)
        dir_acc, roi_smart, roi_dca = run_comprehensive_backtest(t, df_price, df_news_t, vitals_df)
        
        # 2. 執行誤差回測 (Forecast Error)
        mape, pred_price = calc_rolling_forecast_stats(df_price, days=30)
        
        current = df_price['Close'].iloc[-1]
        upside = (pred_price - current) / current
        
        # 3. 判斷模型可靠度
        reliability = "高"
        if dir_acc < 0.5 or mape > 0.2: reliability = "低 (誤差大)"
        elif dir_acc < 0.6: reliability = "中"
        
        results.append({
            'Ticker': t,
            'Dir_Acc': dir_acc,       # 方向準度
            'MAPE': mape,             # 價格誤差
            'Reliability': reliability,
            'Current': current,
            'Pred_30D': pred_price,
            'Upside': upside,
            'Smart_ROI': roi_smart,
            'DCA_ROI': roi_dca
        })
        
    res_df = pd.DataFrame(results)
    
    # 顯示
    show = res_df.copy()
    show['Dir_Acc'] = show['Dir_Acc'].apply(lambda x: f"{x:.0%}")
    show['MAPE'] = show['MAPE'].apply(lambda x: f"±{x:.1%}")
    show['Current'] = show['Current'].apply(lambda x: f"${x:.2f}")
    show['Pred_30D'] = show['Pred_30D'].apply(lambda x: f"${x:.2f}")
    show['Upside'] = show['Upside'].apply(lambda x: f"{x:+.1%}")
    show['Smart_ROI'] = show['Smart_ROI'].apply(lambda x: f"{x:+.1%}")
    show['DCA_ROI'] = show['DCA_ROI'].apply(lambda x: f"{x:+.1%}")
    
    st.dataframe(show.style.map(
        lambda x: 'background-color: #00FF7F; color: black' if '高' in str(x) else ('background-color: #FF4B4B; color: white' if '低' in str(x) else ''), 
        subset=['Reliability']
    ))
    
    st.info("💡 MAPE (平均誤差)：代表預測目標價的偏離程度。若 MAPE 為 ±10%，且預測漲 20%，則實際漲幅可能落在 10%~30% 之間。")