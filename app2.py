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
st.set_page_config(page_title="App 25.0 全能指揮官", layout="wide")
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

st.title("🦅 App 25.0: 全能指揮官 (Bug Fix + Validation)")
st.markdown("""
**系統狀態：**
* ✅ **修復 KeyError**：正確處理 OHLC 資料結構。
* ✅ **新增驗證層**：計算 30 天預測的 **方向準確率 (Dir_Acc)** 與 **價格誤差 (MAPE)**。
* ✅ **宏觀模型**：`預測價 = 4D模型 × 宏觀係數`。
""")

# ==========================================
# 1. 宏觀數據中心 (Macro Data Center)
# ==========================================
@st.cache_data(ttl=3600*4)
def fetch_grand_macro_data():
    # 抓取關鍵指標
    tickers = ['HG=F', 'GC=F', '^TNX', 'BTC-USD', '^VIX', 'DX-Y.NYB']
    try:
        data = yf.download(tickers, period="2y", progress=False)['Close']
        data = data.ffill().dropna()
        
        # 為了避免 MultiIndex 問題，這裡做簡單處理
        if isinstance(data.columns, pd.MultiIndex):
            # 嘗試扁平化或直接取值，視 yfinance 版本而定
            # 這裡假設 columns 是 (Ticker, Type) 或 Ticker
            pass 

        # 重新命名以防萬一 (針對 yfinance 新版)
        # 這裡用更通用的方式計算，假設 index 是日期
        
        # 計算指標
        # 1. 銅金比
        try:
            copper = data['HG=F']
            gold = data['GC=F']
            data['Copper_Gold'] = copper / gold
        except:
            data['Copper_Gold'] = 1.0 # Fallback

        macro_score = pd.DataFrame(index=data.index)
        
        # A. 經濟 (銅金比 > MA50)
        macro_score['Eco_Score'] = np.where(data['Copper_Gold'] > data['Copper_Gold'].rolling(50).mean(), 1, -1)
        
        # B. 流動性 (BTC > MA50)
        btc = data['BTC-USD']
        macro_score['Liq_Score'] = np.where(btc > btc.rolling(50).mean(), 1, -1)
        
        # C. 利率 (TNX < MA50)
        tnx = data['^TNX']
        macro_score['Rate_Score'] = np.where(tnx < tnx.rolling(50).mean(), 1, -1)
        
        # D. 恐慌 (VIX < 20)
        vix = data['^VIX']
        macro_score['VIX_Score'] = np.where(vix < 20, 1, -1)
        
        # E. 美元 (DXY < MA50)
        dxy = data['DX-Y.NYB']
        macro_score['DXY_Score'] = np.where(dxy < dxy.rolling(50).mean(), 1, -1)
        
        # 匯總分數 (-5 ~ +5)
        macro_score['Total_Score'] = (
            macro_score['Eco_Score'] + 
            macro_score['Liq_Score'] + 
            macro_score['Rate_Score'] + 
            macro_score['VIX_Score'] + 
            macro_score['DXY_Score']
        )
        
        # 轉換係數 (0.85 ~ 1.15)
        macro_score['Scalar'] = 1.0 + (macro_score['Total_Score'] * 0.03)
        
        return macro_score
    except Exception as e:
        # st.error(f"Macro Data Error: {e}")
        return pd.DataFrame()

# ==========================================
# 2. 四維定價引擎 (4D Pricing Engine)
# ==========================================
def train_rf_model(df, days=30):
    try:
        data = df[['Close']].copy()
        data['Ret'] = data['Close'].pct_change()
        data['Vol'] = data['Ret'].rolling(20).std()
        data['SMA'] = data['Close'].rolling(20).mean()
        data['Target'] = data['Close'].shift(-days)
        data = data.dropna()
        if len(data) < 60: return None
        X = data[['Ret', 'Vol', 'SMA']]
        y = data['Target']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        last_row = data.iloc[[-1]][['Ret', 'Vol', 'SMA']]
        return model.predict(last_row)[0]
    except: return None

def calc_4d_raw_target(df_price, days=30):
    # 確保有 High/Low (修復 KeyError)
    if 'High' not in df_price.columns or 'Low' not in df_price.columns:
        # 如果真的沒有，用 Close 代替 (Fallback)
        high = df_price['Close']
        low = df_price['Close']
    else:
        high = df_price['High']
        low = df_price['Low']
        
    current = df_price['Close'].iloc[-1]
    
    # 1. ATR (波動)
    tr = high - low
    atr = tr.rolling(14).mean().iloc[-1]
    t_atr = current + (atr * np.sqrt(days))
    
    # 2. Fib (結構)
    recent = df_price['Close'].iloc[-60:]
    t_fib = recent.max() + (recent.max() - recent.min()) * 0.618
    
    # 3. MC (慣性)
    mu = df_price['Close'].pct_change().mean()
    t_mc = current * ((1 + mu) ** days)
    
    # 4. RF (AI)
    t_rf = train_rf_model(df_price, days)
    if t_rf is None: t_rf = t_mc
    
    avg_raw = (t_atr + t_fib + t_mc + t_rf) / 4
    return avg_raw

# ==========================================
# 3. 誤差回測引擎 (Validation Engine)
# ==========================================
def run_forecast_validation(df_price, macro_score, days=30):
    """
    回測過去每一天的預測準度
    為了效能，回測時只用 3D (ATR+Fib+MC) + Macro，不跑 RF (太慢)
    """
    df = df_price.copy()
    
    # 對齊宏觀係數
    if not macro_score.empty:
        macro_aligned = macro_score['Scalar'].reindex(df.index).ffill().fillna(1.0)
        df['Macro_Scalar'] = macro_aligned
    else:
        df['Macro_Scalar'] = 1.0

    # 1. 計算歷史 Rolling Target (模擬當時的情況)
    # ATR
    tr = df['High'] - df['Low']
    atr = tr.rolling(14).mean()
    target_atr = df['Close'] + (atr * np.sqrt(days))
    
    # Fib (Rolling Max/Min)
    roll_max = df['Close'].rolling(60).max()
    roll_min = df['Close'].rolling(60).min()
    target_fib = roll_max + (roll_max - roll_min) * 0.618
    
    # MC (Simple Drift)
    avg_ret = df['Close'].pct_change().rolling(60).mean()
    target_mc = df['Close'] * ((1 + avg_ret) ** days)
    
    # 綜合預測 (Raw)
    raw_pred = (target_atr + target_fib + target_mc) / 3
    
    # 宏觀修正預測 (Final)
    df['Pred_Price'] = raw_pred * df['Macro_Scalar']
    
    # 2. 對答案 (未來價格)
    df['Actual_Future'] = df['Close'].shift(-days)
    
    # 3. 計算誤差
    valid = df.dropna(subset=['Pred_Price', 'Actual_Future'])
    
    if len(valid) == 0: return 0.0, 0.0, pd.DataFrame()
    
    # Metric A: MAPE (平均絕對誤差率)
    valid['Error_Pct'] = (valid['Pred_Price'] - valid['Actual_Future']).abs() / valid['Actual_Future']
    mape = valid['Error_Pct'].mean()
    
    # Metric B: Dir_Acc (方向準確度)
    # 預測方向: Pred > Current ?
    pred_dir = valid['Pred_Price'] > valid['Close']
    # 真實方向: Future > Current ?
    actual_dir = valid['Actual_Future'] > valid['Close']
    
    # 方向相同 = True
    correct = (pred_dir == actual_dir)
    dir_acc = correct.mean()
    
    return dir_acc, mape, valid

# ==========================================
# 4. 主程式
# ==========================================
st.sidebar.title("控制台")
default_tickers = ["TSM", "NVDA", "AMD", "SOXL", "URA", "0050.TW"]
user_tickers = st.sidebar.text_area("代號", ", ".join(default_tickers))
ticker_list = [t.strip().upper() for t in user_tickers.split(',')]

# 1. 宏觀數據
macro_df = fetch_grand_macro_data()
if not macro_df.empty:
    curr_scalar = macro_df['Scalar'].iloc[-1]
    st.subheader(f"🌍 全球宏觀係數: {curr_scalar:.2f}")
    st.divider()

if st.button("🚀 執行驗證與預測"):
    results = []
    st.subheader("📊 預測準度驗證報告 (30天)")
    
    for t in ticker_list:
        # 下載數據 (注意：保留 OHLC)
        df_price = yf.download(t, period="2y", progress=False, auto_adjust=True)
        
        # 處理 MultiIndex (修復 KeyError 的關鍵)
        if isinstance(df_price.columns, pd.MultiIndex):
            # 手動提取需要的欄位
            temp = pd.DataFrame()
            try:
                temp['Close'] = df_price['Close'][t]
                temp['High'] = df_price['High'][t]
                temp['Low'] = df_price['Low'][t]
                temp['Volume'] = df_price['Volume'][t]
                df_price = temp
            except:
                st.error(f"{t} 資料格式錯誤，跳過")
                continue
        else:
            # 確保欄位存在
            needed = ['Close', 'High', 'Low', 'Volume']
            if not all(col in df_price.columns for col in needed):
                # 嘗試簡單修復 (如果只有 Close)
                if 'Close' in df_price.columns:
                    df_price['High'] = df_price['Close']
                    df_price['Low'] = df_price['Close']
                    df_price['Volume'] = 0
                else:
                    st.error(f"{t} 缺少必要欄位")
                    continue

        # 1. 執行誤差驗證 (Validation)
        dir_acc, mape, history = run_forecast_validation(df_price, macro_df, days=30)
        
        # 2. 執行當下預測 (Current Forecast)
        raw_target = calc_4d_raw_target(df_price, days=30)
        final_target = raw_target * (curr_scalar if not macro_df.empty else 1.0)
        
        current_price = df_price['Close'].iloc[-1]
        upside = (final_target - current_price) / current_price
        
        # 判斷信賴度
        reliability = "高"
        if dir_acc < 0.5: reliability = "低 (反指標)"
        elif mape > 0.15: reliability = "中 (波動大)"
        
        results.append({
            'Ticker': t,
            'Current': current_price,
            'Pred_30D': final_target,
            'Upside': upside,
            'Dir_Acc': dir_acc,       # 用戶要求的重點
            'Avg_Error': mape,        # 用戶要求的重點
            'Reliability': reliability
        })
        
        with st.expander(f"🔎 {t}: 準度 {dir_acc:.0%} | 誤差 ±{mape:.1%}"):
            c1, c2 = st.columns(2)
            c1.metric("預測目標價", f"${final_target:.2f}", f"{upside:+.1%}")
            c1.write(f"原始 4D 價格: ${raw_target:.2f}")
            c1.write(f"宏觀修正係數: x{curr_scalar:.2f}")
            
            c2.markdown("#### 誤差分析")
            c2.write(f"方向預測準度: **{dir_acc:.1%}** (>{50}% 為佳)")
            c2.write(f"平均價格誤差: **{mape:.1%}** (越低越準)")
            
            # 畫出預測 vs 真實 (驗證圖)
            if not history.empty:
                fig = go.Figure()
                # 為了圖表清晰，只畫最近 150 天
                recent = history.iloc[-150:]
                fig.add_trace(go.Scatter(x=recent.index, y=recent['Close'], name='真實股價', line=dict(color='white', width=1)))
                fig.add_trace(go.Scatter(x=recent.index, y=recent['Pred_Price'], name='模型預測(30天前)', line=dict(color='#00FF7F', dash='dot')))
                fig.update_layout(height=250, title="過去預測軌跡驗證", template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0))
                c2.plotly_chart(fig, use_container_width=True)

    res_df = pd.DataFrame(results)
    
    st.markdown("### 🏆 最終驗證報告")
    show = res_df.copy()
    show['Current'] = show['Current'].apply(lambda x: f"${x:.2f}")
    show['Pred_30D'] = show['Pred_30D'].apply(lambda x: f"${x:.2f}")
    show['Upside'] = show['Upside'].apply(lambda x: f"{x:+.1%}")
    show['Dir_Acc'] = show['Dir_Acc'].apply(lambda x: f"{x:.0%}")
    show['Avg_Error'] = show['Avg_Error'].apply(lambda x: f"±{x:.1%}")
    
    st.dataframe(show.style.map(
        lambda x: 'background-color: #00FF7F; color: black' if '高' in str(x) else ('background-color: #FF4B4B; color: white' if '低' in str(x) else ''),
        subset=['Reliability']
    ))