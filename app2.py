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
st.set_page_config(page_title="App 24.0 萬物歸一指揮官", layout="wide")
LOCAL_NEWS_FILE = "news_data_local.csv"

# 初始化 Session
if 'news_data' not in st.session_state:
    if os.path.exists(LOCAL_NEWS_FILE):
        try:
            df_local = pd.read_csv(LOCAL_NEWS_FILE)
            if 'Date' in df_local.columns:
                df_local['Date'] = pd.to_datetime(df_local['Date'])
            st.session_state['news_data'] = df_local
        except: st.session_state['news_data'] = pd.DataFrame()
    else: st.session_state['news_data'] = pd.DataFrame()

st.title("🦅 App 24.0: 萬物歸一指揮官 (Grand Unified Model)")
st.markdown("""
**究極融合：微觀定價 + 宏觀修正**
* **微觀 (Micro)**：重啟 **四維模型 (4D)** 計算個股理論目標價。
* **宏觀 (Macro)**：引入 **銅金比、流動性、利率、VIX、美元** 算出環境係數。
* **公式**：`預測價 = 4D理論價 × 宏觀係數 (0.8~1.2)`
""")

# ==========================================
# 1. 宏觀數據中心 (Macro Data Center)
# ==========================================
@st.cache_data(ttl=3600*4)
def fetch_grand_macro_data():
    # 抓取關鍵指標
    # HG=F (銅), GC=F (金), ^TNX (利率), BTC-USD (流動性), ^VIX (恐慌), DX-Y.NYB (美元)
    tickers = ['HG=F', 'GC=F', '^TNX', 'BTC-USD', '^VIX', 'DX-Y.NYB']
    try:
        data = yf.download(tickers, period="2y", progress=False)['Close']
        
        # 處理數據 (填補缺值)
        data = data.ffill().dropna()
        
        # 1. 計算銅金比 (Copper/Gold Ratio) -> 經濟晴雨表
        data['Copper_Gold'] = data['HG=F'] / data['GC=F']
        
        # 2. 計算各指標趨勢 (相對於 50日均線)
        # 為了避免未來函數，我們使用 rolling
        macro_score = pd.DataFrame(index=data.index)
        
        # A. 銅金比: 向上 = 經濟好 (+1)
        cg_ma = data['Copper_Gold'].rolling(50).mean()
        macro_score['Eco_Score'] = np.where(data['Copper_Gold'] > cg_ma, 1, -1)
        
        # B. 流動性 (BTC): 向上 = 錢多 (+1)
        btc_ma = data['BTC-USD'].rolling(50).mean()
        macro_score['Liq_Score'] = np.where(data['BTC-USD'] > btc_ma, 1, -1)
        
        # C. 利率 (TNX): 向下 = 估值壓力小 (+1)
        tnx_ma = data['^TNX'].rolling(50).mean()
        macro_score['Rate_Score'] = np.where(data['^TNX'] < tnx_ma, 1, -1) # 注意方向
        
        # D. 恐慌 (VIX): 低於 20 = 穩定 (+1)
        macro_score['VIX_Score'] = np.where(data['^VIX'] < 20, 1, -1)
        
        # E. 美元 (DXY): 向下 = 資產價格好 (+1)
        dxy_ma = data['DX-Y.NYB'].rolling(50).mean()
        macro_score['DXY_Score'] = np.where(data['DX-Y.NYB'] < dxy_ma, 1, -1)
        
        # 總分 (-5 到 +5)
        macro_score['Total_Score'] = (
            macro_score['Eco_Score'] + 
            macro_score['Liq_Score'] + 
            macro_score['Rate_Score'] + 
            macro_score['VIX_Score'] + 
            macro_score['DXY_Score']
        )
        
        # 轉換為係數 (Scalar): 0.85 (極差) ~ 1.15 (極好)
        # 簡單映射: -5 -> 0.85, 0 -> 1.0, +5 -> 1.15
        # 斜率 = (1.15 - 0.85) / 10 = 0.03
        macro_score['Macro_Scalar'] = 1.0 + (macro_score['Total_Score'] * 0.03)
        
        return macro_score, data
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()

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

def calc_4d_raw_target(ticker, df_price, days=30):
    current = df_price['Close'].iloc[-1]
    
    # 1. ATR (波動邊界)
    tr = df_price['High'] - df_price['Low']
    atr = tr.rolling(14).mean().iloc[-1]
    t_atr = current + (atr * np.sqrt(days))
    
    # 2. Fibonacci (結構壓力)
    recent = df_price['Close'].iloc[-60:]
    t_fib = recent.max() + (recent.max() - recent.min()) * 0.618
    
    # 3. Monte Carlo (統計慣性)
    mu = df_price['Close'].pct_change().mean()
    t_mc = current * ((1 + mu) ** days)
    
    # 4. Random Forest (AI)
    t_rf = train_rf_model(df_price, days)
    if t_rf is None: t_rf = t_mc
    
    avg_raw = (t_atr + t_fib + t_mc + t_rf) / 4
    return avg_raw, t_atr, t_fib, t_mc, t_rf

# ==========================================
# 3. 回測引擎 (Macro-Adjusted Backtest)
# ==========================================
def run_macro_backtest(ticker, df_price, macro_score):
    df = df_price.copy()
    
    # 對齊宏觀數據
    macro_aligned = macro_score.reindex(df.index).ffill().dropna()
    df = df.join(macro_aligned)
    
    # 策略: 動態調整部位
    # 宏觀好 (Scalar > 1.0) -> 滿倉 (100%)
    # 宏觀差 (Scalar < 1.0) -> 減倉/空手 (0%)
    
    cash = 10000.0
    shares = 0.0
    total_invested = 10000.0
    
    dca_shares = 0.0 # Blind DCA
    
    history = []
    last_month = -1
    
    start_idx = 100 # 等宏觀數據穩定
    if len(df) < start_idx: return 0, 0, pd.DataFrame()
    
    for i in range(start_idx, len(df)):
        date = df.index[i]
        price = df['Close'].iloc[i]
        scalar = df['Macro_Scalar'].iloc[i]
        
        # A. 發薪日
        if date.month != last_month:
            if last_month != -1:
                income = 10000.0
                total_invested += income
                cash += income
                dca_shares += income / price
            last_month = date.month
            
        # B. 交易策略 (Macro Timing)
        # 如果環境好 (Scalar > 1.0)，積極買進
        if scalar >= 1.0:
            if cash > 0:
                shares += cash / price
                cash = 0
        # 如果環境極差 (Scalar <= 0.9)，賣出避險
        elif scalar <= 0.9:
            if shares > 0:
                cash += shares * price
                shares = 0
                
        # C. 結算
        val_macro = cash + (shares * price)
        val_dca = dca_shares * price
        
        history.append({
            'Date': date,
            'Macro_Val': val_macro,
            'DCA_Val': val_dca,
            'Invested': total_invested,
            'Scalar': scalar
        })
        
    res_df = pd.DataFrame(history)
    if res_df.empty: return 0, 0, pd.DataFrame()
    
    final_macro = res_df['Macro_Val'].iloc[-1]
    final_dca = res_df['DCA_Val'].iloc[-1]
    tot_inv = res_df['Invested'].iloc[-1]
    
    return (final_macro-tot_inv)/tot_inv, (final_dca-tot_inv)/tot_inv, res_df

# ==========================================
# 4. 主程式
# ==========================================
st.sidebar.title("控制台")
default_tickers = ["TSM", "NVDA", "AMD", "SOXL", "URA", "0050.TW"]
user_tickers = st.sidebar.text_area("代號", ", ".join(default_tickers))
ticker_list = [t.strip().upper() for t in user_tickers.split(',')]

# 1. 獲取宏觀環境
macro_df, raw_macro = fetch_grand_macro_data()

if not macro_df.empty:
    last_m = macro_df.iloc[-1]
    curr_scalar = last_m['Macro_Scalar']
    
    st.subheader(f"🌍 全球宏觀係數: {curr_scalar:.2f} (環境評分: {int(last_m['Total_Score'])}/5)")
    
    # 顯示儀表板
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("銅金比 (經濟)", "擴張" if last_m['Eco_Score']>0 else "收縮", delta_color="normal" if last_m['Eco_Score']>0 else "inverse")
    c2.metric("比特幣 (流動性)", "寬鬆" if last_m['Liq_Score']>0 else "緊縮")
    c3.metric("美債利率", "下降(好)" if last_m['Rate_Score']>0 else "上升(壞)")
    c4.metric("VIX 恐慌", "安穩" if last_m['VIX_Score']>0 else "恐慌")
    c5.metric("美元 DXY", "弱勢(好)" if last_m['DXY_Score']>0 else "強勢(壞)")
    st.divider()

if st.button("🚀 啟動萬物歸一預測"):
    results = []
    st.subheader("📊 宏觀修正後預測 (30天)")
    
    current_scalar = macro_df['Macro_Scalar'].iloc[-1] if not macro_df.empty else 1.0
    
    for t in ticker_list:
        df_price = yf.download(t, period="2y", progress=False, auto_adjust=True)
        if isinstance(df_price.columns, pd.MultiIndex):
            temp = df_price['Close'][[t]].copy(); temp.columns = ['Close']
            df_price = temp
        else:
            df_price = df_price[['Close']]
            
        # 1. 計算 4D 原始目標價
        raw_target, t_atr, t_fib, t_mc, t_rf = calc_4d_raw_target(t, df_price, days=30)
        
        # 2. 進行宏觀修正
        final_target = raw_target * current_scalar
        
        # 3. 執行宏觀回測
        roi_macro, roi_dca, history = run_macro_backtest(t, df_price, macro_df)
        
        current_price = df_price['Close'].iloc[-1]
        upside = (final_target - current_price) / current_price
        
        results.append({
            'Ticker': t,
            'Current': current_price,
            'Raw_Target': raw_target,
            'Final_Target': final_target,
            'Upside': upside,
            'Macro_ROI': roi_macro,
            'DCA_ROI': roi_dca,
            'Alpha': roi_macro - roi_dca
        })
        
        # 詳細圖表 (只顯示預測修正過程)
        with st.expander(f"🔎 {t}: 宏觀修正 {current_scalar:.2f}x -> 目標 ${final_target:.2f}"):
            c1, c2 = st.columns(2)
            c1.markdown("#### 定價公式")
            c1.latex(r"Target_{Final} = Target_{4D} \times Scalar_{Macro}")
            c1.write(f"原始 4D 均價: **${raw_target:.2f}**")
            c1.write(f"宏觀係數: **x {current_scalar:.2f}**")
            c1.write(f"最終預測: **${final_target:.2f}**")
            
            c2.markdown("#### 策略回測 (Macro Filter)")
            fig = go.Figure()
            if not history.empty:
                fig.add_trace(go.Scatter(x=history['Date'], y=history['Macro_Val'], name='宏觀擇時', line=dict(color='#00FF7F')))
                fig.add_trace(go.Scatter(x=history['Date'], y=history['DCA_Val'], name='無腦定投', line=dict(color='gray', dash='dot')))
            fig.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark")
            c2.plotly_chart(fig, use_container_width=True)

    res_df = pd.DataFrame(results)
    
    show = res_df.copy()
    show['Current'] = show['Current'].apply(lambda x: f"${x:.2f}")
    show['Raw_Target'] = show['Raw_Target'].apply(lambda x: f"${x:.2f}")
    show['Final_Target'] = show['Final_Target'].apply(lambda x: f"${x:.2f}")
    show['Upside'] = show['Upside'].apply(lambda x: f"{x:+.1%}")
    show['Macro_ROI'] = show['Macro_ROI'].apply(lambda x: f"{x:+.1%}")
    show['DCA_ROI'] = show['DCA_ROI'].apply(lambda x: f"{x:+.1%}")
    show['Alpha'] = show['Alpha'].apply(lambda x: f"{x:+.1%}")
    
    st.dataframe(show[['Ticker', 'Current', 'Raw_Target', 'Final_Target', 'Upside', 'Macro_ROI', 'DCA_ROI', 'Alpha']].style.map(
        lambda x: 'color: #00FF7F' if '+' in str(x) and float(str(x).strip('%+')) > 0 else 'color: white',
        subset=['Alpha', 'Upside']
    ))