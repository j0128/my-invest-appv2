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
st.set_page_config(page_title="App 18.0 全景指揮官", layout="wide")
LOCAL_NEWS_FILE = "news_data_local.csv"

# 初始化 Session State
if 'news_data' not in st.session_state:
    if os.path.exists(LOCAL_NEWS_FILE):
        try:
            df_local = pd.read_csv(LOCAL_NEWS_FILE)
            if 'Date' in df_local.columns:
                df_local['Date'] = pd.to_datetime(df_local['Date'])
            st.session_state['news_data'] = df_local
        except: st.session_state['news_data'] = pd.DataFrame()
    else: st.session_state['news_data'] = pd.DataFrame()

st.title("🦅 App 18.0: 全景指揮官 (Backtest + Forecast)")
st.markdown("""
**三維戰略系統：**
1.  **現在 (Macro)**：牛市生命徵象監測 (Vitals Monitor)。
2.  **過去 (Backtest)**：智能定投 vs 無腦定投績效驗證。
3.  **未來 (Forecast)**：**四維模型預測未來 30 天目標價**。
""")

# ==========================================
# 1. 核心工具：宏觀 & 技術
# ==========================================
@st.cache_data(ttl=3600*4)
def fetch_market_vitals():
    try:
        data = yf.download(['SPY', '^VIX'], period="2y", progress=False)['Close']
        if isinstance(data, pd.DataFrame) and 'SPY' in data.columns and '^VIX' in data.columns:
            spy = data['SPY']
            vix = data['^VIX']
        else:
            # Fallback handling for newer yfinance versions or single ticker return structure
             return pd.DataFrame(), pd.Series(), pd.Series()

        spy_ma200 = spy.rolling(200).mean()
        spy_ma50 = spy.rolling(50).mean()
        
        # 🟢 綠燈: 在年線之上，且恐慌指數低
        cond_green = (spy > spy_ma200) & (vix < 25)
        # 🔴 紅燈: 跌破年線，且恐慌指數極高
        cond_red = (spy < spy_ma200) & (vix > 30)
        
        vitals = pd.DataFrame(index=data.index)
        vitals['Green'] = cond_green
        vitals['Red'] = cond_red
        vitals['Yellow'] = (~cond_green) & (~cond_red)
        
        return vitals, spy, vix
    except:
        return pd.DataFrame(), pd.Series(), pd.Series()

def calculate_vwap(df, window=20):
    v = df['Volume']
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * v).rolling(window).sum() / v.rolling(window).sum()

# ==========================================
# 2. 預測引擎 (Forecast Engine - 30 Days)
# ==========================================
def train_rf_model(df, ticker, days=30):
    try:
        data = df[['Close']].copy()
        data['Ret'] = data['Close'].pct_change()
        data['Vol'] = data['Ret'].rolling(20).std()
        data['SMA'] = data['Close'].rolling(20).mean()
        data['Target'] = data['Close'].shift(-days) # 預測未來 N 天
        data = data.dropna()
        
        if len(data) < 60: return None
        
        X = data[['Ret', 'Vol', 'SMA']]
        y = data['Target']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # 使用最後一天的數據進行預測
        last_row = data.iloc[[-1]][['Ret', 'Vol', 'SMA']]
        return model.predict(last_row)[0]
    except: return None

def calc_4d_forecast(ticker, df_price, days=30):
    current = df_price['Close'].iloc[-1]
    
    # 1. ATR (物理極限)
    tr = df_price['High'] - df_price['Low']
    atr = tr.rolling(14).mean().iloc[-1]
    # 假設波動擴散 30 天
    t_atr_high = current + (atr * np.sqrt(days))
    
    # 2. Fibonacci (黃金分割延伸)
    recent = df_price['Close'].iloc[-60:] # 近一季
    high = recent.max()
    low = recent.min()
    t_fib = high + (high - low) * 0.618 # 1.618 延伸位
    
    # 3. Mean Reversion / Monte Carlo (統計慣性)
    # 計算日平均報酬與波動
    returns = df_price['Close'].pct_change().dropna()
    mu = returns.mean()
    # 簡單複利推算
    t_mc = current * ((1 + mu) ** days)
    
    # 4. Random Forest (AI 模式識別)
    t_rf = train_rf_model(df_price, ticker, days)
    if t_rf is None: t_rf = t_mc # Fallback
    
    # 綜合加權
    # RF 與 MC 通常比較準，權重稍微高一點
    avg_target = (t_atr_high * 0.2) + (t_fib * 0.2) + (t_mc * 0.3) + (t_rf * 0.3)
    
    return {
        'Avg_Target': avg_target,
        'ATR_Target': t_atr_high,
        'Fib_Target': t_fib,
        'MC_Target': t_mc,
        'RF_Target': t_rf
    }

# ==========================================
# 3. 回測引擎 (Backtest Engine - Smart DCA)
# ==========================================
def run_backtest_simulation(ticker, df_price, vitals):
    df = df_price.copy()
    
    # 對齊生命徵象
    if not vitals.empty:
        vitals_aligned = vitals.reindex(df.index).ffill().fillna(False)
        df = df.join(vitals_aligned)
    else:
        df['Green'] = True # 預設多頭
        df['Yellow'] = False
        df['Red'] = False

    df['MA60'] = df['Close'].rolling(60).mean()
    df['VWAP'] = calculate_vwap(df, 20)
    df['Dev_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']
    
    # 策略變數
    cash = 10000.0
    shares = 0.0
    total_invested = 10000.0
    
    dca_shares = 0.0 # Blind DCA
    
    history = []
    last_month = -1
    start_idx = 200 # Need enough data for MA200 in vitals
    
    if len(df) < start_idx: return 0, 0, 0, pd.DataFrame()

    # 智能買點: 黃燈時，趨勢向上且回調 VWAP
    cond_smart_buy = (df['Close'] > df['MA60']) & (df['Dev_VWAP'].abs() < 0.05)
    
    for i in range(start_idx, len(df)):
        date = df.index[i]
        price = df['Close'].iloc[i]
        
        is_green = df['Green'].iloc[i] if 'Green' in df.columns else True
        is_yellow = df['Yellow'].iloc[i] if 'Yellow' in df.columns else False
        is_red = df['Red'].iloc[i] if 'Red' in df.columns else False
        
        # --- Monthly Contribution ---
        if date.month != last_month:
            if last_month != -1:
                income = 10000.0
                total_invested += income
                cash += income
                dca_shares += income / price
            last_month = date.month
            
        # --- Strategy ---
        if is_green:
            # 綠燈: 全力買進
            if cash > 0:
                shares += cash / price
                cash = 0
        elif is_yellow:
            # 黃燈: 擇機買進
            if cash > 0 and cond_smart_buy.iloc[i]:
                shares += cash / price
                cash = 0
        elif is_red:
            # 紅燈: 停止買進 (持有不動)
            pass
            
        # --- Valuation ---
        val_strat = cash + (shares * price)
        val_dca = dca_shares * price
        
        history.append({
            'Date': date,
            'Strat_Val': val_strat,
            'DCA_Val': val_dca,
            'Invested': total_invested
        })
        
    res_df = pd.DataFrame(history)
    if res_df.empty: return 0, 0, 0, pd.DataFrame()
    
    final_strat = res_df['Strat_Val'].iloc[-1]
    final_dca = res_df['DCA_Val'].iloc[-1]
    tot_inv = res_df['Invested'].iloc[-1]
    
    return (final_strat - tot_inv)/tot_inv, (final_dca - tot_inv)/tot_inv, tot_inv, res_df

# ==========================================
# 4. 主程式介面
# ==========================================
st.sidebar.title("控制台")
default_tickers = ["TSM", "NVDA", "AMD", "SOXL", "URA", "0050.TW"]
user_tickers = st.sidebar.text_area("代號", ", ".join(default_tickers))
ticker_list = [t.strip().upper() for t in user_tickers.split(',')]

# 1. 宏觀監測
vitals_df, spy_s, vix_s = fetch_market_vitals()
if not vitals_df.empty:
    last_v = vitals_df.iloc[-1]
    status = "🟢 牛市健康" if last_v['Green'] else ("🔴 牛市休克" if last_v['Red'] else "🟡 牛市回檔")
    st.subheader(f"🏥 市場生命徵象: {status}")
    st.divider()

if st.button("🚀 執行全景分析"):
    results = []
    
    st.subheader("📊 回測與預測報告")
    
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
            
        # 1. 執行回測 (Smart vs Blind)
        roi_smart, roi_dca, inv, history = run_backtest_simulation(t, df_price, vitals_df)
        
        # 2. 執行預測 (30 Days Forecast)
        forecast_data = calc_4d_forecast(t, df_price, days=30)
        
        current_price = df_price['Close'].iloc[-1]
        target_price = forecast_data['Avg_Target']
        upside = (target_price - current_price) / current_price
        
        results.append({
            'Ticker': t,
            'Current': current_price,
            'Pred_30D': target_price,
            'Upside_30D': upside,
            'Smart_ROI': roi_smart,
            'DCA_ROI': roi_dca,
            'Alpha': roi_smart - roi_dca,
            'Details': forecast_data # For tooltip or details
        })
        
        # 每個 Ticker 的詳細圖表 (只顯示預測部分或回測部分)
        # 這裡我們做一個 Expander 顯示詳細資訊
        with st.expander(f"🔎 {t}: 預測 ${target_price:.2f} ({upside:+.1%}) | Alpha {roi_smart-roi_dca:+.1%}"):
            c1, c2 = st.columns(2)
            
            # 左邊：預測組成
            c1.markdown("#### 30天目標價組成")
            c1.write(f"🤖 AI 模型 (RF): **${forecast_data['RF_Target']:.2f}**")
            c1.write(f"📈 統計慣性 (MC): **${forecast_data['MC_Target']:.2f}**")
            c1.write(f"🌊 波動極限 (ATR): **${forecast_data['ATR_Target']:.2f}**")
            c1.write(f"📐 黃金分割 (Fib): **${forecast_data['Fib_Target']:.2f}**")
            
            # 右邊：回測曲線
            c2.markdown("#### 策略回測曲線")
            fig = go.Figure()
            if not history.empty:
                fig.add_trace(go.Scatter(x=history['Date'], y=history['Strat_Val'], name='智能定投', line=dict(color='#00FF7F')))
                fig.add_trace(go.Scatter(x=history['Date'], y=history['DCA_Val'], name='無腦定投', line=dict(color='gray', dash='dot')))
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=200, template="plotly_dark")
            c2.plotly_chart(fig, use_container_width=True)

    res_df = pd.DataFrame(results)
    
    # 總表顯示
    st.markdown("### 🏆 總結報告")
    show = res_df.copy()
    show['Current'] = show['Current'].apply(lambda x: f"${x:.2f}")
    show['Pred_30D'] = show['Pred_30D'].apply(lambda x: f"${x:.2f}")
    show['Upside_30D'] = show['Upside_30D'].apply(lambda x: f"{x:+.1%}")
    show['Smart_ROI'] = show['Smart_ROI'].apply(lambda x: f"{x:+.1%}")
    show['DCA_ROI'] = show['DCA_ROI'].apply(lambda x: f"{x:+.1%}")
    show['Alpha'] = show['Alpha'].apply(lambda x: f"{x:+.1%}")
    
    st.dataframe(show[['Ticker', 'Current', 'Pred_30D', 'Upside_30D', 'Smart_ROI', 'DCA_ROI', 'Alpha']].style.map(
        lambda x: 'color: #00FF7F' if '+' in str(x) and float(str(x).strip('%+')) > 0 else 'color: white',
        subset=['Upside_30D', 'Alpha']
    ))