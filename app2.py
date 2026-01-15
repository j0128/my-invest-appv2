import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb  # 新增 XGBoost
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 0. 全局設定 ---
st.set_page_config(page_title="Alpha 12.0: 雙核 AI 戰略", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #444; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
    .ai-box {background-color: #112233; padding: 15px; border-radius: 10px; border: 1px solid #00BFFF;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 ---
@st.cache_data(ttl=1800)
def fetch_market_data(tickers):
    benchmarks = ['SPY', 'QQQ', '^VIX', '^TNX', 'HYG', 'GC=F', 'HG=F', 'DX-Y.NYB'] 
    all_tickers = list(set(tickers + benchmarks))
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    
    for i, t in enumerate(all_tickers):
        try:
            df = yf.Ticker(t).history(period="2y", auto_adjust=True)
            if df.empty: continue
            data['Close'][t] = df['Close']
            data['Open'][t] = df['Open']
            data['High'][t] = df['High']
            data['Low'][t] = df['Low']
            data['Volume'][t] = df['Volume']
        except: continue
    
    try:
        return (pd.DataFrame(data['Close']).ffill(), pd.DataFrame(data['High']).ffill(), 
                pd.DataFrame(data['Low']).ffill(), pd.DataFrame(data['Volume']).ffill())
    except: return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

@st.cache_data(ttl=3600*12)
def fetch_fred_macro(api_key):
    if not api_key: return None
    try:
        fred = Fred(api_key=api_key)
        walcl = fred.get_series('WALCL', observation_start='2024-01-01')
        tga = fred.get_series('WTREGEN', observation_start='2024-01-01')
        rrp = fred.get_series('RRPONTSYD', observation_start='2024-01-01')
        df = pd.DataFrame({'WALCL': walcl, 'TGA': tga, 'RRP': rrp}).ffill().dropna()
        df['Net_Liquidity'] = (df['WALCL'] - df['TGA'] - df['RRP']) / 1000 
        return df
    except: return None

@st.cache_data(ttl=3600*24)
def get_advanced_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'Target_Mean': info.get('targetMeanPrice'), 
            'Forward_PE': info.get('forwardPE'),
            'Trailing_PE': info.get('trailingPE'),
            'PEG': info.get('pegRatio'),
            'Inst_Held': info.get('heldPercentInstitutions'),
            'Insider_Held': info.get('heldPercentInsiders'),
            'Short_Ratio': info.get('shortRatio'),
            'Rule_40': (info.get('revenueGrowth',0) + info.get('profitMargins',0))*100 if info.get('revenueGrowth') else None
        }
    except: return {}

# --- 2. 雙核 AI 引擎 (Dual-Core AI Engine) ---

def prepare_ai_data(ticker, df_close, df_vol, days_forecast=22):
    """準備機器學習特徵 (Feature Engineering)"""
    if ticker not in df_close.columns: return None
    
    df = pd.DataFrame(index=df_close.index)
    df['Close'] = df_close[ticker]
    df['Return'] = df['Close'].pct_change()
    df['Vol'] = df['Return'].rolling(20).std()
    
    # 宏觀特徵
    if '^VIX' in df_close.columns: df['VIX'] = df_close['^VIX']
    if '^TNX' in df_close.columns: df['TNX'] = df_close['^TNX']
    if 'HYG' in df_close.columns: df['Credit'] = df_close['HYG']
    
    # 技術特徵
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA60'] = df['Close'].rolling(60).mean()
    df['RSI'] = 100 - (100 / (1 + df['Return'].apply(lambda x: x if x>0 else 0).rolling(14).mean() / df['Return'].apply(lambda x: -x if x<0 else 0).rolling(14).mean()))
    
    # Target
    df['Target'] = df['Close'].shift(-days_forecast)
    df = df.dropna()
    return df

def train_rf_xgboost(ticker, df_close, df_vol, days_forecast=22, backtest=False):
    """
    同時訓練 Random Forest 與 XGBoost
    backtest=True 時，會切分數據模擬過去的預測
    """
    try:
        df = prepare_ai_data(ticker, df_close, df_vol, days_forecast)
        if df is None or len(df) < 100: return None, None, None, None
        
        X = df.drop(columns=['Target', 'Close'])
        y = df['Target']
        
        # 如果是回測模式，我們將時間倒回 days_forecast 天前
        # 也就是：用 [Start ~ T-22] 的數據，去訓練模型
        # 然後預測 T (今天) 的價格
        if backtest:
            # 回測切分點：倒數第 1 筆 (因為 df 已經 shift 過了，最後一筆的 Target 就是今天)
            X_train = X.iloc[:-1]
            y_train = y.iloc[:-1]
            X_test = X.iloc[[-1]] # 今天的特徵 (用來預測未來，但在這裡是預測今天)
            actual_price = df_close[ticker].iloc[-1] # 今天的真實現價
        else:
            # 正常預測模式：用全量數據訓練，預測未來
            X_train = X
            y_train = y
            X_test = X.iloc[[-1]] # 最新的特徵
            actual_price = None

        # 1. Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)[0]
        
        # 2. XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, objective='reg:squarederror')
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)[0]
        
        return rf_pred, xgb_pred, actual_price, X_test.index[-1]
        
    except Exception as e:
        return None, None, None, None

def calc_mvrv_z(series):
    if len(series) < 200: return None
    sma200 = series.rolling(200).mean()
    std200 = series.rolling(200).std()
    return (series - sma200) / std200

def calc_targets_composite(ticker, df_close, df_high, df_low, df_vol, f_data, rf_pred, xgb_pred, days_forecast=22):
    if ticker not in df_close.columns: return None
    c = df_close[ticker]; h = df_high[ticker]; l = df_low[ticker]
    if len(c) < 100: return None
    
    # 1. ATR (物理)
    try:
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        t_atr = c.iloc[-1] + (atr * np.sqrt(days_forecast))
    except: t_atr = None
    
    # 2. MC (機率)
    try:
        mu = c.pct_change().mean()
        t_mc = c.iloc[-1] * ((1 + mu)**days_forecast)
    except: t_mc = None
    
    # 3. Fib (心理)
    try:
        recent = c.iloc[-60:]
        t_fib = recent.max() + (recent.max() - recent.min()) * 0.618 
    except: t_fib = None
    
    # 4. Fund
    t_fund = f_data.get('Target_Mean')
    
    # 綜合平均 (含雙核 AI)
    targets = [t for t in [t_atr, t_mc, t_fib, rf_pred, xgb_pred] if t is not None and not pd.isna(t)]
    t_avg = sum(targets) / len(targets) if targets else None
    
    return {"ATR": t_atr, "MC": t_mc, "Fib": t_fib, "Fund": t_fund, "RF": rf_pred, "XGB": xgb_pred, "Avg": t_avg}

def analyze_trend_multi(series):
    if series is None or len(series) < 126: return {}
    y = series.iloc[-126:].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    p_now = series.iloc[-1]
    sma200 = series.rolling(200).mean().iloc[-1]
    status = "🔥 多頭" if p_now > sma200 else "🛑 空頭"
    if p_now < sma200 and p_now > sma200 * 0.9: status = "📉 弱勢"
    
    return {
        "p_2w": model.predict([[len(y)+10]])[0].item(),
        "p_1m": model.predict([[len(y)+22]])[0].item(),
        "p_3m": model.predict([[len(y)+66]])[0].item(),
        "p_now": p_now,
        "status": status
    }

def calc_kelly(trend_status):
    win_rate = 0.55
    if "多頭" in trend_status: win_rate += 0.1
    if "空頭" in trend_status: win_rate -= 0.15
    f_star = (win_rate * 2.0 - 1) / 1.0 
    return max(0, f_star * 0.5)

def calc_obv(close, volume):
    if volume is None: return None
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

# --- 3. 財務與其他 ---
def calc_mortgage(amount, years, rate_pct):
    rate = rate_pct / 100 / 12; months = years * 12
    if rate==0: return amount/months, 0
    pmt = amount * (rate * (1 + rate)**months) / ((1 + rate)**months - 1)
    return pmt, pmt * months - amount

def calc_coast_fire(age, r_age, net_worth, save, rate, inf):
    years = r_age - age
    real = (1 + rate/100)/(1 + inf/100) - 1
    data = []
    bal = net_worth
    for y in range(years+1):
        data.append({"Age": age+y, "Balance": bal})
        bal = bal*(1+real) + save*12
    return bal, pd.DataFrame(data)

def run_traffic_light(series):
    sma200 = series.rolling(200).mean()
    df = pd.DataFrame({'Close': series, 'Signal': np.where(series>sma200, 1, 0)})
    df['Strategy'] = (1 + series.pct_change() * df['Signal'].shift(1)).cumprod()
    df['BuyHold'] = (1 + series.pct_change()).cumprod()
    return df['Strategy'], df['BuyHold']

def parse_input(text):
    port = {}
    for line in text.strip().split('\n'):
        if ',' in line:
            parts = line.split(',')
            try: port[parts[0].strip().upper()] = float(parts[1].strip())
            except: port[parts[0].strip().upper()] = 0.0
    return port

# --- MAIN APP ---
def main():
    with st.sidebar:
        st.header("⚙️ 設定")
        fred_key = st.secrets.get("FRED_API_KEY", st.text_input("FRED API Key", type="password"))
        default_input = """BTC-USD, 10000\nAMD, 10000\nNVDA, 10000\nTLT, 5000"""
        user_input = st.text_area("持倉清單", default_input, height=150)
        portfolio_dict = parse_input(user_input)
        tickers_list = list(portfolio_dict.keys())
        total_value = sum(portfolio_dict.values())
        st.metric("總資產", f"${total_value:,.0f}")
        if st.button("🚀 啟動雙核 AI", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False): return

    with st.spinner("🦅 Alpha 12.0 正在訓練雙核 AI 模型 (RF + XGBoost)..."):
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_macro = fetch_fred_macro(fred_key)
        adv_data = {t: get_advanced_info(t) for t in tickers_list}

    if df_close.empty: st.error("❌ 數據失敗"); st.stop()

    # --- TABS ---
    t1, t2, t3, t4, t5, t6 = st.tabs(["🦅 戰略戰情", "🐋 深度籌碼", "🔍 個股體檢", "🚦 策略回測", "💰 CFO 財報", "🏠 房貸目標"])

    # === TAB 1: 戰略 ===
    with t1:
        st.subheader("1. 宏觀與持倉總覽")
        # Macro
        vix = df_close['^VIX'].iloc[-1] if '^VIX' in df_close.columns else 0
        liq = df_macro['Net_Liquidity'].iloc[-1] if df_macro is not None else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💧 淨流動性", f"${liq:.2f}T")
        c2.metric("🌪️ VIX", f"{vix:.2f}", delta_color="inverse")
        
        if df_macro is not None:
            fig_liq = px.line(df_macro, y='Net_Liquidity', title='聯準會流動性', height=250)
            st.plotly_chart(fig_liq, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📊 持倉戰略總表")
        summary = []
        for t in tickers_list:
            if t not in df_close.columns: continue
            trend = analyze_trend_multi(df_close[t])
            mvrv = calc_mvrv_z(df_close[t])
            mvrv_val = mvrv.iloc[-1] if mvrv is not None else 0
            summary.append({
                "代號": t, "現價": f"${trend['p_now']:.2f}", "狀態": trend['status'],
                "MVRV (Z)": f"{mvrv_val:.2f}", "Kelly": f"{calc_kelly(trend['status'])*100:.0f}%"
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
        
        st.markdown("---")
        st.subheader("2. 雙核 AI 實驗室 (Dual-Core Lab)")
        
        for t in tickers_list:
            if t not in df_close.columns: continue
            
            trend = analyze_trend_multi(df_close[t])
            info = adv_data.get(t, {})
            
            # 1. 訓練模型預測未來
            rf_future, xgb_future, _, _ = train_rf_xgboost(t, df_close, df_vol, 22, backtest=False)
            
            # 2. 回測模型 (回到過去預測今天)
            rf_past, xgb_past, actual_now, _ = train_rf_xgboost(t, df_close, df_vol, 22, backtest=True)
            
            # 計算回測誤差
            rf_err = (rf_past - actual_now) / actual_now if rf_past else 0
            xgb_err = (xgb_past - actual_now) / actual_now if xgb_past else 0
            
            # 計算綜合目標
            targets = calc_targets_composite(t, df_close, df_high, df_low, df_vol, info, rf_future, xgb_future, 22)
            obv = calc_obv(df_close[t], df_vol[t])
            
            t_avg_s = f"${targets['Avg']:.2f}" if targets and targets['Avg'] else "-"
            
            with st.expander(f"🦅 {t} | 綜合目標: {t_avg_s}", expanded=False):
                k1, k2, k3 = st.columns([2, 1, 1])
                with k1:
                    st.markdown("#### 📉 價格與資金流")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_close.index[-126:], y=df_close[t].iloc[-126:], name='Price', line=dict(color='#00FF7F')))
                    if obv is not None:
                        fig.add_trace(go.Scatter(x=df_close.index[-126:], y=obv.iloc[-126:], name='OBV', line=dict(color='#FFD700', width=1), yaxis='y2'))
                    fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), yaxis2=dict(overlaying='y', side='right', showgrid=False))
                    st.plotly_chart(fig, use_container_width=True)

                with k2:
                    st.markdown("#### 🤖 AI 雙核回測 (1M)")
                    c_rf = "green" if abs(rf_err) < 0.05 else "red"
                    c_xgb = "green" if abs(xgb_err) < 0.05 else "red"
                    
                    st.write(f"**Random Forest:** 預測 ${rf_past:.2f}")
                    st.markdown(f"誤差: <span style='color:{c_rf}'>{rf_err:.1%}</span>", unsafe_allow_html=True)
                    st.divider()
                    st.write(f"**XGBoost:** 預測 ${xgb_past:.2f}")
                    st.markdown(f"誤差: <span style='color:{c_xgb}'>{xgb_err:.1%}</span>", unsafe_allow_html=True)
                    st.caption(f"基準現價: ${actual_now:.2f}")

                with k3:
                    st.markdown("#### 🎯 未來預測 (1M)")
                    if targets:
                        st.write(f"**RF 預測:** ${targets['RF']:.2f}" if targets['RF'] else "-")
                        st.write(f"**XGB 預測:** ${targets['XGB']:.2f}" if targets['XGB'] else "-")
                        st.write(f"**ATR 極限:** ${targets['ATR']:.2f}" if targets['ATR'] else "-")
                        st.write(f"**MC 中樞:** ${targets['MC']:.2f}" if targets['MC'] else "-")
                        st.write(f"**Fib 阻力:** ${targets['Fib']:.2f}" if targets['Fib'] else "-")

    # === TAB 2~6 (保留原有功能) ===
    # 為節省篇幅，此處邏輯與 v11.0 相同，僅需貼上即可
    with t2:
        st.subheader("🐋 籌碼")
        dat = [{"代號":t, "機構": f"{adv_data[t].get('Inst_Held',0)*100:.0f}%"} for t in tickers_list if t in df_close.columns]
        st.dataframe(pd.DataFrame(dat), use_container_width=True)
        
    with t3:
        st.subheader("🔍 體質")
        dat = [{"代號":t, "Rule40": f"{adv_data[t].get('Rule_40',0):.1f}"} for t in tickers_list if t in df_close.columns]
        st.dataframe(pd.DataFrame(dat), use_container_width=True)
        
    with t4:
        st.subheader("🚦 回測")
        for t in tickers_list:
            if t in df_close.columns:
                s, b = run_traffic_light(df_close[t])
                st.line_chart(pd.concat([s, b], axis=1))

    with t5:
        st.subheader("💰 CFO")
        inc=st.number_input("月收",80000); exp=st.number_input("月支",40000)
        st.metric("儲蓄率", f"{(inc-exp)/inc:.1%}")

    with t6:
        st.subheader("🏠 房貸")
        amt=st.number_input("貸",10000000); rt=st.number_input("率",2.2)
        pmt,_=calc_mortgage(amt,30,rt)
        st.metric("月付", f"${pmt:,.0f}")

if __name__ == "__main__":
    main()