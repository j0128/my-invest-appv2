import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 0. 全局設定 ---
st.set_page_config(page_title="Alpha 10.2: 混合智能版", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #444; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] {gap: 5px;}
    .stTabs [data-baseweb="tab"] {height: 50px; background-color: #1E1E1E; border-radius: 5px 5px 0 0; color: white;}
    .stTabs [aria-selected="true"] {background-color: #00BFFF; color: black;}
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
        # 估算 Implied Price based on Forward PE (若無 EPS 數據則回傳 None)
        fwd_pe = info.get('forwardPE')
        # 這裡簡單推算：若有 Forward PE，我們假設這是市場對明年的共識價格基礎
        # 嚴謹的算法需要 Forward EPS，這裡我們直接抓取 Target Mean 作為 DCF/PE 的綜合代表
        
        return {
            'Target_Mean': info.get('targetMeanPrice'), # Wall St. DCF/PE Consensus
            'Forward_PE': fwd_pe,
            'Trailing_PE': info.get('trailingPE'),
            'PEG': info.get('pegRatio'),
            'Inst_Held': info.get('heldPercentInstitutions'),
            'Insider_Held': info.get('heldPercentInsiders'),
            'Short_Ratio': info.get('shortRatio'),
            'Current_Ratio': info.get('currentRatio'),
            'Debt_Equity': info.get('debtToEquity'),
            'ROE': info.get('returnOnEquity'),
            'Rule_40': (info.get('revenueGrowth',0) + info.get('profitMargins',0))*100 if info.get('revenueGrowth') else None
        }
    except: return {}

# --- 2. 戰略運算 (AI & 綜合預測) ---

def train_rf_model(df_close, ticker, days_forecast=22):
    """輕量化隨機森林 (Lightweight Random Forest)"""
    try:
        if ticker not in df_close.columns: return None
        
        # 特徵工程
        df = pd.DataFrame(index=df_close.index)
        df['Close'] = df_close[ticker]
        df['Ret'] = df['Close'].pct_change()
        df['Vol'] = df['Ret'].rolling(20).std()
        df['SMA'] = df['Close'].rolling(20).mean()
        
        # 加入宏觀特徵 (若有)
        if '^VIX' in df_close.columns: df['VIX'] = df_close['^VIX']
        if '^TNX' in df_close.columns: df['TNX'] = df_close['^TNX']
        
        # Target
        df['Target'] = df['Close'].shift(-days_forecast)
        df = df.dropna()
        
        if len(df) < 60: return None # 數據太少不訓練
        
        X = df.drop(columns=['Target', 'Close'])
        y = df['Target']
        
        # 訓練 (限制樹的數量與深度以提升速度)
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)
        
        # 預測
        latest_X = X.iloc[[-1]]
        return model.predict(latest_X)[0]
    except: return None

def calc_targets_composite(ticker, df_close, df_high, df_low, f_data, days_forecast=22):
    if ticker not in df_close.columns: return None
    c = df_close[ticker]; h = df_high[ticker]; l = df_low[ticker]
    if len(c) < 100: return None
    
    # 1. ATR (物理極限)
    try:
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        t_atr = c.iloc[-1] + (atr * np.sqrt(days_forecast))
    except: t_atr = None
    
    # 2. Monte Carlo (統計中樞)
    try:
        mu = c.pct_change().mean()
        t_mc = c.iloc[-1] * ((1 + mu)**days_forecast)
    except: t_mc = None
    
    # 3. Fibonacci (心理阻力)
    try:
        recent = c.iloc[-60:]
        t_fib = recent.max() + (recent.max() - recent.min()) * 0.618 
    except: t_fib = None
    
    # 4. Fundamental (DCF / Forward PE)
    # 使用分析師目標價作為基本面綜合指標
    t_fund = f_data.get('Target_Mean')
    
    # 5. Random Forest (AI 預測)
    t_rf = train_rf_model(df_close, ticker, days_forecast)
    
    # 綜合平均 (Composite Mean)
    targets = [t for t in [t_atr, t_mc, t_fib, t_fund, t_rf] if t is not None and not pd.isna(t)]
    t_avg = sum(targets) / len(targets) if targets else None
    
    return {"ATR": t_atr, "MC": t_mc, "Fib": t_fib, "Fund": t_fund, "RF": t_rf, "Avg": t_avg}

def run_backtest_lab(ticker, df_close, df_high, df_low, days_ago=22):
    """全模組回測 (包含 AI 重訓練)"""
    if ticker not in df_close.columns or len(df_close) < 250: return None
    
    # 切分過去數據
    idx_past = len(df_close) - days_ago - 1
    p_now = df_close[ticker].iloc[-1] # 真實的今天價格
    
    df_past = df_close.iloc[:idx_past+1]
    
    # 1. 回測 RF (用過去數據重新訓練)
    past_rf = train_rf_model(df_past, ticker, days_ago)
    
    # 2. 回測 ATR
    c_slice = df_close[ticker].iloc[:idx_past+1]
    h_slice = df_high[ticker].iloc[:idx_past+1]
    l_slice = df_low[ticker].iloc[:idx_past+1]
    
    tr = pd.concat([h_slice-l_slice], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    past_atr = c_slice.iloc[-1] + (atr * np.sqrt(days_ago))
    
    # 3. 回測 MC (簡化)
    past_mc = c_slice.iloc[-1] * ((1 + c_slice.pct_change().mean())**days_ago)
    
    # 綜合回測值
    valid_past = [x for x in [past_rf, past_atr, past_mc] if x is not None]
    if not valid_past: return None
    
    past_avg = sum(valid_past) / len(valid_past)
    err = (past_avg - p_now) / p_now
    
    return {"Past_Pred": past_avg, "Error": err, "Price_Now": p_now}

def calc_mvrv_z(series):
    if len(series) < 200: return None
    sma200 = series.rolling(200).mean()
    std200 = series.rolling(200).std()
    return (series - sma200) / std200

def analyze_trend_multi(series):
    if series is None or len(series) < 126: return {}
    y = series.iloc[-126:].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    p_now = series.iloc[-1]
    sma200 = series.rolling(200).mean().iloc[-1]
    status = "🔥 多頭" if p_now > sma200 else "🛑 空頭"
    if p_now < sma200 and p_now > sma200 * 0.9: status = "📉 弱勢"
    
    return {"p_1m": model.predict([[len(y)+22]])[0].item(), "p_now": p_now, "status": status}

def calc_kelly(trend_status):
    win = 0.65 if "多頭" in trend_status else 0.45
    return max(0, (win * 2.0 - 1) / 1.0 * 0.5)

def calc_obv(close, volume):
    if volume is None: return None
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

# --- 3. 財務計算 (保留不變) ---
def run_traffic_light(series):
    sma200 = series.rolling(200).mean()
    df = pd.DataFrame({'Close': series, 'SMA200': sma200})
    df['Signal'] = np.where(df['Close'] > df['SMA200'], 1, 0)
    df['Strategy'] = (1 + df['Close'].pct_change() * df['Signal'].shift(1)).cumprod()
    df['BuyHold'] = (1 + df['Close'].pct_change()).cumprod()
    return df['Strategy'], df['BuyHold']

def calc_coast_fire(age, r_age, net, save, rate, inf):
    years = r_age - age
    real = (1 + rate/100)/(1 + inf/100) - 1
    data = []
    bal = net
    for y in range(years+1):
        data.append({"Age": age+y, "Balance": bal})
        bal = bal*(1+real) + save*12
    return bal, pd.DataFrame(data)

def calc_mortgage(amt, yrs, rate):
    r = rate/100/12; m = yrs*12
    pmt = amt * (r * (1 + r)**m) / ((1 + r)**m - 1) if r > 0 else amt/m
    return pmt, pmt*m - amt

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
        st.metric("總資產 (Est.)", f"${total_value:,.0f}")
        if st.button("🚀 啟動混合智能", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False): return

    with st.spinner("🦅 Alpha 10.2 正在執行 AI + 基本面運算..."):
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_macro = fetch_fred_macro(fred_key)
        adv_data = {t: get_advanced_info(t) for t in tickers_list}

    if df_close.empty: st.error("❌ 無數據"); st.stop()

    # --- TABS ---
    t1, t2, t3, t4, t5, t6 = st.tabs(["🦅 戰略戰情", "🐋 深度籌碼", "🔍 個股體檢", "🚦 策略回測", "💰 CFO 財報", "🏠 房貸目標"])

    # === TAB 1: 戰略 ===
    with t1:
        st.subheader("1. 宏觀與總表")
        vix = df_close['^VIX'].iloc[-1] if '^VIX' in df_close.columns else 0
        liq = df_macro['Net_Liquidity'].iloc[-1] if df_macro is not None else 0
        try: cg = (df_close['HG=F'].iloc[-1]/df_close['GC=F'].iloc[-1])*1000
        except: cg = 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💧 淨流動性", f"${liq:.2f}T")
        c2.metric("🌪️ VIX", f"{vix:.2f}", delta_color="inverse")
        c3.metric("🏭 銅金比", f"{cg:.2f}")
        c4.metric("持倉數", len(tickers_list))

        if df_macro is not None:
            fig_liq = px.line(df_macro, y='Net_Liquidity', title='聯準會流動性', height=250)
            st.plotly_chart(fig_liq, use_container_width=True)

        st.markdown("#### 📊 持倉戰略總表")
        summary = []
        for t in tickers_list:
            if t not in df_close.columns: continue
            trend = analyze_trend_multi(df_close[t])
            mvrv = calc_mvrv_z(df_close[t])
            mvrv_val = mvrv.iloc[-1] if mvrv is not None else 0
            
            # 計算包含 RF 與 Fund 的綜合目標
            targets = calc_targets_composite(t, df_close, df_high, df_low, adv_data.get(t,{}), 22)
            
            summary.append({
                "代號": t, "現價": f"${trend['p_now']:.2f}", "狀態": trend['status'],
                "MVRV (Z)": f"{mvrv_val:.2f}", 
                "Kelly": f"{calc_kelly(trend['status'])*100:.0f}%",
                "綜合預測": f"${targets['Avg']:.2f}" if targets and targets['Avg'] else "-"
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
        
        st.markdown("---")
        st.subheader("2. 個股戰略雷達")
        
        for t in tickers_list:
            if t not in df_close.columns: continue
            info = adv_data.get(t, {})
            targets = calc_targets_composite(t, df_close, df_high, df_low, info, 22)
            bt = run_backtest_lab(t, df_close, df_high, df_low, 22)
            obv = calc_obv(df_close[t], df_vol[t])
            mvrv_s = calc_mvrv_z(df_close[t])
            mvrv = mvrv_s.iloc[-1] if mvrv_s is not None else 0
            
            t_avg = f"${targets['Avg']:.2f}" if targets and targets['Avg'] else "-"
            
            with st.expander(f"🦅 {t} | MVRV: {mvrv:.2f} | 綜合: {t_avg}", expanded=False):
                k1, k2, k3 = st.columns([2, 1, 1])
                with k1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_close.index[-126:], y=df_close[t].iloc[-126:], name='Price', line=dict(color='#00FF7F')))
                    if obv is not None:
                        fig.add_trace(go.Scatter(x=df_close.index[-126:], y=obv.iloc[-126:], name='OBV', line=dict(color='#FFD700', width=1), yaxis='y2'))
                    fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), yaxis2=dict(overlaying='y', side='right', showgrid=False))
                    st.plotly_chart(fig, use_container_width=True)
                with k2:
                    st.markdown("#### 🤖 五角定位 (1M)")
                    if targets:
                        st.write(f"**ATR (物理):** ${targets['ATR']:.2f}" if targets['ATR'] else "-")
                        st.write(f"**MC (機率):** ${targets['MC']:.2f}" if targets['MC'] else "-")
                        st.write(f"**Fib (心理):** ${targets['Fib']:.2f}" if targets['Fib'] else "-")
                        st.write(f"**RF (AI):** ${targets['RF']:.2f}" if targets['RF'] else "-")
                        st.write(f"**Fund (DCF):** ${targets['Fund']}" if targets['Fund'] else "N/A")
                    
                    if bt:
                        st.markdown("#### 🧪 綜合回測")
                        err = bt['Error']
                        c = "green" if abs(err)<0.05 else "red"
                        st.markdown(f"1月前預測誤差: <span style='color:{c}'>{err:.1%}</span>", unsafe_allow_html=True)
                        st.caption(f"當時預測 ${bt['Past_Pred']:.2f} vs 今日 ${bt['Price_Now']:.2f}")
                with k3:
                    st.markdown("#### 💎 指標")
                    st.metric("Rule 40", f"{info.get('Rule_40', 0):.1f}" if info.get('Rule_40') else "-")
                    st.metric("Forward PE", f"{info.get('Forward_PE')}")

    # === TAB 2~6 (保留不變) ===
    with t2:
        st.subheader("🐋 籌碼")
        dat = [{"代號":t, "機構": f"{adv_data[t].get('Inst_Held',0)*100:.0f}%"} for t in tickers_list if t in df_close.columns]
        st.dataframe(pd.DataFrame(dat), use_container_width=True)
        
    with t3:
        st.subheader("🔍 體質")
        dat = [{"代號":t, "PEG": f"{adv_data[t].get('PEG',0)}", "ROE": f"{adv_data[t].get('ROE',0)*100:.1f}%" if adv_data[t].get('ROE') else "-"} for t in tickers_list if t in df_close.columns]
        st.dataframe(pd.DataFrame(dat), use_container_width=True)
        
    with t4:
        st.subheader("🚦 回測")
        for t in tickers_list:
            if t in df_close.columns:
                s, b = run_traffic_light(df_close[t])
                if s is not None: st.line_chart(pd.concat([s, b], axis=1))

    with t5:
        st.subheader("💰 CFO")
        c1,c2 = st.columns(2)
        inc=c1.number_input("月收",80000); exp=c1.number_input("月支",40000)
        c1.metric("儲蓄率", f"{(inc-exp)/inc:.1%}")
        ast=c2.number_input("資產",15000000); lia=c2.number_input("負債",8000000)
        c2.metric("淨值", f"${ast-lia:,.0f}")

    with t6:
        st.subheader("🏠 房貸")
        amt=st.number_input("貸",10000000); rt=st.number_input("率",2.2)
        pmt,_=calc_mortgage(amt,30,rt)
        st.metric("月付", f"${pmt:,.0f}")

if __name__ == "__main__":
    main()