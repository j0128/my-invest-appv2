import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 0. 全局設定 ---
st.set_page_config(page_title="Alpha 4.2: 戰略修正版", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #262730; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- 1. 數據引擎 ---
@st.cache_data(ttl=1800)
def fetch_market_data(tickers):
    benchmarks = ['SPY', 'QQQ', 'BTC-USD', '^VIX', '^TNX', 'HYG']
    all_tickers = list(set(tickers + benchmarks))
    
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    progress_bar = st.progress(0, text="☁️ Alpha 正在連線華爾街資料庫...")
    
    for i, t in enumerate(all_tickers):
        try:
            progress_bar.progress((i + 1) / len(all_tickers), text=f"下載: {t} ...")
            df = yf.Ticker(t).history(period="2y", auto_adjust=True)
            if df.empty: continue
            
            data['Close'][t] = df['Close']
            data['Open'][t] = df['Open']
            data['High'][t] = df['High']
            data['Low'][t] = df['Low']
            data['Volume'][t] = df['Volume']
        except: continue
            
    progress_bar.empty()
    return (pd.DataFrame(data['Close']).ffill(), 
            pd.DataFrame(data['Open']).ffill(), 
            pd.DataFrame(data['High']).ffill(), 
            pd.DataFrame(data['Low']).ffill(),
            pd.DataFrame(data['Volume']).ffill())

@st.cache_data(ttl=3600*12)
def fetch_fred_liquidity(api_key):
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
        rev_growth = info.get('revenueGrowth')
        profit_margin = info.get('profitMargins')
        rule_of_40 = (rev_growth + profit_margin) * 100 if (rev_growth and profit_margin) else None
        
        return {
            'Rule40': rule_of_40,
            'PEG': info.get('pegRatio'),
            'Inst_Held': info.get('heldPercentInstitutions'),
            'Short_Ratio': info.get('shortRatio'),
            'Target_Mean': info.get('targetMeanPrice'),
            'PE': info.get('forwardPE')
        }
    except: return {}

# --- 2. 核心運算 ---
def format_number(num):
    if num is None: return "N/A"
    if abs(num) >= 1_000_000: return f"{num/1_000_000:.2f}M"
    elif abs(num) >= 1_000: return f"{num/1_000:.2f}K"
    return f"{num:.2f}"

def calc_rrg_metrics(df_close, tickers, benchmark='SPY'):
    if benchmark not in df_close.columns: return pd.DataFrame()
    rrg_data = []
    bench_close = df_close[benchmark]
    for t in tickers:
        if t not in df_close.columns or t == benchmark: continue
        rs = df_close[t] / bench_close
        rs_mean_short = rs.rolling(10).mean()
        rs_mean_long = rs.rolling(60).mean()
        if len(rs_mean_short.dropna()) < 60: continue
        
        rs_ratio = (rs_mean_short / rs_mean_long * 100).iloc[-1]
        rs_ratio_series = rs_mean_short / rs_mean_long * 100
        change = rs_ratio_series.iloc[-1] - rs_ratio_series.iloc[-10]
        rs_momentum = (change * 5) + 100
        
        if rs_ratio > 100 and rs_momentum > 100: quadrant = "🟢 領先"
        elif rs_ratio > 100 and rs_momentum < 100: quadrant = "🟡 轉弱"
        elif rs_ratio < 100 and rs_momentum < 100: quadrant = "🔴 落後"
        else: quadrant = "🔵 改善"
        
        rrg_data.append({'Ticker': t, 'RS_Ratio': rs_ratio, 'RS_Momentum': rs_momentum, 'Quadrant': quadrant})
    return pd.DataFrame(rrg_data)

def calc_mvrv_z_score(series):
    """
    計算 MVRV Z-Score (或價格偏離度 Z-Score)
    Z = (Price - 200SMA) / StdDev(200)
    """
    try:
        sma200 = series.rolling(200).mean()
        std200 = series.rolling(200).std()
        z_score = (series - sma200) / std200
        return z_score
    except: return None

def calc_targets(close, high, low):
    if len(close) < 60: return None, None, None
    try:
        # ATR
        prev_close = close.shift(1)
        tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        t_atr = close.iloc[-1] + atr * np.sqrt(22) * 1.2
        
        # MC
        returns = close.pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        sim_last = []
        for _ in range(300):
            prices = [close.iloc[-1]]
            vol = np.random.normal(mu, sigma, 22)
            for v in vol: prices.append(prices[-1]*(1+v))
            sim_last.append(prices[-1])
        t_mc = np.percentile(sim_last, 50)
        
        # Fib
        rw = close.iloc[-60:]
        t_fib = rw.max() + (rw.max() - rw.min()) * 0.618
        
        return t_atr, t_mc, t_fib
    except: return None, None, None

def analyze_trend(series):
    if series is None or len(series) < 200: return None
    p_now = series.iloc[-1]
    sma200 = series.rolling(200).mean().iloc[-1]
    ema20 = series.ewm(span=20).mean().iloc[-1]
    
    # 修正後的邏輯
    status = "🛡️ 區間"
    if p_now > sma200 and p_now > ema20: status = "🔥 強勢"
    elif p_now > sma200 and p_now < ema20: status = "⚠️ 整理"
    elif p_now < sma200 and p_now > sma200 * 0.85: status = "📉 回調 (洗盤)" # 修正這裡
    elif p_now < sma200 * 0.85: status = "🛑 熊市"
    
    y = series.dropna().values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    p_2w = model.predict([[len(y)+10]])[0].item()
    p_1m = model.predict([[len(y)+22]])[0].item()
    
    return {"status": status, "p_now": p_now, "p_2w": p_2w, "p_1m": p_1m, "sma200": sma200}

def calc_fund_flow(close, volume):
    if volume is None or volume.empty: return None, None
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    y = obv.values[-20:].reshape(-1, 1)
    x = np.arange(20).reshape(-1, 1)
    slope = LinearRegression().fit(x, y).coef_[0].item()
    return slope, obv

def parse_input(text):
    port = {}
    for line in text.strip().split('\n'):
        if ',' in line:
            parts = line.split(',')
            try: port[parts[0].strip().upper()] = float(parts[1].strip())
            except: port[parts[0].strip().upper()] = 0.0
    return port

# --- MAIN ---
def main():
    st.title("Alpha 4.2: 戰略修正版")
    st.caption("v4.2 | MVRV 回歸 | 趨勢判定優化 | 公式白皮書")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ 設定")
        fred_key = st.secrets.get("FRED_API_KEY", st.text_input("FRED API Key", type="password"))
        st.header("💼 資產")
        default_input = """BTC-USD, 10000
AMD, 10000
NVDA, 10000
PLTR, 5000"""
        user_input = st.text_area("清單", default_input, height=200)
        portfolio_dict = parse_input(user_input)
        tickers_list = list(portfolio_dict.keys())
        total_value = sum(portfolio_dict.values())
        st.metric("總估值", f"${total_value:,.0f}")
        if st.button("🚀 啟動全域掃描", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False): return

    with st.spinner("計算中..."):
        df_close, df_open, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_liquidity = fetch_fred_liquidity(fred_key)
        adv_data = {t: get_advanced_info(t) for t in tickers_list}

    if df_close.empty: st.error("No Data"); return

    # --- 1. RRG ---
    st.subheader("1. 資金流向 (RRG & Macro)")
    rrg_df = calc_rrg_metrics(df_close, tickers_list)
    if not rrg_df.empty:
        fig_rrg = px.scatter(rrg_df, x='RS_Ratio', y='RS_Momentum', color='Quadrant', text='Ticker', title="RRG 動能輪動 (vs SPY)", 
                             color_discrete_map={'🟢 領先': '#00FF7F', '🟡 轉弱': '#FFFF00', '🔴 落後': '#FF4B4B', '🔵 改善': '#00BFFF'})
        fig_rrg.add_vline(x=100, line_dash="dash", line_color="gray"); fig_rrg.add_hline(y=100, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_rrg, use_container_width=True)
    
    # --- 2. 深度審計 ---
    st.subheader("2. 深度審計 (Deep Audit)")
    
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        slope, obv = calc_fund_flow(df_close[ticker], df_vol[ticker])
        info = adv_data.get(ticker, {})
        t_atr, t_mc, t_fib = calc_targets(df_close[ticker], df_high[ticker], df_low[ticker])
        
        # MVRV Z-Score 計算
        z_score_series = calc_mvrv_z_score(df_close[ticker])
        z_now = z_score_series.iloc[-1] if z_score_series is not None else 0
        z_color = "red" if z_now > 2 else ("green" if z_now < 0 else "orange")
        
        with st.expander(f"📊 {ticker} - {trend['status']} | MVRV-Z: {z_now:.2f}", expanded=True):
            k1, k2, k3 = st.columns([2, 1, 1])
            
            with k1: # 圖表
                st.markdown("#### 📈 MVRV Z-Score (估值位階)")
                # 雙軸圖：價格 + Z-Score
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_close.index[-200:], y=df_close[ticker].iloc[-200:], name='Price', line=dict(color='white')))
                # 副圖 Z-Score
                fig.add_trace(go.Scatter(x=df_close.index[-200:], y=z_score_series.iloc[-200:], name='Z-Score', line=dict(color='cyan'), yaxis='y2'))
                fig.update_layout(height=300, yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Z-Score'))
                # 畫出 Z=0 和 Z=2 的線
                fig.add_hline(y=2, line_dash="dot", line_color="red", yref="y2", annotation_text="Overvalued")
                fig.add_hline(y=0, line_dash="dot", line_color="green", yref="y2", annotation_text="Fair Value")
                st.plotly_chart(fig, use_container_width=True)

            with k2: # 籌碼
                st.markdown("#### 🎯 目標價與籌碼")
                st.write(f"**Monte Carlo:** ${t_mc:.2f}")
                st.write(f"**Analyst:** ${info.get('Target_Mean')}" if info.get('Target_Mean') else "-")
                inst = info.get('Inst_Held')
                st.metric("機構持股", f"{inst*100:.1f}%" if inst else "-", delta="高度控盤" if inst and inst > 0.7 else "散戶多")
                st.metric("OBV 斜率", format_number(slope), "吸籌" if slope>0 else "出貨")

            with k3: # 基本面
                st.markdown("#### 💎 財務體質")
                r40 = info.get('Rule40')
                r40_icon = "✅" if r40 and r40>40 else "❌"
                st.metric("Rule of 40", f"{r40:.1f}" if r40 else "-", delta=r40_icon)
                st.metric("Forward P/E", f"{info.get('PE', 'N/A')}")
                st.caption(f"2週預測: ${trend['p_2w']:.2f}")

    # --- 3. 公式白皮書 ---
    st.markdown("---")
    st.header("3. 量化模型公式手冊 (Formulas)")
    
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            st.info("### 📐 MVRV Z-Score (估值偏離度)")
            st.latex(r'''Z = \frac{P_{now} - SMA_{200}}{\sigma_{200}}''')
            st.markdown("* **Z > 2.0:** 價格過熱 (Overvalued)，注意回調。\n* **Z < 0.0:** 價格低於長期均線 (Undervalued)，潛在買點。")
            
            st.divider()
            st.info("### 🌀 RRG 相對強度 (JdK RS-Ratio)")
            st.latex(r'''RS = \frac{Price_{Stock}}{Price_{SPY}} \times 100''')
            st.latex(r'''\text{Ratio} = \frac{MA_{10}(RS)}{MA_{60}(RS)} \times 100''')
            
        with c2:
            st.info("### 💎 Rule of 40 (SaaS 估值)")
            st.latex(r'''R_{40} = \text{Revenue Growth (\%)} + \text{Profit Margin (\%)}''')
            st.markdown("* **> 40:** 優質成長股 (如 PLTR, CRWD)。\n* **< 40:** 體質尚需改善。")
            
            st.divider()
            st.info("### 🎲 Monte Carlo (蒙地卡羅預測)")
            st.latex(r'''P_{t} = P_{t-1} \times e^{(\mu - \frac{\sigma^2}{2})\Delta t + \sigma \epsilon \sqrt{\Delta t}}''')
            st.markdown("透過幾何布朗運動 (GBM) 模擬 300 次未來路徑，取中位數 (P50) 為目標價。")

if __name__ == "__main__":
    main()