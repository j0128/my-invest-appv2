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
st.set_page_config(page_title="Alpha 7.0: 全域資金流戰略", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #444; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
    .explanation-box {background-color: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 5px solid #00BFFF;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 (Data Engine) ---
@st.cache_data(ttl=1800)
def fetch_market_data(tickers):
    # 強制加入宏觀基準: SPY(大盤), QQQ(科技), VIX(恐慌), TNX(長債), IRX(短債/Fed預期)
    benchmarks = ['SPY', 'QQQ', '^VIX', '^TNX', '^IRX', 'HYG'] 
    all_tickers = list(set(tickers + benchmarks))
    
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    progress_bar = st.progress(0, text="🦅 Alpha 7.0 正在掃描全域資金流...")
    
    for i, t in enumerate(all_tickers):
        try:
            progress_bar.progress((i + 1) / len(all_tickers), text=f"下載: {t} ...")
            # 抓取 2 年數據以計算長期 RRG 與 MVRV
            df = yf.Ticker(t).history(period="2y", auto_adjust=True)
            if df.empty: continue
            data['Close'][t] = df['Close']
            data['Open'][t] = df['Open']
            data['High'][t] = df['High']
            data['Low'][t] = df['Low']
            data['Volume'][t] = df['Volume']
        except: continue
    progress_bar.empty()
    return (pd.DataFrame(data['Close']).ffill(), pd.DataFrame(data['High']).ffill(), 
            pd.DataFrame(data['Low']).ffill(), pd.DataFrame(data['Volume']).ffill())

@st.cache_data(ttl=3600*12)
def fetch_fred_macro(api_key):
    if not api_key: return None
    try:
        fred = Fred(api_key=api_key)
        # 流動性公式: WALCL (Fed資產) - TGA (財政部帳戶) - RRP (逆回購)
        walcl = fred.get_series('WALCL', observation_start='2024-01-01')
        tga = fred.get_series('WTREGEN', observation_start='2024-01-01')
        rrp = fred.get_series('RRPONTSYD', observation_start='2024-01-01')
        df = pd.DataFrame({'WALCL': walcl, 'TGA': tga, 'RRP': rrp}).ffill().dropna()
        df['Net_Liquidity'] = (df['WALCL'] - df['TGA'] - df['RRP']) / 1000 # 兆
        return df
    except: return None

@st.cache_data(ttl=3600*24)
def get_advanced_metrics(ticker):
    """抓取基本面與機構數據"""
    try:
        info = yf.Ticker(ticker).info
        # Rule of 40 計算
        rev_g = info.get('revenueGrowth', 0)
        prof_m = info.get('profitMargins', 0)
        r40 = (rev_g + prof_m) * 100 if rev_g and prof_m else None
        
        return {
            'Target_Mean': info.get('targetMeanPrice'), # 華爾街共識
            'Forward_PE': info.get('forwardPE'),
            'Inst_Held': info.get('heldPercentInstitutions'), # 機構持股
            'Rule_40': r40,
            'PEG': info.get('pegRatio')
        }
    except: return {}

# --- 2. 核心運算模型 ---

# A. RRG 資金流向 (取代 Excel)
def calc_rrg(df_close, tickers, benchmark='SPY'):
    if benchmark not in df_close.columns: return pd.DataFrame()
    rrg_data = []
    bench = df_close[benchmark]
    
    for t in tickers:
        if t not in df_close.columns or t == benchmark: continue
        # 1. 相對強度 (RS)
        rs = df_close[t] / bench
        # 2. RS-Ratio (趨勢): 短期RS / 長期RS
        rs_mean_short = rs.rolling(10).mean()
        rs_mean_long = rs.rolling(60).mean()
        if len(rs_mean_short.dropna()) < 60: continue
        
        rs_ratio = (rs_mean_short / rs_mean_long * 100).iloc[-1]
        
        # 3. RS-Momentum (動能): Ratio 的變化率
        rs_ratio_series = rs_mean_short / rs_mean_long * 100
        rs_mom = ((rs_ratio_series.iloc[-1] - rs_ratio_series.iloc[-10]) * 5) + 100
        
        # 4. 象限
        if rs_ratio > 100 and rs_mom > 100: q = "🟢 領先 (Leading)"
        elif rs_ratio > 100 and rs_mom < 100: q = "🟡 轉弱 (Weakening)"
        elif rs_ratio < 100 and rs_mom < 100: q = "🔴 落後 (Lagging)"
        else: q = "🔵 改善 (Improving)"
        
        rrg_data.append({'Ticker': t, 'RS_Ratio': rs_ratio, 'RS_Momentum': rs_mom, 'Quadrant': q})
    return pd.DataFrame(rrg_data)

# B. MVRV Z-Score (估值偏離度)
def calc_mvrv_z(series):
    try:
        sma200 = series.rolling(200).mean()
        std200 = series.rolling(200).std()
        z = (series - sma200) / std200
        return z
    except: return None

# C. 四角定位 (v3 精準版) + 平均
def calc_targets_composite(close, high, low, f_data, days_forecast=22):
    if len(close) < 252: return None
    
    # 1. ATR (趨勢調整版)
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    
    # 線性預測未來均價
    y = close.iloc[-126:].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    price_projected = model.predict([[len(y) + days_forecast]])[0].item()
    t_atr = price_projected + (atr * np.sqrt(days_forecast))
    
    # 2. Monte Carlo (P50)
    returns = close.iloc[-252:].pct_change().dropna()
    mu, sigma = returns.mean(), returns.std()
    sims = []
    for _ in range(500):
        p = close.iloc[-1]
        for _ in range(days_forecast): p *= (1 + np.random.normal(mu, sigma))
        sims.append(p)
    t_mc = np.percentile(sims, 50)
    
    # 3. Fibonacci
    recent = close.iloc[-60:]
    h, l = recent.max(), recent.min()
    t_fib = h + (h - l) * 0.618
    
    # 4. Fundamental
    t_fund = f_data.get('Target_Mean')
    
    # 綜合平均 (僅技術面)
    tech_avg = (t_atr + t_mc + t_fib) / 3
    
    return {"ATR": t_atr, "MC": t_mc, "Fib": t_fib, "Fund": t_fund, "Avg": tech_avg}

# D. 全模組回測
def run_backtest_composite(close, high, low, days_ago=22):
    if len(close) < 300: return None
    idx_past = len(close) - days_ago - 1
    p_now = close.iloc[-1]
    
    # 切片數據
    c_slice = close.iloc[:idx_past+1]
    h_slice = high.iloc[:idx_past+1]
    l_slice = low.iloc[:idx_past+1]
    
    # 重跑模型 (當時視角)
    # ATR
    tr = pd.concat([h_slice-l_slice], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    y = c_slice.iloc[-126:].values.reshape(-1, 1)
    model = LinearRegression().fit(np.arange(len(y)).reshape(-1, 1), y)
    pred_trend = model.predict([[len(y) + days_ago]])[0].item()
    past_atr = pred_trend + (atr * np.sqrt(days_ago))
    
    # Fib
    recent = c_slice.iloc[-60:]
    past_fib = recent.max() + (recent.max() - recent.min()) * 0.618
    
    # MC (簡化)
    past_mc = c_slice.iloc[-1] * (1 + c_slice.pct_change().mean() * days_ago)
    
    past_avg = (past_atr + past_fib + past_mc) / 3
    err = (past_avg - p_now) / p_now
    
    return {"Past_Avg": past_avg, "Error": err, "Price_Now": p_now}

def calc_kelly(trend_status, win_rate=0.55):
    if "Bull" in trend_status: win_rate += 0.1
    if "Bear" in trend_status: win_rate -= 0.15
    f_star = (win_rate * 3 - 1) / 2 # Odds約為2
    return max(0, f_star * 0.5)

def analyze_trend_matrix(series):
    if len(series) < 126: return None
    y = series.iloc[-126:].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    p_2w = model.predict([[len(y)+10]])[0].item()
    p_1m = model.predict([[len(y)+22]])[0].item()
    p_3m = model.predict([[len(y)+66]])[0].item()
    
    p_now = series.iloc[-1]
    sma200 = series.rolling(200).mean().iloc[-1]
    
    status = "🛡️ 區間"
    if p_now > sma200: status = "🔥 牛市"
    elif p_now < sma200 * 0.9: status = "🛑 熊市"
    else: status = "⚠️ 弱勢"
    return {"p_2w": p_2w, "p_1m": p_1m, "p_3m": p_3m, "status": status}

def calc_obv(close, volume):
    if volume is None: return None
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

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
    st.title("Alpha 7.0: 全域資金流戰略 (Omni-Flow)")
    st.caption("v7.0 | RRG 資金流 | MVRV 估值 | 宏觀利率 | 四角回測")
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
        st.metric("總資產", f"${total_value:,.0f}")
        if st.button("🚀 啟動全域掃描", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False): return

    with st.spinner("🦅 Alpha 7.0 正在連線華爾街資料庫..."):
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_macro = fetch_fred_macro(fred_key)
        adv_data = {t: get_advanced_metrics(t) for t in tickers_list}

    if df_close.empty: st.error("No Data"); return

    # --- PART 1: 宏觀與資金流 (Macro & RRG) ---
    st.subheader("1. 宏觀與資金流向 (Macro & Fund Flow)")
    
    # 宏觀指標
    vix = df_close['^VIX'].iloc[-1]
    tnx = df_close['^TNX'].iloc[-1]
    irx = df_close['^IRX'].iloc[-1] # 13週短債，作為 Fed 利率預期代理
    liq_val = df_macro['Net_Liquidity'].iloc[-1] if df_macro is not None else 0
    
    # 判斷 Fed 方向
    fed_trend = "維持高利"
    if irx < 4.5: fed_trend = "📉 降息預期 (Dovish)"
    elif irx > 5.0: fed_trend = "📈 升息壓力 (Hawkish)"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💧 美元淨流動性", f"${liq_val:.2f}T" if df_macro is not None else "N/A")
    c2.metric("🌪️ VIX 恐慌指數", f"{vix:.2f}", delta="避險成本", delta_color="inverse")
    c3.metric("⚖️ 10年殖利率", f"{tnx:.2f}%", "定價錨")
    c4.metric("🏦 Fed 利率方向", fed_trend, f"短債: {irx:.2f}%")

    # RRG 圖表
    rrg_df = calc_rrg(df_close, tickers_list)
    if not rrg_df.empty:
        fig_rrg = px.scatter(rrg_df, x='RS_Ratio', y='RS_Momentum', color='Quadrant', text='Ticker',
                             title="RRG 資金流向雷達 (vs SPY)",
                             color_discrete_map={'🟢 領先 (Leading)': '#00FF7F', '🟡 轉弱 (Weakening)': '#FFFF00',
                                                 '🔴 落後 (Lagging)': '#FF4B4B', '🔵 改善 (Improving)': '#00BFFF'})
        fig_rrg.add_vline(x=100, line_dash="dash", line_color="gray")
        fig_rrg.add_hline(y=100, line_dash="dash", line_color="gray")
        fig_rrg.update_layout(xaxis_title="RS-Ratio (趨勢強度)", yaxis_title="RS-Momentum (動能速度)", height=500)
        st.plotly_chart(fig_rrg, use_container_width=True)
    
    st.markdown("---")

    # --- PART 2: 個股全域分析 ---
    st.subheader("2. 個股全域分析 (Deep Dive)")
    
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        
        trend = analyze_trend_matrix(df_close[ticker])
        info = adv_data.get(ticker, {})
        targets = calc_targets_composite(df_close[ticker], df_high[ticker], df_low[ticker], info, days_forecast=22)
        kelly = calc_kelly(trend['status'])
        bt = run_backtest_composite(df_close[ticker], df_high[ticker], df_low[ticker], days_ago=22)
        obv = calc_obv(df_close[ticker], df_vol[ticker])
        mvrv_series = calc_mvrv_z(df_close[ticker])
        mvrv_now = mvrv_series.iloc[-1] if mvrv_series is not None else 0
        
        # 標題
        t_avg_s = f"${targets['Avg']:.2f}" if targets and targets['Avg'] else "-"
        
        with st.expander(f"🦅 {ticker} | {trend['status']} | 綜合目標: {t_avg_s}", expanded=True):
            k1, k2, k3 = st.columns([2, 1, 1])
            
            with k1: # 圖表 (價格+OBV)
                st.markdown("#### 📉 價格與資金流 (Price & OBV)")
                fig = go.Figure()
                dates = df_close.index[-126:]
                fig.add_trace(go.Scatter(x=dates, y=df_close[ticker].iloc[-126:], name='Price', line=dict(color='#00FF7F', width=2)))
                fig.add_trace(go.Scatter(x=dates, y=df_close[ticker].rolling(200).mean().iloc[-126:], name='SMA200', line=dict(color='gray', dash='dash')))
                if obv is not None:
                    fig.add_trace(go.Scatter(x=dates, y=obv.iloc[-126:], name='OBV', line=dict(color='#FFD700', width=1), yaxis='y2'))
                fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), yaxis2=dict(overlaying='y', side='right', showgrid=False, title='OBV'))
                st.plotly_chart(fig, use_container_width=True)

            with k2: # 預測與回測
                st.markdown("#### 🎯 四角定位 (1M)")
                if targets:
                    st.write(f"**1. 物理 (ATR):** ${targets['ATR']:.2f}")
                    st.write(f"**2. 統計 (MC):** ${targets['MC']:.2f}")
                    st.write(f"**3. 心理 (Fib):** ${targets['Fib']:.2f}")
                    st.write(f"**4. 價值 (DCF):** ${targets['Fund']}" if targets['Fund'] else "N/A")
                
                st.divider()
                st.markdown("#### 🧪 平均模型回測")
                if bt:
                    err = bt['Error']
                    c_err = "green" if abs(err) < 0.05 else "red"
                    st.markdown(f"1月前預測誤差: <span style='color:{c_err}'>{err:.1%}</span>", unsafe_allow_html=True)
                    st.caption(f"當時預測 ${bt['Past_Avg']:.2f} vs 今日 ${bt['Price_Now']:.2f}")

            with k3: # 戰略指標 (MVRV, Rule40, Kelly)
                st.markdown("#### 💎 戰略指標")
                # MVRV Z-Score Gauge
                z_col = "red" if mvrv_now > 2 else ("green" if mvrv_now < 0 else "orange")
                st.metric("MVRV Z-Score", f"{mvrv_now:.2f}", delta="過熱" if mvrv_now>2 else ("超賣" if mvrv_now<0 else "正常"), delta_color="inverse")
                
                # Rule of 40
                r40 = info.get('Rule_40')
                st.metric("Rule of 40", f"{r40:.1f}" if r40 else "-", delta="優質" if r40 and r40>40 else "普通")
                
                # 機構持股
                inst = info.get('Inst_Held')
                st.metric("機構持股比", f"{inst*100:.0f}%" if inst else "-")
                
                st.divider()
                st.metric("Kelly 建議倉位", f"{kelly*100:.1f}%")

    st.markdown("---")
    
    # --- PART 3: 質性說明書 ---
    st.header("3. 系統運作原理與質性說明")
    with st.container():
        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
        
        st.markdown("### 🌊 RRG 資金流向 (Relative Rotation Graph)")
        st.info("透過比較每一檔資產相對於 **SPY (大盤)** 的強度與動能，將資金流向可視化。\n* **🟢 領先 (Leading):** 趨勢強、動能強 (資金流入)。\n* **🔴 落後 (Lagging):** 趨勢弱、動能弱 (資金流出)。")
        

        st.divider()
        st.markdown("### 📉 MVRV Z-Score (估值偏離)")
        st.info("計算價格與 200日均線 的標準差距離。這是一個均值回歸指標。\n* **Z > 2.0:** 價格嚴重偏離，風險極高 (紅色)。\n* **Z < 0.0:** 價格低於長期均線，潛在低估 (綠色)。")
        
        st.divider()
        st.markdown("### 🎯 四角定位與回測")
        st.markdown("""
        * **物理 (ATR Trend):** 考慮趨勢斜率與波動率的極限價格。
        * **統計 (Monte Carlo):** 1000次隨機漫步的中位數。
        * **心理 (Fibonacci):** 1.618 黃金擴展位。
        * **價值 (Fundamental):** 華爾街 DCF/PE 共識。
        * **回測 (Backtest):** 系統自動回溯至 22 天前，重跑模型並計算當時預測值與今日現價的誤差。
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()