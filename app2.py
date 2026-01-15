import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 0. 全局設定 (戰略黑金版) ---
st.set_page_config(page_title="Alpha 5.0: 戰略地平線", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #444; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
    .highlight-box {border-left: 5px solid #00BFFF; background-color: #1a1a1a; padding: 10px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 (宏觀+個股) ---
@st.cache_data(ttl=1800)
def fetch_market_data(tickers):
    # 增加宏觀指標: 10年債(^TNX), 恐慌(^VIX), 短債/Fed預期(^IRX)
    benchmarks = ['SPY', 'QQQ', '^VIX', '^TNX', '^IRX', 'HYG'] 
    all_tickers = list(set(tickers + benchmarks))
    
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    progress_bar = st.progress(0, text="🦅 Alpha 5.0 正在掃描正負六個月戰略區間...")
    
    for i, t in enumerate(all_tickers):
        try:
            progress_bar.progress((i + 1) / len(all_tickers), text=f"連線中: {t} ...")
            # 抓取 1 年數據，以確保能完整計算 6 個月的技術指標
            df = yf.Ticker(t).history(period="1y", auto_adjust=True)
            if df.empty: continue
            
            data['Close'][t] = df['Close']
            data['Open'][t] = df['Open']
            data['High'][t] = df['High']
            data['Low'][t] = df['Low']
            data['Volume'][t] = df['Volume']
        except: continue
            
    progress_bar.empty()
    return (pd.DataFrame(data['Close']).ffill(), 
            pd.DataFrame(data['High']).ffill(), 
            pd.DataFrame(data['Low']).ffill(),
            pd.DataFrame(data['Volume']).ffill())

@st.cache_data(ttl=3600*12)
def fetch_fred_macro(api_key):
    """抓取 Fed 真實流動性與利率"""
    if not api_key: return None
    try:
        fred = Fred(api_key=api_key)
        # WALCL: Fed資產, WTREGEN: TGA帳戶, RRPONTSYD: 逆回購
        walcl = fred.get_series('WALCL', observation_start='2024-01-01')
        tga = fred.get_series('WTREGEN', observation_start='2024-01-01')
        rrp = fred.get_series('RRPONTSYD', observation_start='2024-01-01')
        fed_funds = fred.get_series('FEDFUNDS', observation_start='2023-01-01') # 聯邦基金利率
        
        df = pd.DataFrame({'WALCL': walcl, 'TGA': tga, 'RRP': rrp, 'RATE': fed_funds}).ffill().dropna()
        df['Net_Liquidity'] = (df['WALCL'] - df['TGA'] - df['RRP']) / 1000 
        return df
    except: return None

@st.cache_data(ttl=3600*24)
def get_fundamental_anchor(ticker):
    """基本面錨點 (DCF/PE/Analyst)"""
    try:
        info = yf.Ticker(ticker).info
        return {
            'Target_Mean': info.get('targetMeanPrice'), # 華爾街共識 (隱含 DCF/PE)
            'Forward_PE': info.get('forwardPE'),
            'Trailing_PE': info.get('trailingPE'),
            'PEG': info.get('pegRatio'),
            'Recommendation': info.get('recommendationKey')
        }
    except: return {}

# --- 2. 核心運算模型 ---

def calc_kelly_criterion(trend_data, win_rate=0.55, odds=2.0):
    """
    凱利公式 (半凱利模式)
    f* = (p(b+1) - 1) / b
    """
    if not trend_data: return "0%"
    # 動態調整勝率
    if "Bull" in trend_data['status']: win_rate += 0.1
    if "Bear" in trend_data['status']: win_rate -= 0.15
    
    f_star = (win_rate * (odds + 1) - 1) / odds
    safe_kelly = max(0, f_star * 0.5) # 半凱利，安全第一
    return f"{safe_kelly*100:.1f}%"

def calc_quad_targets(close, high, low, f_data):
    """
    四角定位運算 (含時間維度)
    """
    if len(close) < 60: return None, None, None, None
    try:
        current_price = close.iloc[-1]
        
        # 1. ATR (物理極限) - 預測 1個月 (22天)
        tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        t_atr = current_price + (atr * np.sqrt(22) * 1.5) # 1.5倍月波動作為極限
        
        # 2. Monte Carlo (統計中樞) - 預測 1個月
        returns = close.pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        sims = []
        for _ in range(500):
            p = current_price
            for _ in range(22): p *= (1 + np.random.normal(mu, sigma))
            sims.append(p)
        t_mc = np.percentile(sims, 50)
        
        # 3. Fibonacci (群眾心理) - 過去 6 個月高點擴展
        lookback = 126 # 6個月 (約126交易日)
        recent = close.iloc[-lookback:]
        h, l = recent.max(), recent.min()
        t_fib = h + (h - l) * 0.618
        
        # 4. Fundamental (價值)
        t_fund = f_data.get('Target_Mean')
        
        return t_atr, t_mc, t_fib, t_fund
    except: return None, None, None, None

def analyze_trend_6m(series):
    """
    正負六個月趨勢判定 (Regime Filter)
    """
    if series is None or len(series) < 126: return None
    
    p_now = series.iloc[-1]
    sma200 = series.rolling(200).mean().iloc[-1]
    sma50 = series.rolling(50).mean().iloc[-1]
    
    # 趨勢斜率 (過去 6 個月)
    y = series.iloc[-126:].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    # 預測未來 (2週, 1月, 3月)
    p_2w = model.predict([[len(y)+10]])[0].item()
    p_1m = model.predict([[len(y)+22]])[0].item()
    p_3m = model.predict([[len(y)+66]])[0].item()
    
    # 修正後的狀態判定
    if p_now > sma200:
        if p_now > sma50: status = "🔥 強勢牛 (Bull)"
        else: status = "⚠️ 回調 (Correction)"
    else:
        # 如果跌破年線但在 88% 以上，視為假跌破/弱勢整理，而非熊市
        if p_now > sma200 * 0.88: status = "📉 弱勢整理 (Weak)"
        else: status = "🛑 熊市 (Bear)"
        
    return {"status": status, "p_now": p_now, "p_2w": p_2w, "p_1m": p_1m, "p_3m": p_3m, "sma200": sma200}

def backtest_lab(ticker, close, high, low):
    """
    回測實驗室：驗證 1 個月前的預測準不準
    """
    if len(close) < 250: return None
    
    # 回到 22 天前
    idx_past = len(close) - 22 - 1
    p_past = close.iloc[idx_past]
    p_now = close.iloc[-1]
    
    # 用當時數據算目標
    c_slice = close.iloc[:idx_past+1]
    h_slice = high.iloc[:idx_past+1]
    l_slice = low.iloc[:idx_past+1]
    
    # 簡化版計算
    tr = pd.concat([h_slice-l_slice], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    pred_atr = p_past + (atr * np.sqrt(22) * 1.5)
    
    # 誤差
    err = (pred_atr - p_now) / p_now
    return {"pred": pred_atr, "actual": p_now, "error": err}

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
    st.title("Alpha 5.0: 戰略地平線 (Strategic Horizon)")
    st.caption("v5.0 | ±6個月趨勢 | 宏觀四維度 | 四角定位 | Kelly公式")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ 戰情設定")
        fred_key = st.secrets.get("FRED_API_KEY", st.text_input("FRED API Key (宏觀數據用)", type="password"))
        
        st.header("💼 資產配置")
        default_input = """BTC-USD, 10000
AMD, 10000
NVDA, 10000
PLTR, 5000"""
        user_input = st.text_area("持倉清單", default_input, height=200)
        portfolio_dict = parse_input(user_input)
        tickers_list = list(portfolio_dict.keys())
        total_value = sum(portfolio_dict.values())
        st.metric("總資產估值 (Est.)", f"${total_value:,.0f}")
        
        if st.button("🚀 啟動戰略掃描", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False):
        st.info("👈 請輸入資產並啟動。系統將執行 ±6 個月的深度戰略推演。")
        return

    # --- 數據下載 ---
    with st.spinner("🦅 正在建立宏觀與微觀連線..."):
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_macro = fetch_fred_macro(fred_key)
        
        # 準備個股基本面
        fund_data = {t: get_fundamental_anchor(t) for t in tickers_list}

    if df_close.empty: st.error("數據獲取失敗"); return

    # --- PART 1: 宏觀戰略儀表 (Macro 4D) ---
    st.subheader("1. 宏觀戰略儀表 (The Macro 4D)")
    
    # 準備數據
    tnx = df_close['^TNX'].iloc[-1] if '^TNX' in df_close else 4.0
    vix = df_close['^VIX'].iloc[-1] if '^VIX' in df_close else 15.0
    irx = df_close['^IRX'].iloc[-1] if '^IRX' in df_close else 5.0 # 短債近似利率
    
    # 如果有 FRED 數據，覆蓋流動性與利率
    liq_val = "N/A"
    fed_rate = irx # 預設用短債
    if df_macro is not None and not df_macro.empty:
        liq_val = f"${df_macro['Net_Liquidity'].iloc[-1]:.2f}T"
        fed_rate = df_macro['RATE'].iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💧 全球流動性 (Fed)", liq_val, "總資金水位")
    c2.metric("🌪️ 市場恐慌 (VIX)", f"{vix:.2f}", "避險成本", delta_color="inverse")
    c3.metric("⚖️ 10年殖利率 (TNX)", f"{tnx:.2f}%", "資產定價錨")
    c4.metric("🏦 Fed 基準利率", f"{fed_rate:.2f}%", "資金成本")
    
    # 宏觀判讀
    macro_signal = "中性震盪"
    if tnx < 4.0 and vix < 20: macro_signal = "🟢 Risk On (適合進攻)"
    elif tnx > 4.5 or vix > 25: macro_signal = "🔴 Risk Off (防禦為上)"
    st.caption(f"當前宏觀訊號：{macro_signal}")
    st.markdown("---")

    # --- PART 2: 個股戰略雷達 ---
    st.subheader("2. 個股戰略雷達 (Strategic Radar ±6M)")
    
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        
        # 運算
        trend = analyze_trend_6m(df_close[ticker])
        f_data = fund_data.get(ticker, {})
        t_atr, t_mc, t_fib, t_fund = calc_quad_targets(df_close[ticker], df_high[ticker], df_low[ticker], f_data)
        kelly = calc_kelly_criterion(trend)
        bt = backtest_lab(ticker, df_close[ticker], df_high[ticker], df_low[ticker])
        
        # 顯示卡片
        with st.expander(f"🦅 {ticker} | {trend['status']} | Kelly: {kelly}", expanded=True):
            k1, k2, k3 = st.columns([2, 1, 1])
            
            with k1: # 價格與預測
                st.markdown("#### 🎯 四角定位 (Quad-Anchor)")
                c_a, c_b = st.columns(2)
                c_a.write(f"**1. 物理 (ATR):** ${t_atr:.2f}" if t_atr else "-")
                c_a.write(f"**2. 統計 (MC):** ${t_mc:.2f}" if t_mc else "-")
                c_b.write(f"**3. 心理 (Fib):** ${t_fib:.2f}" if t_fib else "-")
                c_b.write(f"**4. 價值 (Wall St.):** ${t_fund}" if t_fund else "N/A")
                
                # 繪製 ±6個月 圖表
                dates = df_close.index[-126:] # 過去6個月
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=df_close[ticker].iloc[-126:], name='Price', line=dict(color='#00FF7F', width=2)))
                fig.add_trace(go.Scatter(x=dates, y=df_close[ticker].rolling(200).mean().iloc[-126:], name='SMA200 (牛熊線)', line=dict(color='orange', dash='dash')))
                fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), title=f"{ticker} 過去 6 個月走勢")
                st.plotly_chart(fig, use_container_width=True)

            with k2: # 未來推演
                st.markdown("#### 🔮 未來 3 個月推演")
                st.metric("2週方向", f"${trend['p_2w']:.2f}")
                st.metric("1月方向", f"${trend['p_1m']:.2f}")
                st.metric("3月方向", f"${trend['p_3m']:.2f}")
                st.caption("基於線性回歸通道")

            with k3: # 估值與回測
                st.markdown("#### ⚖️ 估值與驗證")
                pe = f_data.get('Forward_PE')
                st.metric("Forward P/E", f"{pe:.1f}" if pe else "N/A")
                
                if bt:
                    st.markdown("#### 🧪 回測實驗室")
                    err = bt['error']
                    color = "green" if abs(err) < 0.05 else "red"
                    st.markdown(f"1月前預測誤差: <span style='color:{color}'>{err:.1%}</span>", unsafe_allow_html=True)
                    st.caption("模型: ATR極限法")

    st.markdown("---")
    
    # --- PART 3: 資產總表 ---
    st.subheader("3. 投資組合總表")
    table_data = []
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend_6m(df_close[ticker])
        f_data = fund_data.get(ticker, {})
        k_val = calc_kelly_criterion(trend)
        
        # 使用分析師目標價，若無則用 Monte Carlo
        tgt = f_data.get('Target_Mean')
        if not tgt: 
            _, t_mc, _, _ = calc_quad_targets(df_close[ticker], df_high[ticker], df_low[ticker], f_data)
            tgt = f"${t_mc:.2f} (MC)"
        else:
            tgt = f"${tgt} (Fund)"

        table_data.append({
            "代號": ticker,
            "現價": f"${trend['p_now']:.2f}",
            "趨勢狀態": trend['status'],
            "目標價 (6M)": tgt,
            "Kelly倉位": k_val,
            "Forward P/E": f_data.get('Forward_PE', '-')
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # --- PART 4: 公式白皮書 ---
    st.markdown("---")
    st.header("4. 量化模型公式手冊 (Quantitative Whitepaper)")
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("### 🎯 四角定位 (Quad-Anchor)")
            st.markdown("**1. 物理 (ATR):** $P + (ATR \\times \\sqrt{t} \\times 1.5)$")
            st.markdown("**2. 統計 (MC):** 隨機漫步模擬中位數")
            st.markdown("**3. 心理 (Fib):** $H + (H-L) \\times 0.618$")
            st.markdown("**4. 價值 (DCF):** 華爾街共識目標價")
        with c2:
            st.info("### 🎲 凱利公式 (Half-Kelly)")
            st.latex(r'''f^* = \frac{p(b+1)-1}{b} \times 0.5''')
            st.markdown("* **p:** 勝率 (動態調整)")
            st.markdown("* **b:** 賠率 (設為 2.0)")
        with c3:
            st.info("### 🔮 線性推演 (Linear Projection)")
            st.latex(r'''y = \alpha + \beta x''')
            st.markdown("基於過去 6 個月 ($N=126$) 的回歸斜率，推演未來 $t+10, t+22, t+66$ 的價格中樞。")

if __name__ == "__main__":
    main()