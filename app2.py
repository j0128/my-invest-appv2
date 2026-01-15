import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 0. 全局設定 ---
st.set_page_config(page_title="Alpha 8.0: 機器學習戰略", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #444; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
    .explanation-box {background-color: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 5px solid #00BFFF;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 ---
@st.cache_data(ttl=1800)
def fetch_market_data(tickers):
    # 加入宏觀因子: 銅(HG=F), 黃金(GC=F), 油(CL=F), 美元(DX-Y.NYB)
    benchmarks = ['SPY', 'QQQ', '^VIX', '^TNX', '^IRX', 'HYG', 'HG=F', 'GC=F', 'CL=F', 'DX-Y.NYB'] 
    all_tickers = list(set(tickers + benchmarks))
    
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    progress_bar = st.progress(0, text="🦅 Alpha 8.0 正在訓練 AI 模型...")
    
    for i, t in enumerate(all_tickers):
        try:
            progress_bar.progress((i + 1) / len(all_tickers), text=f"下載與特徵工程: {t} ...")
            # 抓取 3 年數據以供機器學習訓練
            df = yf.Ticker(t).history(period="3y", auto_adjust=True)
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
        walcl = fred.get_series('WALCL', observation_start='2023-01-01')
        tga = fred.get_series('WTREGEN', observation_start='2023-01-01')
        rrp = fred.get_series('RRPONTSYD', observation_start='2023-01-01')
        df = pd.DataFrame({'WALCL': walcl, 'TGA': tga, 'RRP': rrp}).ffill().dropna()
        df['Net_Liquidity'] = (df['WALCL'] - df['TGA'] - df['RRP']) / 1000 
        return df
    except: return None

@st.cache_data(ttl=3600*24)
def get_fundamental_anchor(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'Target_Mean': info.get('targetMeanPrice'), 
            'Forward_PE': info.get('forwardPE'),
            'Trailing_PE': info.get('trailingPE')
        }
    except: return {}

# --- 2. 機器學習引擎 (AI Engine) ---

def train_ai_model(target_ticker, df_close, df_vol, days_forecast=22):
    """
    訓練隨機森林 (Random Forest) 預測 1 個月後的價格
    特徵: RSI, 波動率, 均線乖離, 宏觀因子(VIX, 殖利率, 銅金比)
    """
    try:
        # 1. 準備特徵 (Features)
        df = pd.DataFrame(index=df_close.index)
        df['Close'] = df_close[target_ticker]
        
        # 技術指標
        df['RSI'] = 100 - (100 / (1 + df['Close'].diff().apply(lambda x: x if x>0 else 0).rolling(14).mean() / df['Close'].diff().apply(lambda x: -x if x<0 else 0).rolling(14).mean()))
        df['SMA_50'] = df['Close'] / df['Close'].rolling(50).mean() - 1 # 乖離率
        df['Vol_20'] = df['Close'].pct_change().rolling(20).std()
        
        # 宏觀因子 (如果有的話)
        if '^VIX' in df_close.columns: df['VIX'] = df_close['^VIX']
        if '^TNX' in df_close.columns: df['TNX'] = df_close['^TNX']
        if 'HG=F' in df_close.columns and 'GC=F' in df_close.columns:
            df['Copper_Gold'] = df_close['HG=F'] / df_close['GC=F']
            
        # 2. 準備標籤 (Target): 未來 N 天的收益率
        df['Target'] = df['Close'].shift(-days_forecast) # 未來價格
        
        # 清洗數據
        df = df.dropna()
        if len(df) < 100: return None # 數據太少不訓練
        
        # 3. 訓練模型
        X = df.drop(columns=['Target', 'Close']) # 使用所有特徵
        y = df['Target']
        
        # 分割訓練集與測試集 (不使用未來數據訓練)
        split = int(len(df) * 0.9)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # 4. 預測最新一筆
        latest_features = X.iloc[[-1]]
        pred_price = model.predict(latest_features)[0]
        
        return pred_price
    except: return None

# --- 3. 核心運算 (綜合模型 v2) ---

def calc_kelly(trend_status, win_rate=0.55, odds=2.0):
    if "Bull" in trend_status: win_rate += 0.1
    if "Bear" in trend_status: win_rate -= 0.15
    f_star = (win_rate * (odds + 1) - 1) / odds
    return max(0, f_star * 0.5)

def calc_trend_projection(series, days_future):
    y = series.values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    return model.predict([[len(y) + days_future]])[0].item()

def calc_targets_composite_v2(ticker, close, high, low, vol, f_data, days_forecast=22):
    if len(close) < 252: return None
    
    # 1. ATR (物理 - 趨勢調整)
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    price_projected = calc_trend_projection(close.iloc[-126:], days_forecast) 
    t_atr = price_projected + (atr * np.sqrt(days_forecast))
    
    # 2. Monte Carlo (機率 - P50)
    returns = close.iloc[-252:].pct_change().dropna()
    mu, sigma = returns.mean(), returns.std()
    sims = []
    for _ in range(1000):
        p = close.iloc[-1]
        for _ in range(days_forecast): p *= (1 + np.random.normal(mu, sigma))
        sims.append(p)
    t_mc = np.percentile(sims, 50)
    
    # 3. Fibonacci (心理)
    recent = close.iloc[-60:]
    h, l = recent.max(), recent.min()
    t_fib = h + (h - l) * 0.618 
    
    # 4. Fundamental (價值)
    t_fund = f_data.get('Target_Mean')
    
    # 5. AI Prediction (機器學習) - NEW!
    t_ai = train_ai_model(ticker, close.to_frame(ticker).join(close.to_frame('^VIX'), rsuffix='_vix'), vol, days_forecast)
    
    # 綜合平均 (包含 AI)
    targets_list = [t for t in [t_atr, t_mc, t_fib, t_ai] if t is not None]
    t_avg = sum(targets_list) / len(targets_list) if targets_list else None
    
    return {
        "ATR": t_atr, "MC": t_mc, "Fib": t_fib, "Fund": t_fund, "AI": t_ai, "Avg": t_avg
    }

def run_backtest_composite(close, high, low, days_ago=22):
    if len(close) < 300: return None
    idx_past = len(close) - days_ago - 1
    p_now = close.iloc[-1]
    
    # 簡化回測: 比較 ATR 與 趨勢線 的準確度作為代表
    c_slice = close.iloc[:idx_past+1]
    y = c_slice.iloc[-126:].values.reshape(-1, 1)
    model = LinearRegression().fit(np.arange(len(y)).reshape(-1, 1), y)
    pred = model.predict([[len(y) + days_ago]])[0].item()
    err = (pred - p_now) / p_now
    
    return {"Past_Pred": pred, "Error": err, "Price_Now": p_now}

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
    st.title("Alpha 8.0: 機器學習戰略 (ML Enhanced)")
    st.caption("v8.0 | AI 隨機森林預測 | 宏觀因子 | 銅金比 | 綜合回測")
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
        if st.button("🚀 啟動 AI 運算", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False): return

    with st.spinner("🦅 正在訓練 AI 模型與下載宏觀數據..."):
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_macro = fetch_fred_macro(fred_key)
        fund_data = {t: get_fundamental_anchor(t) for t in tickers_list}

    if df_close.empty: st.error("No Data"); return

    # --- PART 1: 宏觀與經濟 (Macro & Economy) ---
    st.subheader("1. 宏觀經濟晴雨表 (Macro Dashboard)")
    
    # 宏觀指標
    vix = df_close['^VIX'].iloc[-1]
    tnx = df_close['^TNX'].iloc[-1]
    dxy = df_close['DX-Y.NYB'].iloc[-1] if 'DX-Y.NYB' in df_close else 0
    # 銅金比 (Copper/Gold) - 經濟領先指標
    cg_ratio = (df_close['HG=F'].iloc[-1] / df_close['GC=F'].iloc[-1]) * 1000 if 'HG=F' in df_close and 'GC=F' in df_close else 0
    
    liq_val = df_macro['Net_Liquidity'].iloc[-1] if df_macro is not None else 0
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💧 淨流動性", f"${liq_val:.2f}T")
    c2.metric("🌪️ VIX", f"{vix:.2f}", delta_color="inverse")
    c3.metric("⚖️ 10年殖利率", f"{tnx:.2f}%")
    c4.metric("🏭 銅金比 (經濟)", f"{cg_ratio:.2f}", "數值高=景氣好")
    c5.metric("💵 美元指數", f"{dxy:.2f}")

    if df_macro is not None:
        fig_liq = px.line(df_macro, y='Net_Liquidity', title='聯準會淨流動性趨勢', color_discrete_sequence=['#00BFFF'])
        fig_liq.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_liq, use_container_width=True)
    st.markdown("---")

    # --- PART 2: 個股 AI 戰略 ---
    st.subheader("2. 個股 AI 戰略 (AI Strategic Radar)")
    
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        
        trend = analyze_trend_matrix(df_close[ticker])
        f_info = fund_data.get(ticker, {})
        # 計算 1個月 的目標價 (五角定位: ATR, MC, Fib, Fund, AI)
        targets = calc_targets_composite_v2(ticker, df_close, df_high, df_low, df_vol, f_info, days_forecast=22)
        kelly = calc_kelly(trend['status'])
        bt = run_backtest_composite(df_close[ticker], df_high[ticker], df_low[ticker], days_ago=22)
        obv = calc_obv(df_close[ticker], df_vol[ticker])
        
        t_avg_s = f"${targets['Avg']:.2f}" if targets and targets['Avg'] else "-"
        
        with st.expander(f"🦅 {ticker} | {trend['status']} | 綜合目標: {t_avg_s}", expanded=True):
            k1, k2, k3 = st.columns([2, 1, 1])
            
            with k1: # 圖表
                st.markdown("#### 📉 雙軸圖 (Price & OBV)")
                fig = go.Figure()
                dates = df_close.index[-126:]
                fig.add_trace(go.Scatter(x=dates, y=df_close[ticker].iloc[-126:], name='Price', line=dict(color='#00FF7F', width=2)))
                fig.add_trace(go.Scatter(x=dates, y=df_close[ticker].rolling(200).mean().iloc[-126:], name='SMA200', line=dict(color='gray', dash='dash')))
                if obv is not None:
                    fig.add_trace(go.Scatter(x=dates, y=obv.iloc[-126:], name='OBV', line=dict(color='#FFD700', width=1), yaxis='y2'))
                fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0), yaxis2=dict(overlaying='y', side='right', showgrid=False, title='OBV'), legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)

            with k2: # AI 預測矩陣
                st.markdown("#### 🤖 五角定位 (1M)")
                if targets:
                    st.write(f"**1. 物理 (ATR):** ${targets['ATR']:.2f}")
                    st.write(f"**2. 統計 (MC):** ${targets['MC']:.2f}")
                    st.write(f"**3. 心理 (Fib):** ${targets['Fib']:.2f}")
                    st.write(f"**4. 智能 (AI):** ${targets['AI']:.2f}" if targets['AI'] else "N/A")
                    st.caption("AI 模型: Random Forest Regressor")
                    st.write(f"**5. 價值 (DCF):** ${targets['Fund']}" if targets['Fund'] else "N/A")
                
                st.divider()
                if bt:
                    err = bt['Error']
                    c_err = "green" if abs(err) < 0.05 else "red"
                    st.markdown(f"回測誤差: <span style='color:{c_err}'>{err:.1%}</span>", unsafe_allow_html=True)

            with k3: # 未來推演
                st.markdown("#### 🔮 趨勢推演")
                st.metric("2週方向", f"${trend['p_2w']:.2f}")
                st.metric("1月方向", f"${trend['p_1m']:.2f}")
                st.metric("3月方向", f"${trend['p_3m']:.2f}")
                
                st.divider()
                st.metric("Forward P/E", f"{f_info.get('Forward_PE')}")

    st.markdown("---")
    
    # --- PART 3: 說明書 ---
    st.header("3. 系統運作原理與質性說明")
    with st.container():
        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
        st.markdown("### 🤖 機器學習 (Random Forest)")
        st.info("系統現場訓練一個 **隨機森林模型**，學習該資產過去 3 年的價格行為、波動率、RSI 以及宏觀因子 (VIX, 殖利率) 之間的非線性關係，並預測 1 個月後的價格。這是比線性回歸更先進的預測方法。")
        
        st.divider()
        st.markdown("### 🏭 銅金比 (Copper/Gold Ratio)")
        st.info("銅代表工業需求 (實體經濟)，黃金代表避險需求 (恐慌)。\n* **銅金比上升:** 經濟復甦，有利股市 (Risk On)。\n* **銅金比下降:** 經濟衰退，資金轉向避險 (Risk Off)。")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()