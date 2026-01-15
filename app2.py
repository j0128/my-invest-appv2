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
st.set_page_config(page_title="Alpha 5.1: 戰略地平線 Pro", layout="wide", page_icon="🦅")

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
    benchmarks = ['SPY', 'QQQ', '^VIX', '^TNX', '^IRX', 'HYG'] 
    all_tickers = list(set(tickers + benchmarks))
    
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    progress_bar = st.progress(0, text="🦅 Alpha 5.1 正在建立正負六個月戰略模型...")
    
    for i, t in enumerate(all_tickers):
        try:
            progress_bar.progress((i + 1) / len(all_tickers), text=f"下載: {t} ...")
            # 抓取 1.5 年數據，確保能運算過去 1 年的波動率與回測
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
            pd.DataFrame(data['High']).ffill(), 
            pd.DataFrame(data['Low']).ffill(),
            pd.DataFrame(data['Volume']).ffill())

@st.cache_data(ttl=3600*12)
def fetch_fred_macro(api_key):
    if not api_key: return None
    try:
        fred = Fred(api_key=api_key)
        walcl = fred.get_series('WALCL', observation_start='2024-01-01')
        tga = fred.get_series('WTREGEN', observation_start='2024-01-01')
        rrp = fred.get_series('RRPONTSYD', observation_start='2024-01-01')
        rate = fred.get_series('FEDFUNDS', observation_start='2024-01-01')
        
        df = pd.DataFrame({'WALCL': walcl, 'TGA': tga, 'RRP': rrp, 'RATE': rate}).ffill().dropna()
        df['Net_Liquidity'] = (df['WALCL'] - df['TGA'] - df['RRP']) / 1000 # 單位: 兆
        return df
    except: return None

@st.cache_data(ttl=3600*24)
def get_fundamental_anchor(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'Target_Mean': info.get('targetMeanPrice'), # 華爾街 DCF/PE 共識
            'Forward_PE': info.get('forwardPE'),
            'PEG': info.get('pegRatio'),
            'High_52w': info.get('fiftyTwoWeekHigh'),
            'Low_52w': info.get('fiftyTwoWeekLow')
        }
    except: return {}

# --- 2. 核心運算模型 ---

def calc_kelly(trend_status, win_rate=0.55, odds=2.0):
    if "Bull" in trend_status: win_rate += 0.1
    if "Bear" in trend_status: win_rate -= 0.15
    f_star = (win_rate * (odds + 1) - 1) / odds
    return max(0, f_star * 0.5) # 半凱利

def calc_targets_v2(close, high, low, f_data, days_forecast=22):
    """
    四種模型計算目標價
    """
    if len(close) < 252: return None
    p_now = close.iloc[-1]
    
    # 1. ATR (物理極限) - 基於 days_forecast (預設1個月=22天)
    # 假設未來波動率不變，計算合理極限
    tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    # 公式: 現價 + (ATR * sqrt(天數) * 係數)
    t_atr = p_now + (atr * np.sqrt(days_forecast) * 1.2) 
    
    # 2. Monte Carlo (統計中樞 P50)
    returns = close.iloc[-252:].pct_change().dropna() # 過去一年波動
    mu, sigma = returns.mean(), returns.std()
    sims = []
    for _ in range(1000): # 1000次模擬
        p = p_now
        # 模擬未來 days_forecast 天
        for _ in range(days_forecast):
            p *= (1 + np.random.normal(mu, sigma))
        sims.append(p)
    t_mc = np.percentile(sims, 50) # P50 中位數
    
    # 3. Fibonacci (群眾心理)
    # 抓過去一季 (60天) 高低點
    recent = close.iloc[-60:]
    h, l = recent.max(), recent.min()
    t_fib = h + (h - l) * 0.618 # 1.618 擴展
    
    # 4. Fundamental (價值) - DCF/Forward PE
    t_fund = f_data.get('Target_Mean') # 華爾街共識目標 (通常是12個月)
    
    return t_atr, t_mc, t_fib, t_fund

def run_backtest(close, high, low, days_ago=22):
    """回測實驗室: 驗證 N 天前的模型預測"""
    if len(close) < 300: return None
    
    idx_past = len(close) - days_ago - 1
    p_past = close.iloc[idx_past]
    p_now = close.iloc[-1]
    
    # 切片數據
    c_slice = close.iloc[:idx_past+1]
    h_slice = high.iloc[:idx_past+1]
    l_slice = low.iloc[:idx_past+1]
    
    # 1. 回測 ATR
    tr = pd.concat([h_slice-l_slice], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    pred_atr = p_past + (atr * np.sqrt(days_ago) * 1.2)
    err_atr = (pred_atr - p_now) / p_now
    
    # 2. 回測 MC (簡化版)
    # 由於無法重跑 1000 次模擬的隨機性，這裡比較「當時的預期波動範圍」是否涵蓋今日價格
    
    return {"ATR_Error": err_atr, "Price_Past": p_past, "Price_Now": p_now}

def analyze_trend_matrix(series):
    """
    計算 2週, 1月, 3月 的線性回歸預測
    """
    if len(series) < 126: return None
    
    # 使用過去半年 (126天) 的數據建立趨勢線
    y = series.iloc[-126:].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    # 預測未來
    p_2w = model.predict([[len(y)+10]])[0].item() # 2週
    p_1m = model.predict([[len(y)+22]])[0].item() # 1月
    p_3m = model.predict([[len(y)+66]])[0].item() # 3月
    
    p_now = series.iloc[-1]
    sma200 = series.rolling(200).mean().iloc[-1]
    
    status = "🛡️ 區間震盪"
    if p_now > sma200: status = "🔥 多頭 (Bull)"
    elif p_now < sma200 * 0.9: status = "🛑 空頭 (Bear)"
    else: status = "⚠️ 弱勢整理"
        
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
    st.title("Alpha 5.1: 戰略地平線 Pro")
    st.caption("v5.1 | ±6個月趨勢 | 資金流圖表 | 四角定位回測 | 質性說明書")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ 參數設定")
        fred_key = st.secrets.get("FRED_API_KEY", st.text_input("FRED API Key", type="password"))
        
        st.header("💼 資產配置")
        default_input = """BTC-USD, 10000
AMD, 10000
NVDA, 10000
PLTR, 5000"""
        user_input = st.text_area("持倉清單", default_input, height=200)
        portfolio_dict = parse_input(user_input)
        tickers_list = list(portfolio_dict.keys())
        total_value = sum(portfolio_dict.values())
        st.metric("總資產估值", f"${total_value:,.0f}")
        
        if st.button("🚀 啟動戰略掃描", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False): return

    # --- 數據下載 ---
    with st.spinner("🦅 正在執行多維度戰略運算..."):
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_macro = fetch_fred_macro(fred_key)
        fund_data = {t: get_fundamental_anchor(t) for t in tickers_list}

    if df_close.empty: st.error("數據獲取失敗"); return

    # --- PART 1: 宏觀戰略儀表 (Macro & Liquidity) ---
    st.subheader("1. 宏觀戰略儀表 (Macro 4D)")
    
    # 數據準備
    vix = df_close['^VIX'].iloc[-1]
    tnx = df_close['^TNX'].iloc[-1]
    liq_val = df_macro['Net_Liquidity'].iloc[-1] if df_macro is not None else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💧 全球流動性", f"${liq_val:.2f}T" if df_macro is not None else "N/A", "Fed 燃料")
    c2.metric("🌪️ VIX 恐慌指數", f"{vix:.2f}", delta="避險成本", delta_color="inverse")
    c3.metric("⚖️ 10年殖利率", f"{tnx:.2f}%", "無風險利率")
    c4.metric("🏦 Fed 利率方向", "降息預期" if tnx < 4.5 else "高利率維持", "貨幣政策")

    # [新增] 流動性圖表
    if df_macro is not None:
        fig_liq = px.line(df_macro, y='Net_Liquidity', title='聯準會淨流動性趨勢 (Net Liquidity)', color_discrete_sequence=['#00BFFF'])
        fig_liq.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_liq, use_container_width=True)
    
    st.markdown("---")

    # --- PART 2: 個股戰略雷達 ---
    st.subheader("2. 個股戰略雷達 (Strategic Radar ±6M)")
    
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        
        # 運算
        trend = analyze_trend_matrix(df_close[ticker])
        f_info = fund_data.get(ticker, {})
        # 計算 1個月 (22天) 的目標價作為標準
        t_atr, t_mc, t_fib, t_fund = calc_targets_v2(df_close[ticker], df_high[ticker], df_low[ticker], f_info, days_forecast=22)
        kelly = calc_kelly(trend['status'])
        bt = run_backtest(df_close[ticker], df_high[ticker], df_low[ticker], days_ago=22)
        obv = calc_obv(df_close[ticker], df_vol[ticker])
        
        with st.expander(f"🦅 {ticker} | {trend['status']} | Kelly: {kelly}", expanded=True):
            k1, k2, k3 = st.columns([2, 1, 1])
            
            with k1: # 圖表 (價格 + OBV)
                st.markdown("#### 📉 價格與資金流 (Price & Fund Flow)")
                fig = go.Figure()
                # 主圖: 價格
                dates = df_close.index[-126:] # 過去半年
                fig.add_trace(go.Scatter(x=dates, y=df_close[ticker].iloc[-126:], name='Price', line=dict(color='#00FF7F', width=2)))
                fig.add_trace(go.Scatter(x=dates, y=df_close[ticker].rolling(200).mean().iloc[-126:], name='SMA200', line=dict(color='gray', dash='dash')))
                # 副圖: OBV
                if obv is not None:
                    fig.add_trace(go.Scatter(x=dates, y=obv.iloc[-126:], name='OBV (資金)', line=dict(color='#FFD700', width=1), yaxis='y2'))
                
                fig.update_layout(height=350, margin=dict(l=0,r=0,t=30,b=0),
                                  yaxis2=dict(overlaying='y', side='right', showgrid=False, title='OBV'))
                st.plotly_chart(fig, use_container_width=True)

            with k2: # 目標價矩陣
                st.markdown("#### 🎯 四角定位 (1個月預測)")
                st.write(f"**1. 物理 (ATR):** ${t_atr:.2f}" if t_atr else "-")
                st.write(f"**2. 統計 (MC P50):** ${t_mc:.2f}" if t_mc else "-")
                st.write(f"**3. 心理 (Fib 1.618):** ${t_fib:.2f}" if t_fib else "-")
                st.write(f"**4. 價值 (DCF/PE):** ${t_fund}" if t_fund else "N/A")
                
                st.divider()
                st.markdown("#### 🧪 回測驗證")
                if bt:
                    err = bt['ATR_Error']
                    c_err = "green" if abs(err) < 0.05 else "red"
                    st.markdown(f"ATR 模型誤差 (1M): <span style='color:{c_err}'>{err:.1%}</span>", unsafe_allow_html=True)
                    st.caption(f"1月前預測 vs 今日現價")

            with k3: # 未來推演
                st.markdown("#### 🔮 未來趨勢推演")
                st.metric("2週方向", f"${trend['p_2w']:.2f}")
                st.metric("1月方向", f"${trend['p_1m']:.2f}")
                st.metric("3月方向", f"${trend['p_3m']:.2f}")
                
                st.divider()
                st.markdown("#### 💎 估值")
                pe = f_info.get('Forward_PE')
                st.metric("Forward P/E", f"{pe:.1f}" if pe else "N/A")

    st.markdown("---")
    
    # --- PART 3: 質性說明書 (Qualitative Explanation) ---
    st.header("3. 系統運作原理與質性說明 (System Logic)")
    
    with st.container():
        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
        st.markdown("### 📊 各項數據的意義與運算邏輯")
        
        st.markdown("#### 1. 預測模型 (Prediction Models)")
        st.info("""
        * **🎯 保守目標 (ATR 物理極限):** 利用「平均真實波幅 (ATR)」計算股價在物理慣性下，未來一個月內「正常能量釋放」所能到達的極限邊界。這通常是波段操作的止盈點。
        * **⚖️ 中樞目標 (蒙地卡羅 P50):** 電腦進行 1,000 次隨機漫步模擬 (Monte Carlo Simulation)，基於過去一年的波動率。取第 50 百分位數 (中位數)，代表統計學上「最可能發生」的落點。
        * **🚀 樂觀目標 (費波那契 1.618):** 抓取過去一季 (60天) 的高低點，計算 1.618 黃金分割擴展位。這是群眾情緒瘋狂時，最容易產生共識的阻力位。
        * **🏦 價值目標 (DCF/PE):** 採用華爾街分析師的平均目標價。這背後隱含了現金流折現 (DCF) 與遠期本益比 (Forward PE) 的專業估值。
        """)
        
        st.divider()
        
        st.markdown("#### 2. 趨勢與資金 (Trend & Flow)")
        st.info("""
        * **OBV 資金流 (黃線):** 「能量潮指標」。當股價盤整但 OBV 創新高，代表主力正在吸籌 (Smart Money In)。反之則為出貨。圖表中採用雙軸顯示，方便對比價量背離。
        * **線性推演 (2W/1M/3M):** 基於過去半年 (126個交易日) 的股價走勢，畫出一條最適合的線性回歸趨勢線，並向右延伸推算未來 2週、1個月、3個月 的理論價格。
        * **Kelly 公式:** 根據趨勢多空動態調整勝率，計算出「數學上最佳」的持倉比例，以最大化長期幾何成長率並避免破產風險。
        """)
        
        st.markdown("#### 3. 宏觀四維度 (Macro 4D)")
        st.info("""
        * **💧 淨流動性 (Net Liquidity):** Fed 資產負債表 - TGA - 逆回購。這是美股的「真實燃料」。水位上升有利風險資產。
        * **⚖️ 10年殖利率 (TNX):** 全球資產定價的錨。殖利率過高會壓抑科技股估值 (P/E)。
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()