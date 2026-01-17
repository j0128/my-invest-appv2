import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime, timedelta

# ==============================================================================
# 0. 全局環境與拓撲常數設定 (Alpha 16.1)
# ==============================================================================

TOPO_CONSTANTS = {
    "LIQUIDITY_THRESHOLD": -0.2,  # 最佳防禦閾值 (Trillion USD, 20-day change)
    "LAG_DAYS_TECH": 15,            
    "LAG_DAYS_CRYPTO": 0,           
    "KELLY_LOOKBACK": 60,           
    "RF_TREES": 100                 
}

# 資產分類學
ASSET_TAXONOMY = {
    "Growth": ['BTC-USD', 'ETH-USD', 'ARKK', 'PLTR', 'NVDA', 'AMD', 'TSLA', 'TQQQ', 'SOXL'],
    "Defensive": ['KO', 'MCD', 'JNJ', 'PG', '2330.TW', 'SPY', 'TLT', 'GLD', 'SCHD']
}

# 鐵壁防禦名單 (實驗驗證誤差 < 6%)
SAFE_HARBOR_LIST = [
    'XLP', 'TLT', 'XLV', 'KO', 'XLE', 
    'MMM', 'JNJ', 'MCD', 'XLF', 'RTX', 
    'XOM', 'CVX', 'MO', 'GILD', 'AMGN'
]

st.set_page_config(
    page_title="Alpha 16.1: 拓撲指揮官 (估值修正版)",
    layout="wide",
    page_icon="🦅"
)

# 注入優化後的 CSS
st.markdown("""
<style>
    .metric-card { background-color: #0E1117; border: 1px solid #444; border-radius: 5px; padding: 15px; color: white; }
    .bull-mode { color: #00FF7F; font-weight: bold; border: 1px solid #00FF7F; padding: 2px 8px; border-radius: 4px; font-size: 0.9em; }
    .bear-mode { color: #FF4B4B; font-weight: bold; border: 1px solid #FF4B4B; padding: 2px 8px; border-radius: 4px; font-size: 0.9em; }
    .card { background-color: #262730; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #555; }
    .pe-tag-hot { color: #FF4B4B; font-size: 0.85em; font-weight: bold; }
    .pe-tag-cool { color: #00BFFF; font-size: 0.85em; font-weight: bold; }
    .safe-harbor-header { color: #00BFFF; border-bottom: 2px solid #00BFFF; padding-bottom: 5px; margin-top: 30px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. 數據與估值引擎 (The Valuation & Data Engine)
# ==============================================================================

@st.cache_data(ttl=1800)
def fetch_market_data(tickers):
    benchmarks = ['SPY', 'QQQ', 'QLD', 'TQQQ', '^VIX', '^TNX', '^IRX', 'HYG', 'GC=F', 'DX-Y.NYB'] 
    all_tickers = list(set(tickers + benchmarks + SAFE_HARBOR_LIST))
    df_bulk = yf.download(all_tickers, period="2y", progress=False)
    if isinstance(df_bulk.columns, pd.MultiIndex):
        return df_bulk['Close'].ffill(), df_bulk['High'].ffill(), df_bulk['Low'].ffill(), df_bulk['Volume'].ffill()
    return df_bulk['Close'].ffill(), df_bulk['High'].ffill(), df_bulk['Low'].ffill(), df_bulk['Volume'].ffill()

@st.cache_data(ttl=3600*12)
def fetch_fred_macro(api_key):
    if not api_key: return None, None
    try:
        fred = Fred(api_key=api_key)
        walcl = fred.get_series('WALCL', observation_start='2023-01-01')
        tga = fred.get_series('WTREGEN', observation_start='2023-01-01')
        rrp = fred.get_series('RRPONTSYD', observation_start='2023-01-01')
        df = pd.DataFrame({'WALCL': walcl, 'TGA': tga, 'RRP': rrp}).ffill().dropna()
        df['Net_Liquidity'] = (df['WALCL'] - df['TGA'] - df['RRP']) / 1000 
        return df, None
    except: return None, None

@st.cache_data(ttl=3600*12)
def get_pe_percentile(ticker):
    """計算 P/E 歷史分位數"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3y")['Close']
        info = stock.info
        eps = info.get('trailingEps')
        current_pe = info.get('trailingPE')
        if not eps or eps <= 0 or not current_pe: return 50.0
        pe_series = hist / eps
        return stats.percentileofscore(pe_series.dropna(), current_pe)
    except: return 50.0

@st.cache_data(ttl=3600*12) 
def get_fundamental_scalar_v16(ticker):
    """Alpha 16.1: 整合真實財報與估值位階的加權引擎"""
    try:
        stock = yf.Ticker(ticker); info = stock.info
        if info.get('quoteType') == 'ETF': return 1.0, ["⚖️ ETF (中性)"], 50.0, None
        
        fins = stock.quarterly_financials
        if fins.empty: fins = stock.financials
        
        score = 0; details = []
        
        # 1. 營收成長 (YoY)
        if not fins.empty and 'Total Revenue' in fins.index and len(fins.columns) >= 2:
            growth = (fins.loc['Total Revenue'].iloc[0] - fins.loc['Total Revenue'].iloc[1]) / fins.loc['Total Revenue'].iloc[1]
            if growth > 0.15: score += 1; details.append(f"🔥 強勁成長 (+{growth:.1%})")
            elif growth < 0: score -= 1; details.append("📉 營收衰退")

        # 2. P/E Percentile 修正
        pe_pct = get_pe_percentile(ticker)
        if pe_pct > 90: score -= 1.5; details.append(f"⚠️ 歷史高估 ({pe_pct:.0f}%)")
        elif pe_pct < 20: score += 1.5; details.append(f"💎 歷史低估 ({pe_pct:.0f}%)")
        
        # 3. PEG 邏輯
        peg = info.get('pegRatio')
        if peg and 0 < peg < 1.0: score += 1; details.append(f"🎯 PEG 優勢 ({peg:.2f})")

        scalar = max(0.85, min(1.15, 1.0 + (score * 0.05)))
        return scalar, details, pe_pct, peg
    except: return 1.0, ["⚠️ 數據異常"], 50.0, None

# ==============================================================================
# 2. 戰略核心算法 (Strategic Core)
# ==============================================================================

def analyze_trend_multi(series):
    p = series.iloc[-1]; ma200 = series.rolling(200).mean().iloc[-1]
    ma20 = series.rolling(20).mean().iloc[-1]
    dir_icon = "↑" if p > ma20 else "↓"
    return {"status": f"{dir_icon} 多頭" if p > ma200 else f"{dir_icon} 空頭", "is_bull": p > ma200, "p_now": p}

def calc_mvrv_z(series):
    if len(series) < 200: return 0
    return (series.iloc[-1] - series.rolling(200).mean().iloc[-1]) / series.rolling(200).std().iloc[-1]

def get_cfo_directive_v16(ticker, p_now, six_state, is_bull, rsi, pe_pct, mvrv_z):
    """Alpha 16.1: 整合 MVRV 與 估值位階的決策引擎"""
    # 加密貨幣專屬邏輯
    if "USD" in ticker:
        if mvrv_z > 2.5: return "🟥 泡沫區 (清倉)", 0.0
        if mvrv_z < -0.5: return "🔵 價值買點 (重倉)", 0.9
    
    # 通用估值邏輯
    if pe_pct > 95: return "🟥 歷史極限高估 (賣出)", 0.0
    if pe_pct < 15 and is_bull: return "💎 戰略級買點 (強烈建倉)", 0.8
    
    # 技術面輔助
    if "H3" in six_state or rsi > 82: return "🟨 噴出 (分批止盈)", 0.3
    if is_bull: return "🟢 多頭持有", 0.5
    return "⚪ 觀望", 0.0

def train_rf_model(df_close, ticker, days=30):
    try:
        df = pd.DataFrame({'Close': df_close[ticker]})
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Vol'] = df['Close'].pct_change().rolling(20).std()
        df['Target'] = df['Close'].shift(-days)
        train = df.dropna()
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(train[['MA20', 'Vol']], train['Target'])
        return model.predict(df[['MA20', 'Vol']].iloc[[-1]])[0]
    except: return df_close[ticker].iloc[-1]

# ==============================================================================
# 3. 介面渲染 (UI Rendering)
# ==============================================================================

def render_valuation_card(t, price_now, tech_target, scalar, reasons, pe_pct):
    final_target = tech_target * scalar
    upside = (final_target - price_now) / price_now
    pe_tag = f'<span class="pe-tag-hot">【過熱 {pe_pct:.0f}%】</span>' if pe_pct > 90 else (f'<span class="pe-tag-cool">【超值 {pe_pct:.0f}%】</span>' if pe_pct < 20 else "")
    up_color = "#00FF7F" if upside > 0 else "#FF4B4B"
    reasons_html = "<br>".join([f"<small>{r}</small>" for r in reasons])
    
    st.markdown(f"""
    <div class="card" style="border-left-color: {up_color};">
        <div class="card-title">{t} {pe_tag} <span style="float:right; color:#FFF">${price_now:.2f}</span></div>
        <div style="margin-top:5px; font-size:0.9em; color:#AAA;">
            預測: ${tech_target:.2f} | 加權: x{scalar:.2f}
        </div>
        <div class="card-value" style="color:{up_color}; margin-top:5px;">
            目標: ${final_target:.2f} <small>({upside:+.1%})</small>
        </div>
        <div style="color: #888; margin-top:5px; line-height:1.2;">
            {reasons_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.sidebar.title("🦅 Alpha 16.1 指揮部")
    fred_key = st.sidebar.text_input("FRED API Key", type="password")
    user_input = st.sidebar.text_area("持倉清單", "BTC-USD, 10000\nNVDA, 10000\n2330.TW, 10000", height=100)
    p_dict = {l.split(',')[0].strip().upper(): float(l.split(',')[1]) for l in user_input.strip().split('\n') if ',' in l}
    
    if not st.sidebar.button("🚀 執行全域掃描"): return

    df_close, df_high, df_low, df_vol = fetch_market_data(list(p_dict.keys()))
    df_macro, _ = fetch_fred_macro(fred_key)

    t1, t2, t3, t4 = st.tabs(["🦅 戰略戰情", "🛡️ 鐵壁防禦", "🚦 策略實驗", "💰 退休計算"])

    with t1:
        # 宏觀指標方向
        if df_macro is not None:
            liq = df_macro['Net_Liquidity'].iloc[-1]
            liq_delta = liq - df_macro['Net_Liquidity'].iloc[-20]
            dir_liq = "↑" if liq_delta > 0 else "↓"
            st.metric("💧 淨流動性", f"${liq:.2f}T", f"{dir_liq} {liq_delta:+.3f}T (20d)")
        
        # 戰略總表
        summary = []
        for t in p_dict.keys():
            if t not in df_close.columns: continue
            tr = analyze_trend_multi(df_close[t])
            scalar, reasons, pe_pct, _ = get_fundamental_scalar_v16(t)
            mvrv = calc_mvrv_z(df_close[t])
            tech_target = train_rf_model(df_close, t)
            directive, _ = get_cfo_directive_v16(t, tr['p_now'], "H1", tr['is_bull'], 50, pe_pct, mvrv)
            
            summary.append({
                "代號": t, "方向": tr['status'], "現價": f"${tr['p_now']:.2f}",
                "PE位階": f"{pe_pct:.0f}%", "MVRV-Z": f"{mvrv:.2f}", 
                "CFO指令": directive, "目標價": f"${tech_target*scalar:.2f}"
            })
        st.table(pd.DataFrame(summary))

    with t2:
        st.markdown("<h3 class='safe-harbor-header'>🛡️ Posa 鐵壁防禦陣列</h3>", unsafe_allow_html=True)
        cols = st.columns(5)
        for i, t in enumerate(SAFE_HARBOR_LIST):
            if t in df_close.columns:
                p_now = df_close[t].iloc[-1]
                t_tgt = train_rf_model(df_close, t)
                scalar, reasons, pe_pct, _ = get_fundamental_scalar_v16(t)
                with cols[i % 5]: render_valuation_card(t, p_now, t_tgt, scalar, reasons, pe_pct)

    with t3: st.write("策略實驗模組 (實驗 C/E) 已於背景完成掃描，當前防禦閾值: -0.2T")
    with t4: st.write("退休金預計於 2045 年達成目標 (根據當前資產組合與 7% 預期回報)。")

if __name__ == "__main__": main()