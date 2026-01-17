import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timedelta

# ==============================================================================
# 0. 全局環境設定 (Alpha 16.2)
# ==============================================================================
st.set_page_config(page_title="Alpha 16.2: 拓撲準確度中心", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .metric-card { background-color: #0E1117; border: 1px solid #444; border-radius: 5px; padding: 15px; color: white; }
    .bull-mode { color: #00FF7F; font-weight: bold; }
    .bear-mode { color: #FF4B4B; font-weight: bold; }
    .status-tag { padding: 2px 8px; border-radius: 4px; font-size: 0.9em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. 量化引擎：估值、技術與準確度
# ==============================================================================

@st.cache_data(ttl=3600*12)
def get_valuation_scalar(ticker):
    """計算基本面加權純量 (含 PE 位階、PEG、財報成長)"""
    try:
        stock = yf.Ticker(ticker); info = stock.info
        if info.get('quoteType') == 'ETF': return 1.0, 50.0, None
        
        fins = stock.quarterly_financials
        if fins.empty: fins = stock.financials
        
        score = 0
        pe_pct = 50.0
        
        # A. 營收成長
        if not fins.empty and 'Total Revenue' in fins.index and len(fins.columns) >= 2:
            growth = (fins.loc['Total Revenue'].iloc[0] - fins.loc['Total Revenue'].iloc[1]) / fins.loc['Total Revenue'].iloc[1]
            if growth > 0.1: score += 1
            elif growth < 0: score -= 1

        # B. PE Percentile (3年位階)
        hist = stock.history(period="3y")['Close']
        eps = info.get('trailingEps')
        curr_pe = info.get('trailingPE')
        if eps and eps > 0 and curr_pe:
            pe_series = hist / eps
            pe_pct = stats.percentileofscore(pe_series.dropna(), curr_pe)
            if pe_pct > 90: score -= 1.5
            elif pe_pct < 20: score += 1.5
            
        # C. PEG
        peg = info.get('pegRatio')
        if peg and 0 < peg < 1.2: score += 1

        scalar = max(0.85, min(1.15, 1.0 + (score * 0.05)))
        return scalar, pe_pct, peg
    except: return 1.0, 50.0, None

def calculate_daily_accuracy(ticker, df_close, scalar, test_days=60):
    """計算過去指定時間內的每日預測準確度"""
    series = df_close[ticker].dropna()
    results = []
    
    # 預測窗口為 14 天後的價格 (簡化回測以便於每日展示)
    window = 14 
    
    # 為了展示每日準確度，我們對過去 test_days 天進行回溯
    for i in range(len(series) - test_days - window, len(series) - window):
        train_data = series.iloc[:i]
        actual_future = series.iloc[i + window]
        
        # 簡單移動平均 + 波動率預測 (模擬技術面)
        tech_pred = train_data.iloc[-1] * (1 + train_data.pct_change().iloc[-20:].mean() * window)
        
        # 融合基本面權重
        final_pred = tech_target = tech_pred * scalar
        
        # 計算誤差百分比
        error = abs(final_pred - actual_future) / actual_future
        accuracy = max(0, 1 - error)
        
        results.append({
            "Date": series.index[i + window],
            "Actual": actual_future,
            "Predicted": final_pred,
            "Accuracy": accuracy
        })
    return pd.DataFrame(results)

# ==============================================================================
# 2. 界面與數據展示
# ==============================================================================

def main():
    st.sidebar.title("🦅 Alpha 16.2 指揮部")
    fred_key = st.sidebar.text_input("FRED API Key", type="password")
    user_input = st.sidebar.text_area("持倉清單 (代號, 金額)", "BTC-USD, 10000\nNVDA, 10000\n2330.TW, 10000\nCLS, 5000", height=120)
    p_dict = {l.split(',')[0].strip().upper(): float(l.split(',')[1]) for l in user_input.strip().split('\n') if ',' in l}
    
    # 準確度回測天數設定
    st.sidebar.markdown("---")
    backtest_range = st.sidebar.slider("準確度分析回測天數", 30, 120, 60)
    
    if not st.sidebar.button("🚀 啟動準確度掃描"): return

    with st.spinner("🦅 正在執行全量化分析..."):
        all_tickers = list(p_dict.keys())
        df_close = yf.download(all_tickers + ['^VIX'], period="2y", progress=False)['Close'].ffill()
        
        # 宏觀流動性 (如果有 Key)
        liq_val, liq_delta = 0.0, 0.0
        if fred_key:
            fred = Fred(api_key=fred_key)
            walcl = fred.get_series('WALCL').iloc[-1] / 1000000
            tga = fred.get_series('WTREGEN').iloc[-1] / 1000
            rrp = fred.get_series('RRPONTSYD').iloc[-1] / 1000
            liq_val = walcl - tga - rrp
            prev_liq = fred.get_series('WALCL').iloc[-20]/1000000 - fred.get_series('WTREGEN').iloc[-20]/1000 - fred.get_series('RRPONTSYD').iloc[-20]/1000
            liq_delta = liq_val - prev_liq

    # --- 戰略儀表板 ---
    st.title("🦅 Alpha 16.2: 戰略指揮與預測準確度中心")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        dir_icon = "↑" if liq_delta > 0 else "↓"
        st.metric("💧 全域淨流動性", f"${liq_val:.2f}T", f"{dir_icon} {liq_delta:+.3f}T", delta_color="normal")
    with m2:
        vix = df_close['^VIX'].iloc[-1]
        vix_delta = vix - df_close['^VIX'].iloc[-5]
        st.metric("🌪️ VIX 恐慌指數", f"{vix:.2f}", f"{vix_delta:+.2f} (5d)", delta_color="inverse")
    with m3:
        st.metric("📅 掃描時間", datetime.now().strftime("%Y-%m-%d"), "Alpha 16.2 Active")

    # --- 1. 即時戰略總表 ---
    st.subheader("⚔️ 指揮官戰略總表 (含指標方向)")
    summary = []
    for t in p_dict.keys():
        if t not in df_close.columns: continue
        p_now = df_close[t].iloc[-1]
        ma20 = df_close[t].rolling(20).mean().iloc[-1]
        scalar, pe_pct, peg = get_valuation_scalar(t)
        
        # 方向與狀態
        trend_dir = "↑" if p_now > ma20 else "↓"
        trend_class = "bull-mode" if p_now > ma20 else "bear-mode"
        pe_status = "💎 低估" if pe_pct < 20 else ("⚠️ 高估" if pe_pct > 85 else "⚖️ 合理")
        
        # 加密貨幣 MVRV-Z
        mvrv_z = 0
        if "USD" in t:
            mvrv_z = (p_now - df_close[t].rolling(200).mean().iloc[-1]) / df_close[t].rolling(200).std().iloc[-1]

        summary.append({
            "標的": t,
            "方向": f"{trend_dir}",
            "現價": f"${p_now:.2f}",
            "PE位階": f"{pe_pct:.0f}%",
            "PEG": f"{peg:.2f}" if peg else "N/A",
            "MVRV-Z": f"{mvrv_z:.2f}" if mvrv_z != 0 else "-",
            "估值狀態": pe_status,
            "財報加權": f"x{scalar:.2f}"
        })
    
    st.table(pd.DataFrame(summary))

    # --- 2. 歷史準確度每日追蹤 ---
    st.markdown("---")
    st.subheader("🎯 量化模型每日準確度追蹤 (Daily Accuracy Tracker)")
    st.write("此功能計算「技術預測 + 財報加權 + 估值修正」後的預測模型在過去一段時間內的真實表現。")
    
    acc_cols = st.columns(len(p_dict.keys()))
    for idx, t in enumerate(p_dict.keys()):
        if t not in df_close.columns: continue
        
        # 獲取標的基本面權重
        scalar, _, _ = get_valuation_scalar(t)
        # 執行每日準確度回測
        acc_df = calculate_daily_accuracy(t, df_close, scalar, test_days=backtest_range)
        
        with st.expander(f"📊 {t} 預測準確度分析", expanded=(idx==0)):
            c1, c2 = st.columns([2, 1])
            with c1:
                # 繪製預測 vs 真實
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=acc_df['Date'], y=acc_df['Actual'], name="真實股價", line=dict(color='#00FF7F')))
                fig.add_trace(go.Scatter(x=acc_df['Date'], y=acc_df['Predicted'], name="模型預測", line=dict(color='#FFA500', dash='dash')))
                fig.update_layout(title=f"{t} 預測 vs 真實軌跡", template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                avg_acc = acc_df['Accuracy'].mean()
                st.metric(f"{t} 平均準確度", f"{avg_acc:.2%}")
                # 準確度分布
                fig_acc = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = avg_acc * 100,
                    title = {'text': "Confidence Score"},
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00BFFF"}}
                ))
                fig_acc.update_layout(height=250, template="plotly_dark")
                st.plotly_chart(fig_acc, use_container_width=True)

if __name__ == "__main__": main()