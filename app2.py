import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timedelta

# ==============================================================================
# 0. 全局環境設定 (Alpha 16.3)
# ==============================================================================
st.set_page_config(page_title="Alpha 16.3: 量化準確度指揮部", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .bull-mode { color: #00FF7F; font-weight: bold; }
    .bear-mode { color: #FF4B4B; font-weight: bold; }
    .accuracy-high { color: #00FF7F; font-weight: bold; }
    .accuracy-low { color: #FFD700; font-weight: bold; }
    .accuracy-danger { color: #FF4B4B; font-weight: bold; }
    .card { background-color: #0E1117; border: 1px solid #444; border-radius: 8px; padding: 15px; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. 雙核心量化引擎：方向與準確度
# ==============================================================================

@st.cache_data(ttl=3600*12)
def get_valuation_logic(ticker):
    """計算 PE Percentile 與 基本面加權"""
    try:
        stock = yf.Ticker(ticker); info = stock.info
        hist = stock.history(period="3y")['Close']
        eps = info.get('trailingEps'); curr_pe = info.get('trailingPE')
        pe_pct = 50.0
        if eps and eps > 0 and curr_pe:
            pe_series = hist / eps
            pe_pct = stats.percentileofscore(pe_series.dropna(), curr_pe)
        
        # 簡單加權邏輯
        score = 0
        if pe_pct > 90: score -= 1.5
        elif pe_pct < 20: score += 1.5
        
        scalar = max(0.85, min(1.15, 1.0 + (score * 0.05)))
        return scalar, pe_pct, info.get('pegRatio')
    except: return 1.0, 50.0, None

def run_accuracy_backtest(ticker, df_close, scalar, days=60):
    """執行歷史準確度分析，返回每日誤差序列"""
    series = df_close[ticker].dropna()
    window = 14
    results = []
    
    # 滾動回測過去 days 天
    for i in range(len(series) - days - window, len(series) - window):
        train = series.iloc[:i]
        actual = series.iloc[i + window]
        
        # 技術面預測 (簡單動量 + 均線)
        tech_pred = train.iloc[-1] * (1 + train.pct_change().iloc[-20:].mean() * window)
        final_pred = tech_pred * scalar
        
        error = abs(final_pred - actual) / actual
        # 方向判定：預測漲且實際漲，或預測跌且實際跌
        dir_correct = (final_pred > train.iloc[-1]) == (actual > train.iloc[-1])
        
        results.append({
            "Date": series.index[i + window],
            "Actual": actual,
            "Predicted": final_pred,
            "Error": error,
            "Dir_Correct": dir_correct
        })
    return pd.DataFrame(results)

# ==============================================================================
# 2. 界面渲染
# ==============================================================================

def main():
    st.sidebar.title("🦅 Alpha 16.3 準確度實驗室")
    user_input = st.sidebar.text_area("持倉清單", "BTC-USD, 10000\nNVDA, 10000\nAMD, 10000\nCLS, 5000", height=120)
    p_dict = {l.split(',')[0].strip().upper(): float(l.split(',')[1]) for l in user_input.strip().split('\n') if ',' in l}
    
    backtest_range = st.sidebar.slider("分析天數", 30, 120, 60)
    if not st.sidebar.button("🚀 執行量化掃描"): return

    with st.spinner("正在掃描方向與誤差範圍..."):
        df_close = yf.download(list(p_dict.keys()) + ['^VIX'], period="2y", progress=False)['Close'].ffill()

    st.title("🦅 Alpha 16.3: 戰略預測與準確度中心")
    
    # --- 1. 方向與誤差總表 ---
    st.subheader("⚔️ 指揮官戰略總表：方向與誤差範圍 (Accuracy)")
    summary = []
    for t in p_dict.keys():
        if t not in df_close.columns: continue
        p_now = df_close[t].iloc[-1]
        ma20 = df_close[t].rolling(20).mean().iloc[-1]
        scalar, pe_pct, peg = get_valuation_logic(t)
        
        # 執行準確度回測
        acc_df = run_accuracy_backtest(t, df_close, scalar, days=backtest_range)
        avg_acc = 1 - acc_df['Error'].mean()
        hit_rate = acc_df['Dir_Correct'].mean()
        
        # 方向判定
        trend_icon = "↑" if p_now > ma20 else "↓"
        trend_style = "bull-mode" if p_now > ma20 else "bear-mode"
        
        # 誤差顏色
        acc_style = "accuracy-high" if avg_acc > 0.85 else ("accuracy-low" if avg_acc > 0.75 else "accuracy-danger")

        summary.append({
            "標的": t,
            "方向預測 (14D)": f"{trend_icon}",
            "方向勝率 (Hit Rate)": f"{hit_rate:.1%}",
            "平均準確度 (Accuracy)": f"{avg_acc:.1%}",
            "誤差範圍 (MAPE)": f"±{1-avg_acc:.1%}",
            "PE位階": f"{pe_pct:.0f}%",
            "加權狀態": "💎 低估加成" if pe_pct < 20 else ("⚠️ 高估懲罰" if pe_pct > 85 else "⚖️ 合理")
        })
    
    # 渲染自定義 HTML 表格以呈現顏色
    st.table(pd.DataFrame(summary))

    # --- 2. 每日準確度趨勢圖 ---
    st.markdown("---")
    st.subheader("🎯 預測軌跡與誤差範圍 (Daily Tracker)")
    
    cols = st.columns(2)
    for idx, t in enumerate(p_dict.keys()):
        if t not in df_close.columns: continue
        scalar, _, _ = get_valuation_logic(t)
        acc_df = run_accuracy_backtest(t, df_close, scalar, days=backtest_range)
        
        with cols[idx % 2]:
            st.markdown(f"#### {t} 預測 vs 真實")
            fig = go.Figure()
            # 繪製真實價格
            fig.add_trace(go.Scatter(x=acc_df['Date'], y=acc_df['Actual'], name="真實 (Actual)", line=dict(color='#00FF7F', width=2)))
            # 繪製預測價格
            fig.add_trace(go.Scatter(x=acc_df['Date'], y=acc_df['Predicted'], name="預測 (Predicted)", line=dict(color='#FFA500', dash='dash')))
            # 繪製誤差帶 (Error Band)
            fig.add_trace(go.Scatter(
                x=acc_df['Date'].tolist() + acc_df['Date'].tolist()[::-1],
                y=(acc_df['Predicted'] * 1.05).tolist() + (acc_df['Predicted'] * 0.95).tolist()[::-1],
                fill='toself', fillcolor='rgba(255,165,0,0.1)', line=dict(color='rgba(255,255,255,0)'),
                name="5% 誤差邊界"
            ))
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__": main()