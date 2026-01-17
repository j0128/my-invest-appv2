import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timedelta

# ==============================================================================
# 0. 全局環境設定
# ==============================================================================
st.set_page_config(page_title="Alpha 16.4: 14D 每日驗證中心", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .bull-mode { color: #00FF7F; font-weight: bold; }
    .bear-mode { color: #FF4B4B; font-weight: bold; }
    .correct-tag { background-color: #006400; color: #00FF7F; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
    .wrong-tag { background-color: #8B0000; color: #FF4B4B; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. 核心量化函數
# ==============================================================================

@st.cache_data(ttl=3600*12)
def get_valuation_scalar(ticker):
    try:
        stock = yf.Ticker(ticker); info = stock.info
        hist = stock.history(period="3y")['Close']
        eps = info.get('trailingEps'); curr_pe = info.get('trailingPE')
        pe_pct = 50.0
        if eps and eps > 0 and curr_pe:
            pe_series = hist / eps
            pe_pct = stats.percentileofscore(pe_series.dropna(), curr_pe)
        score = 0
        if pe_pct > 90: score -= 1.5
        elif pe_pct < 20: score += 1.5
        return max(0.85, min(1.15, 1.0 + (score * 0.05))), pe_pct
    except: return 1.0, 50.0

def generate_daily_report(ticker, df_close, scalar):
    """
    生成過去 14 天的每日預測與真實值對照表
    邏輯：對於 T 日，抓取 T-14 日時模型做出的預測。
    """
    series = df_close[ticker].dropna()
    window = 14
    report_data = []
    
    # 我們分析最近的 14 個交易日
    for i in range(len(series) - 14, len(series)):
        # 預測日 (T-14)
        pred_made_idx = i - window
        if pred_made_idx < 0: continue
        
        base_price = series.iloc[pred_made_idx]
        actual_price = series.iloc[i]
        date = series.index[i]
        
        # 模擬當時的預測 (技術動能 + 財報加權)
        # 抓取 pred_made_idx 之前的 20 天動能
        lookback_vol = series.iloc[pred_made_idx-20 : pred_made_idx].pct_change().mean()
        pred_price = base_price * (1 + lookback_vol * window) * scalar
        
        pred_dir = "↑" if pred_price > base_price else "↓"
        actual_dir = "↑" if actual_price > base_price else "↓"
        
        is_correct = pred_dir == actual_dir
        error = abs(pred_price - actual_price) / actual_price
        
        report_data.append({
            "日期": date.strftime("%m-%d"),
            "真實股價": f"${actual_price:.2f}",
            "預測股價": f"${pred_price:.2f}",
            "預測方向": pred_dir,
            "真實方向": actual_dir,
            "方向正確": "✅ 正確" if is_correct else "❌ 誤差",
            "誤差值": f"{error:.1%}"
        })
    return pd.DataFrame(report_data)

# ==============================================================================
# 2. 界面渲染
# ==============================================================================

def main():
    st.sidebar.title("🦅 Alpha 16.4 指揮部")
    user_input = st.sidebar.text_area("持倉清單", "NVDA, 10000\nAMD, 10000\nCLS, 5000", height=120)
    p_dict = {l.split(',')[0].strip().upper(): float(l.split(',')[1]) for l in user_input.strip().split('\n') if ',' in l}
    
    if not st.sidebar.button("🚀 啟動 14D 驗證"): return

    with st.spinner("正在對沖 14 天歷史數據..."):
        df_close = yf.download(list(p_dict.keys()), period="1y", progress=False)['Close'].ffill()

    st.title("🦅 Alpha 16.4: 14D 每日預測準確度驗證")
    st.markdown("此分頁將展示模型在過去 14 天中，每一天對當下價格與方向預測的**實戰表現**。")

    for t in p_dict.keys():
        if t not in df_close.columns: continue
        
        scalar, pe_pct = get_valuation_scalar(t)
        report_df = generate_daily_report(t, df_close, scalar)
        
        # 計算此標的的 Hit Rate (勝率)
        hit_rate = (report_df["方向正確"] == "✅ 正確").mean()
        
        with st.expander(f"📊 {t} 每日預測對沖報表 (勝率: {hit_rate:.1%})", expanded=True):
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.metric("P/E 歷史位階", f"{pe_pct:.0f}%")
                st.metric("財報修正權重", f"x{scalar:.2f}")
                
                # 方向與誤差分類說明
                st.markdown("""
                **分類說明：**
                1. **方向預測**：判斷 T-14 至 T 日的趨勢性質。
                2. **誤差範圍**：預測值與真實值的絕對偏離度。
                """)
            
            with c2:
                # 使用 HTML 渲染表格以顯示標籤顏色
                st.dataframe(report_df, use_container_width=True)
                
        # 繪製該標的的預測曲線對比
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=report_df["日期"], y=report_df["真實股價"].str.replace('$','').astype(float), name="真實 (Actual)", line=dict(color='#00FF7F')))
        fig.add_trace(go.Scatter(x=report_df["日期"], y=report_df["預測股價"].str.replace('$','').astype(float), name="預測 (Predicted)", line=dict(color='#FFA500', dash='dash')))
        fig.update_layout(title=f"{t} 預測軌跡對比", template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__": main()