import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==============================================================================
# 0. 全局設定
# ==============================================================================
st.set_page_config(page_title="Alpha 16.5: 30D 戰略指揮部", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .big-font { font-size: 1.2em !important; font-weight: bold; }
    .box-good { color: #00FF7F; }
    .box-bad { color: #FF4B4B; }
    .report-area { background-color: #262730; padding: 10px; border-radius: 5px; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. Alpha 16.5 核心引擎 (Grand Unified Model)
# ==============================================================================

@st.cache_data(ttl=3600*4)
def get_fundamental_data(ticker):
    """獲取基本面數據：PE, PEG, 品質因子"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 1. 獲取 EPS
        eps = info.get('trailingEps')
        if eps is None:
            fin = stock.quarterly_financials
            if not fin.empty and 'Basic EPS' in fin.index:
                eps = fin.loc['Basic EPS'].iloc[:4].sum()
        
        # 2. 計算 PE 位階 (模擬 3 年)
        hist = stock.history(period="3y")['Close']
        pe_pct = 50.0
        curr_pe = info.get('trailingPE')
        
        if eps and eps > 0:
            pe_series = hist / eps
            if not curr_pe: curr_pe = hist.iloc[-1] / eps
            pe_pct = stats.percentileofscore(pe_series.dropna(), curr_pe)
            
        # 3. 品質因子 (Quality) - 使用淨利率代理
        margin = info.get('profitMargins', 0.1)
        quality_mult = 1.0
        if margin > 0.20: quality_mult = 1.15 # 高品質溢價 (NVDA, META)
        elif margin < 0.05: quality_mult = 0.90 # 低品質折價
        
        return pe_pct, quality_mult, info.get('pegRatio', 2.0)
    except:
        return 50.0, 1.0, 2.0

def run_30d_unified_model(ticker, df_close, pe_pct, quality_mult, lookback_days=250):
    """執行 30天 全因子滾動預測"""
    series = df_close[ticker].dropna()
    results = []
    
    # 至少需要 200 天 MA + 30 天預測窗口
    start_idx = max(200, len(series) - lookback_days - 30)
    
    ma200_series = series.rolling(200).mean()
    vol_series = series.pct_change().rolling(30).std() * np.sqrt(30)
    
    is_crypto = "USD" in ticker or "BTC" in ticker
    
    for i in range(start_idx, len(series) - 30):
        # T日 數據
        date_t = series.index[i]
        price_t = series.iloc[i]
        bias_t = (price_t - ma200_series.iloc[i]) / ma200_series.iloc[i]
        vol_t = vol_series.iloc[i]
        
        # --- 預測演算法 ---
        # 1. 估值重力 (模擬動態 PE)
        sim_pe = pe_pct
        if bias_t > 0.3: sim_pe = 95
        elif bias_t < -0.2: sim_pe = 10
        
        gravity = 0
        if sim_pe > 85: gravity = -0.06
        elif sim_pe < 15: gravity = 0.08
        
        # 2. 品質加權
        gravity *= quality_mult
        
        # 3. 趨勢慣性 (乖離過大煞車)
        mom = (price_t - series.iloc[i-30]) / series.iloc[i-30]
        if bias_t > 0.45: mom = 0
        
        # 4. 綜合回報
        exp_ret = (mom * 0.4) + gravity
        if is_crypto and bias_t > 0.8: exp_ret -= 0.1 # Crypto 泡沫修正
        
        # 5. 預測箱體
        pred_mean = price_t * (1 + exp_ret)
        upper = pred_mean * (1 + vol_t * 1.5)
        lower = pred_mean * (1 - vol_t * 1.5)
        
        # T+30日 真實結果
        price_actual = series.iloc[i+30]
        date_future = series.index[i+30]
        
        in_box = lower <= price_actual <= upper
        dir_correct = (pred_mean > price_t) == (price_actual > price_t)
        
        results.append({
            "Date": date_future,
            "Actual": price_actual,
            "Pred": pred_mean,
            "Upper": upper,
            "Lower": lower,
            "In_Box": in_box,
            "Dir_Correct": dir_correct
        })
        
    return pd.DataFrame(results)

# ==============================================================================
# 2. 主界面
# ==============================================================================

def main():
    st.sidebar.title("🦅 Alpha 16.5 指揮部")
    user_input = st.sidebar.text_area("持倉清單 (代號, 份額)", "NVDA, 1000\nAMD, 1000\nCLS, 500\nSOXL, 2000\n2330.TW, 1000\nBTC-USD, 500", height=150)
    p_dict = {l.split(',')[0].strip().upper(): float(l.split(',')[1]) for l in user_input.strip().split('\n') if ',' in l}
    
    backtest_range = st.sidebar.slider("回測樣本天數", 100, 400, 250)
    
    if not st.sidebar.button("🚀 啟動 30D 戰略掃描"): return

    with st.spinner("正在進行全因子演算 (PE + Quality + Trend)..."):
        # 下載數據
        df_close = yf.download(list(p_dict.keys()), period="2y", progress=False)['Close'].ffill()

    st.title("🦅 Alpha 16.5: 30D 全因子戰略預測")
    st.markdown("此模型融合 **估值重力、品質因子、趨勢慣性**，預測未來 30 天的股價機率箱體。")
    
    report_text = f"=== Alpha 16.5 診斷報告 ({datetime.now().strftime('%Y-%m-%d')}) ===\n"
    report_text += f"參數: 樣本={backtest_range}天 | 預測窗口=30天\n\n"

    # --- 逐一標的分析 ---
    tabs = st.tabs(list(p_dict.keys()))
    
    for i, ticker in enumerate(p_dict.keys()):
        if ticker not in df_close.columns: continue
        
        with tabs[i]:
            # 1. 執行運算
            pe_pct, qual, peg = get_fundamental_data(ticker)
            res_df = run_30d_unified_model(ticker, df_close, pe_pct, qual, lookback_days=backtest_range)
            
            if res_df.empty:
                st.error("數據不足，無法運算")
                continue

            # 統計指標
            acc_box = res_df['In_Box'].mean()
            acc_dir = res_df['Dir_Correct'].mean()
            last_pred = res_df.iloc[-1]
            curr_price = df_close[ticker].iloc[-1]
            
            # 生成報告文字
            signal = "🟢 強勢" if last_pred['Pred'] > curr_price else "🔴 修正"
            report_text += f"[{ticker}]\n"
            report_text += f"  - PE位階: {pe_pct:.0f}% | 品質加權: x{qual:.2f}\n"
            report_text += f"  - 箱體捕獲率: {acc_box:.1%} | 方向勝率: {acc_dir:.1%}\n"
            report_text += f"  - 30天信號: {signal} (目標 ${last_pred['Pred']:.2f})\n\n"

            # 2. 顯示儀表板
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PE 歷史位階", f"{pe_pct:.0f}%", "高於 85% 警戒" if pe_pct>85 else "安全")
            c2.metric("品質加權", f"x{qual:.2f}", "ROIC / Margin")
            c3.metric("箱體捕獲率 (風險)", f"{acc_box:.1%}", "越低風險越高")
            c4.metric("方向勝率 (30D)", f"{acc_dir:.1%}", "趨勢可靠度")
            
            # 3. 繪圖
            fig = go.Figure()
            # 預測箱體
            fig.add_trace(go.Scatter(
                x=res_df['Date'].tolist() + res_df['Date'].tolist()[::-1],
                y=res_df['Upper'].tolist() + res_df['Lower'].tolist()[::-1],
                fill='toself', fillcolor='rgba(0,191,255,0.15)', line=dict(color='rgba(255,255,255,0)'),
                name='30D 機率箱體'
            ))
            # 預測中軸
            fig.add_trace(go.Scatter(x=res_df['Date'], y=res_df['Pred'], name='預測路徑', line=dict(color='orange', dash='dash')))
            # 真實價格
            fig.add_trace(go.Scatter(x=res_df['Date'], y=res_df['Actual'], name='真實走勢', line=dict(color='#00FF7F', width=2)))
            
            fig.update_layout(title=f"{ticker} 30天 戰略預測驗證", template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)
            
            # 4. 顯示最近 5 筆預測
            st.subheader("📋 最近 5 筆預測驗證")
            st.dataframe(res_df.tail(5).style.format({"Actual": "{:.2f}", "Pred": "{:.2f}", "Upper": "{:.2f}", "Lower": "{:.2f}"}))

    # --- 報告生成區 ---
    st.markdown("---")
    st.subheader("📋 生成 AI 診斷報告")
    st.info("請複製下方文字，貼回對話視窗，讓我為您進行深度解讀：")
    st.code(report_text, language='text')

if __name__ == "__main__": main()