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
st.set_page_config(page_title="Alpha 4.1: 全雲端戰略版", layout="wide", page_icon="🦅")

# 自定義 CSS (黑金風格)
st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #262730; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
    .rrg-box {border: 1px solid #444; padding: 10px; border-radius: 5px; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 (網路實時抓取) ---
@st.cache_data(ttl=1800) # 每30分鐘更新一次
def fetch_market_data(tickers):
    # 強制加入 SPY (基準), QQQ (科技基準), HYG (債券), VIX (恐慌)
    benchmarks = ['SPY', 'QQQ', 'BTC-USD', '^VIX', '^TNX', 'HYG']
    all_tickers = list(set(tickers + benchmarks))
    
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    
    # 建立進度條
    progress_bar = st.progress(0, text="☁️ Alpha 正在連線華爾街資料庫...")
    
    for i, t in enumerate(all_tickers):
        try:
            progress_bar.progress((i + 1) / len(all_tickers), text=f"下載數據中: {t} ...")
            # 抓取 2 年數據以計算長期均線與 RRG
            df = yf.Ticker(t).history(period="2y", auto_adjust=True)
            if df.empty: continue
            
            data['Close'][t] = df['Close']
            data['Open'][t] = df['Open']
            data['High'][t] = df['High']
            data['Low'][t] = df['Low']
            data['Volume'][t] = df['Volume']
        except: continue
            
    progress_bar.empty()
    # 將 dict 轉為 DataFrame 並處理缺失值
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

# --- 2. 進階數據引擎 (基本面/籌碼) ---
@st.cache_data(ttl=3600*24) # 基本面一天更新一次即可
def get_advanced_info(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # A. 基本面 (Rule of 40)
        # Yahoo Finance 數據通常是小數 (例如 0.25 代表 25%)
        rev_growth = info.get('revenueGrowth')
        profit_margin = info.get('profitMargins')
        
        rule_of_40 = None
        if rev_growth is not None and profit_margin is not None:
            rule_of_40 = (rev_growth + profit_margin) * 100
            
        # B. 機構籌碼
        inst_held = info.get('heldPercentInstitutions')
        insider_held = info.get('heldPercentInsiders')
        short_ratio = info.get('shortRatio')
        
        # C. 華爾街目標
        target_mean = info.get('targetMeanPrice')
        
        return {
            'Rule40': rule_of_40,
            'Rev_Growth': rev_growth,
            'Profit_Margin': profit_margin,
            'PEG': info.get('pegRatio'),
            'Inst_Held': inst_held,
            'Short_Ratio': short_ratio,
            'Target_Mean': target_mean,
            'PE': info.get('forwardPE')
        }
    except: return {}

# --- 3. RRG 動態運算核心 (Python 實時版) ---
def calc_rrg_metrics(df_close, tickers, benchmark='SPY'):
    """
    完全不依賴 Excel，直接用 Python 計算 JdK RRG 指標
    """
    if benchmark not in df_close.columns: return pd.DataFrame()
    
    rrg_data = []
    bench_close = df_close[benchmark]
    
    for t in tickers:
        if t not in df_close.columns or t == benchmark: continue
        
        # 1. 相對強度 (RS)
        rs = df_close[t] / bench_close
        
        # 2. RRG 核心邏輯 (簡化版 JdK RS-Ratio)
        # RS-Ratio = (短期RS均線 / 長期RS均線) * 100
        # 這裡設定 Short=10天, Long=60天 (適合波段)
        rs_mean_short = rs.rolling(10).mean()
        rs_mean_long = rs.rolling(60).mean()
        
        if len(rs_mean_short.dropna()) < 60: continue

        rs_ratio = (rs_mean_short / rs_mean_long * 100).iloc[-1]
        
        # 3. RS-Momentum (動能)
        # RS-Momentum = (RS-Ratio 的變化率)
        # ((當前 Ratio - 10天前 Ratio) * 係數) + 100
        rs_ratio_series = rs_mean_short / rs_mean_long * 100
        change = rs_ratio_series.iloc[-1] - rs_ratio_series.iloc[-10]
        rs_momentum = (change * 5) + 100 # *5 是為了放大波動，讓圖表更易讀
        
        # 4. 象限判定
        if rs_ratio > 100 and rs_momentum > 100: quadrant = "🟢 領先 (Leading)"
        elif rs_ratio > 100 and rs_momentum < 100: quadrant = "🟡 轉弱 (Weakening)"
        elif rs_ratio < 100 and rs_momentum < 100: quadrant = "🔴 落後 (Lagging)"
        else: quadrant = "🔵 改善 (Improving)"
        
        rrg_data.append({
            'Ticker': t,
            'RS_Ratio': rs_ratio,
            'RS_Momentum': rs_momentum,
            'Quadrant': quadrant
        })
        
    return pd.DataFrame(rrg_data)

# --- 4. 輔助運算函式 ---
def format_number(num):
    if num is None: return "N/A"
    abs_num = abs(num)
    if abs_num >= 1_000_000: return f"{num/1_000_000:.2f}M"
    elif abs_num >= 1_000: return f"{num/1_000:.2f}K"
    else: return f"{num:.2f}"

def calc_targets(close, high, low):
    # 這裡一次計算三種目標，減少代碼重複
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
        for _ in range(300): # 模擬300次
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

def calc_fund_flow(close, volume):
    if volume is None or volume.empty: return None, None
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    
    # 斜率
    y, x = obv.values[-20:].reshape(-1, 1), np.arange(20).reshape(-1, 1)
    slope = LinearRegression().fit(x, y).coef_[0].item()
    
    return slope, obv

def analyze_trend(series):
    if series is None or len(series) < 200: return None
    p_now = series.iloc[-1]
    sma200 = series.rolling(200).mean().iloc[-1]
    
    status = "🛡️ 區間"
    if p_now < sma200: status = "🛑 熊市"
    elif p_now > series.ewm(span=20).mean().iloc[-1]: status = "🔥 進攻"
    
    # 預測
    y, x = series.dropna().values.reshape(-1, 1), np.arange(len(series.dropna())).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    p_2w = model.predict([[len(y)+10]])[0].item()
    p_1m = model.predict([[len(y)+22]])[0].item()
    
    return {"status": status, "p_now": p_now, "p_2w": p_2w, "p_1m": p_1m, "sma200": sma200}

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
    st.title("Alpha 4.1: 全雲端戰略版")
    st.caption("v4.1 | 移除 Excel 依賴 | RRG / 籌碼 / 財報 實時連線")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ 參數設定")
        fred_key = st.secrets.get("FRED_API_KEY", st.text_input("FRED API Key (選填)", type="password"))
        
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
        
        # 新增：簡易財務目標 (取代 Excel)
        with st.expander("💰 簡易財務目標 (FIRE)"):
            goal = st.number_input("退休目標金額", value=30000000)
            st.progress(min(total_value / goal, 1.0))
            st.caption(f"達成率: {total_value/goal:.1%}")

        if st.button("🚀 啟動全域掃描", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False):
        st.info("👈 請點擊『啟動全域掃描』。系統將直接連線華爾街資料庫。")
        return

    # --- 數據下載區 ---
    with st.spinner("☁️ Alpha 正在雲端下載與運算..."):
        df_close, df_open, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_liquidity = fetch_fred_liquidity(fred_key)
        adv_data = {t: get_advanced_info(t) for t in tickers_list}

    if df_close.empty: st.error("數據獲取失敗，請檢查代號是否正確。"); return

    # --- 1. 宏觀 (Macro) ---
    st.subheader("1. 宏觀與流動性 (Macro & Liquidity)")
    vix = df_close.get('^VIX').iloc[-1] if '^VIX' in df_close else None
    hyg = analyze_trend(df_close.get('HYG'))
    
    liq_s = "未知"
    if df_liquidity is not None:
        curr, prev = df_liquidity['Net_Liquidity'].iloc[-1], df_liquidity['Net_Liquidity'].iloc[-5]
        liq_s = "擴張 (印鈔中)" if curr > prev else "收縮 (抽水中)"
    
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("美元淨流動性", liq_s, f"${df_liquidity['Net_Liquidity'].iloc[-1]:.2f}T" if df_liquidity is not None else "No Key")
    with c2: st.metric("信用市場 (HYG)", "充裕" if hyg and hyg['p_now'] > hyg['sma200'] else "枯竭")
    with c3: st.metric("VIX 恐慌指數", f"{vix:.2f}" if vix else "N/A", delta="風暴" if vix and vix>22 else "平靜", delta_color="inverse")
    st.markdown("---")

    # --- 2. RRG 動態輪動 (Live Calculation) ---
    st.subheader("2. 雲端 RRG 板塊輪動 (Live RRG)")
    st.markdown("直接運算 **相對於 SPY** 的強度與動能。不依賴 Excel，即時顯示資金流向。")
    
    rrg_df = calc_rrg_metrics(df_close, tickers_list)
    if not rrg_df.empty:
        fig_rrg = px.scatter(rrg_df, x='RS_Ratio', y='RS_Momentum', color='Quadrant', text='Ticker',
                             title="RRG 動態輪動 (vs SPY)", 
                             color_discrete_map={'🟢 領先 (Leading)': '#00FF7F', '🟡 轉弱 (Weakening)': '#FFFF00',
                                                 '🔴 落後 (Lagging)': '#FF4B4B', '🔵 改善 (Improving)': '#00BFFF'})
        # 畫十字線
        fig_rrg.add_vline(x=100, line_width=1, line_dash="dash", line_color="gray")
        fig_rrg.add_hline(y=100, line_width=1, line_dash="dash", line_color="gray")
        fig_rrg.update_layout(xaxis_title="RS-Ratio (趨勢強度)", yaxis_title="RS-Momentum (動能速度)", height=500)
        st.plotly_chart(fig_rrg, use_container_width=True)
    else:
        st.warning("⚠️ 數據不足，無法繪製 RRG (需要至少 60 天歷史數據)。")
    st.markdown("---")

    # --- 3. 深度審計 (Deep Audit) ---
    st.subheader("3. 深度資產審計 (Fundamental & Institutional)")
    
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        slope, obv_series = calc_fund_flow(df_close[ticker], df_vol[ticker])
        info = adv_data.get(ticker, {})
        
        # 三角定位
        t_atr, t_mc, t_fib = calc_targets(df_close[ticker], df_high[ticker], df_low[ticker])
        
        # Rule of 40
        r40 = info.get('Rule40')
        r40_badge = "✅ 通過" if r40 and r40 > 40 else ("❌ 未通過" if r40 else "N/A")
        
        # 繪圖
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_close.index[-100:], y=df_close[ticker].iloc[-100:], name='Price', line=dict(color='#00FF7F')))
        if obv_series is not None:
             fig.add_trace(go.Scatter(x=df_close.index[-100:], y=obv_series.iloc[-100:], name='OBV', line=dict(color='#00BFFF'), yaxis='y2'))
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0),
                          yaxis2=dict(overlaying='y', side='right', showgrid=False))

        with st.expander(f"📊 {ticker} - {trend['status']} | Rule of 40: {r40_badge}", expanded=True):
            k1, k2, k3 = st.columns([2, 1, 1])
            
            with k1: # 技術與圖表
                st.markdown("#### 🎯 四角定位 & 走勢")
                col_a, col_b = st.columns(2)
                col_a.write(f"**ATR Target:** ${t_atr:.2f}" if t_atr else "-")
                col_a.write(f"**Monte Carlo:** ${t_mc:.2f}" if t_mc else "-")
                col_b.write(f"**Fibonacci:** ${t_fib:.2f}" if t_fib else "-")
                col_b.write(f"**Wall St.:** ${info.get('Target_Mean')}" if info.get('Target_Mean') else "-")
                st.plotly_chart(fig, use_container_width=True)

            with k2: # 籌碼 (Live)
                st.markdown("#### 🏦 機構籌碼")
                inst = info.get('Inst_Held')
                st.metric("機構持股比", f"{inst*100:.1f}%" if inst else "N/A", 
                          delta="高度控盤" if inst and inst > 0.7 else "散戶多")
                
                obv_s = "吸籌" if slope and slope > 0 else "出貨"
                st.metric("OBV 資金流", format_number(slope), obv_s)
                st.caption(f"空單比例: {info.get('Short_Ratio', 0)}")

            with k3: # 基本面 (Live)
                st.markdown("#### 💎 財報體質")
                st.metric("Rule of 40", f"{r40:.1f}" if r40 else "N/A", delta=r40_badge)
                peg = info.get('PEG')
                st.metric("PEG 估值", f"{peg}" if peg else "N/A", delta="低估" if peg and peg < 1 else "偏高", delta_color="inverse")
                
                st.write("**三階段推演:**")
                st.caption(f"2週: ${trend['p_2w']:.2f}")
                st.caption(f"1月: ${trend['p_1m']:.2f}")

    st.markdown("---")
    
    # --- 4. 總表 ---
    st.subheader("4. 資產配置總表")
    table_data = []
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        info = adv_data.get(ticker, {})
        slope, _ = calc_fund_flow(df_close[ticker], df_vol[ticker])
        
        table_data.append({
            "代號": ticker,
            "現價": f"${trend['p_now']:.2f}",
            "趨勢": trend['status'],
            "Rule 40": f"{info.get('Rule40', 0):.1f}" if info.get('Rule40') else "-",
            "機構持股": f"{info.get('Inst_Held', 0)*100:.0f}%" if info.get('Inst_Held') else "-",
            "OBV": "流入" if slope and slope > 0 else "流出"
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.info("💡 系統說明：本版本所有數據 (股價、財報、籌碼) 皆透過 API 實時連線華爾街資料庫，無須上傳 Excel。")

if __name__ == "__main__":
    main()