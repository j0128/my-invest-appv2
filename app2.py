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
st.set_page_config(page_title="Alpha 4.0: 全域戰略旗艦版", layout="wide", page_icon="🦅")

# 自定義 CSS (黑金風格)
st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #262730; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
    .quadrant-box {background-color: #1E1E1E; padding: 10px; border-radius: 5px; text-align: center;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 (Data Engine) ---
@st.cache_data(ttl=3600)
def fetch_market_data(tickers):
    # 增加 SPY 作為 RRG 基準
    benchmarks = ['QQQ', 'SPY', 'BTC-USD', '^VIX', '^TNX', 'HYG']
    all_tickers = list(set(tickers + benchmarks))
    
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    
    progress_bar = st.progress(0, text="Alpha 4.0 正在建立全域連線...")
    
    for i, t in enumerate(all_tickers):
        try:
            progress_bar.progress((i + 1) / len(all_tickers), text=f"正在下載: {t} ...")
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

# --- 2. 新增：基本面與機構籌碼引擎 ---
@st.cache_data(ttl=3600*12)
def get_advanced_info(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # 1. 基本面數據 (Rule of 40 & PEG)
        rev_growth = info.get('revenueGrowth', 0)
        profit_margin = info.get('profitMargins', 0)
        rule_of_40_score = (rev_growth + profit_margin) * 100 if rev_growth and profit_margin else None
        
        # 2. 機構籌碼數據 (COT Proxy)
        inst_held = info.get('heldPercentInstitutions', 0)
        insider_held = info.get('heldPercentInsiders', 0)
        short_ratio = info.get('shortRatio', 0)
        
        # 3. 華爾街目標
        target_mean = info.get('targetMeanPrice', None)
        
        return {
            'Rule40': rule_of_40_score,
            'PEG': info.get('pegRatio', None),
            'Inst_Held': inst_held,
            'Insider_Held': insider_held,
            'Short_Ratio': short_ratio,
            'Target_Mean': target_mean,
            'PE': info.get('forwardPE', None)
        }
    except: return {}

# --- 3. 新增：RRG 動能輪動算法 ---
def calc_rrg_metrics(df_close, tickers, benchmark='SPY'):
    if benchmark not in df_close.columns: return pd.DataFrame()
    
    rrg_data = []
    bench_close = df_close[benchmark]
    
    for t in tickers:
        if t not in df_close.columns or t == benchmark: continue
        
        # 1. 計算相對強度 (Relative Strength)
        rs = df_close[t] / bench_close
        
        # 2. JdK RS-Ratio (趨勢): 100日均線的標準化
        # 這裡用簡化版算法：RS的短期均線 / RS的長期均線 * 100
        rs_mean_short = rs.rolling(10).mean()
        rs_mean_long = rs.rolling(100).mean()
        rs_ratio = (rs_mean_short / rs_mean_long * 100).iloc[-1]
        
        # 3. JdK RS-Momentum (動能): RS-Ratio 的變化率
        # 這裡用簡化版：(當前Ratio - 10天前Ratio) + 100
        rs_ratio_series = rs_mean_short / rs_mean_long * 100
        rs_momentum = ((rs_ratio_series.iloc[-1] - rs_ratio_series.iloc[-10]) * 10) + 100 # 放大波動以便觀察
        
        # 4. 決定象限
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

# --- 4. 既有算法 (三角定位、資金流、趨勢) ---
def format_number(num):
    if num is None: return "N/A"
    abs_num = abs(num)
    if abs_num >= 1_000_000: return f"{num/1_000_000:.2f}M"
    elif abs_num >= 1_000: return f"{num/1_000:.2f}K"
    else: return f"{num:.2f}"

def calc_atr_target(close, high, low):
    try:
        prev_close = close.shift(1)
        tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        return close.iloc[-1] + atr * np.sqrt(22) * 1.2
    except: return None

def calc_monte_carlo_target(series):
    try:
        returns = series.pct_change().dropna()
        last_price = series.iloc[-1]
        mu, sigma = returns.mean(), returns.std()
        sim_df = pd.DataFrame()
        for i in range(500):
            daily_vol = np.random.normal(mu, sigma, 22)
            prices = [last_price]
            for x in daily_vol: prices.append(prices[-1]*(1+x))
            sim_df[i] = prices
        return np.percentile(sim_df.iloc[-1], 50)
    except: return None

def calc_fib_target(series):
    try:
        rw = series.iloc[-60:]
        return rw.max() + (rw.max() - rw.min()) * 0.618
    except: return None

def calc_fund_flow(close, high, low, volume):
    if volume is None or volume.empty: return None
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    y, x = obv.values[-20:].reshape(-1, 1), np.arange(20).reshape(-1, 1)
    slope = LinearRegression().fit(x, y).coef_[0].item()
    
    tp = (high + low + close) / 3
    mf = tp * volume
    pos = np.where(tp > tp.shift(1), mf, 0)
    neg = np.where(tp < tp.shift(1), mf, 0)
    mfi = 100 - (100 / (1 + pd.Series(pos).rolling(14).sum().iloc[-1] / pd.Series(neg).rolling(14).sum().iloc[-1]))
    return {"obv_slope": slope, "mfi": mfi, "obv_series": obv}

def analyze_trend(series):
    if series is None or len(series) < 200: return None
    series = series.dropna()
    y, x = series.values.reshape(-1, 1), np.arange(len(series)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    p_now = series.iloc[-1].item()
    p_2w = model.predict([[len(y)+10]])[0].item()
    p_1m = model.predict([[len(y)+22]])[0].item()
    p_3m = model.predict([[len(y)+66]])[0].item()
    
    ema20 = series.ewm(span=20).mean().iloc[-1].item()
    sma200 = series.rolling(200).mean().iloc[-1].item()
    
    status = "🛡️ 區間"
    if p_now < sma200: status = "🛑 熊市"
    elif p_now > ema20 and model.coef_[0].item() > 0: status = "🔥 進攻"
    elif p_now < ema20: status = "⚠️ 減弱"
    
    return {"status": status, "p_now": p_now, "p_2w": p_2w, "p_1m": p_1m, "p_3m": p_3m, "sma200": sma200}

def determine_strategy_gear(qqq_trend, vix, hyg_trend, net_liq_trend):
    if not qqq_trend: return "N/A", "No Data"
    price = qqq_trend['p_now']
    if net_liq_trend == "收縮": return "檔位 1 (QQQ)", "💧 聯準會縮表：流動性下降。"
    if hyg_trend and hyg_trend['p_now'] < hyg_trend['sma200']: return "檔位 0 (現金)", "💔 信用破裂：HYG 跌破年線。"
    if price < qqq_trend['sma200']: return "檔位 0 (現金)", "🛑 熊市：跌破年線。"
    if vix and vix > 22: return "檔位 1 (QQQ)", "🌩️ VIX 恐慌模式。"
    return "檔位 3 (TQQQ)", "🚀 完美風口：流動性充裕 + 趨勢向上。"

def parse_input(text):
    port = {}
    for line in text.strip().split('\n'):
        if ',' in line:
            parts = line.split(',')
            try: port[parts[0].strip().upper()] = float(parts[1].strip())
            except: port[parts[0].strip().upper()] = 0.0
    return port

# --- MAIN ---
def main():
    st.title("Alpha 4.0: 全域戰略旗艦版")
    st.caption("v4.0 | RRG 動能輪動 + 機構籌碼 + Rule of 40 基本面")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ 參數設定")
        fred_key = st.secrets.get("FRED_API_KEY", st.text_input("FRED API Key (選填)", type="password"))
        
        st.header("💼 資產配置")
        # [更新] 預設資產配置為 BTC 和 AMD
        default_input = """BTC-USD, 10000
AMD, 10000"""
        user_input = st.text_area("持倉清單", default_input, height=200)
        portfolio_dict = parse_input(user_input)
        tickers_list = list(portfolio_dict.keys())
        total_value = sum(portfolio_dict.values())
        st.metric("總資產估值 (Est.)", f"${total_value:,.0f}")
        if st.button("🚀 啟動全域掃描", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False):
        st.info("👈 請點擊『啟動全域掃描』。")
        return

    with st.spinner("Alpha 4.0 正在進行多維度運算..."):
        df_close, df_open, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_liquidity = fetch_fred_liquidity(fred_key)
        
        # 抓取基本面與籌碼數據
        adv_data = {t: get_advanced_info(t) for t in tickers_list}

    if df_close.empty: st.error("數據獲取失敗"); return

    # --- 1. 宏觀與流動性 ---
    st.subheader("1. 宏觀與流動性 (Macro & Liquidity)")
    vix = df_close.get('^VIX').iloc[-1] if '^VIX' in df_close else None
    hyg_trend = analyze_trend(df_close.get('HYG'))
    
    liq_status, liq_trend_val = "未知", "N/A"
    if df_liquidity is not None:
        curr, prev = df_liquidity['Net_Liquidity'].iloc[-1], df_liquidity['Net_Liquidity'].iloc[-5]
        liq_status = "擴張 (印鈔中)" if curr > prev else "收縮 (抽水中)"
        liq_trend_val = "擴張" if curr > prev else "收縮"
    
    qqq_trend = analyze_trend(df_close.get('QQQ')) # 如果沒有 QQQ 會回傳 None
    gear, reason = determine_strategy_gear(qqq_trend, vix, None, hyg_trend, liq_trend_val)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("美元淨流動性", liq_status, f"${df_liquidity['Net_Liquidity'].iloc[-1]:.2f}T" if df_liquidity is not None else "No Key")
    c2.metric("信用市場 (HYG)", "充裕" if hyg_trend and hyg_trend['p_now'] > hyg_trend['sma200'] else "枯竭")
    c3.metric("VIX", f"{vix:.2f}" if vix else "N/A")
    c4.metric("Alpha 指令", gear)
    
    if "收縮" in liq_status: st.warning(f"⚠️ {reason}")
    else: st.success(f"✅ {reason}")
    st.markdown("---")

    # --- 2. RRG 動能輪動 (New Feature) ---
    st.subheader("2. RRG 板塊輪動 (Relative Rotation Graph)")
    st.markdown("以 **SPY (S&P 500)** 為中心，觀測資金流向。X軸=相對強度 (趨勢)，Y軸=相對動能 (速度)。")
    
    rrg_df = calc_rrg_metrics(df_close, tickers_list)
    if not rrg_df.empty:
        fig_rrg = px.scatter(rrg_df, x='RS_Ratio', y='RS_Momentum', color='Quadrant', text='Ticker',
                             title="RRG 動態輪動圖", 
                             color_discrete_map={'🟢 領先 (Leading)': '#00FF7F', '🟡 轉弱 (Weakening)': '#FFFF00',
                                                 '🔴 落後 (Lagging)': '#FF4B4B', '🔵 改善 (Improving)': '#00BFFF'})
        fig_rrg.add_shape(type="line", x0=100, y0=0, x1=100, y1=200, line=dict(color="gray", width=1, dash="dash"))
        fig_rrg.add_shape(type="line", x0=0, y0=100, x1=200, y1=100, line=dict(color="gray", width=1, dash="dash"))
        fig_rrg.update_layout(xaxis_title="RS-Ratio (趨勢)", yaxis_title="RS-Momentum (動能)", height=500)
        st.plotly_chart(fig_rrg, use_container_width=True)
    st.markdown("---")

    # --- 3. 深度審計 (基本面 + 籌碼 + 技術) ---
    st.subheader("3. 深度資產審計 (Fundamental & Institutional Audit)")
    
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        ff = calc_fund_flow(df_close[ticker], df_high[ticker], df_low[ticker], df_vol[ticker])
        info = adv_data.get(ticker, {})
        
        # 三角定位
        t_atr = calc_atr_target(df_close[ticker], df_high[ticker], df_low[ticker])
        t_mc = calc_monte_carlo_target(df_close[ticker])
        t_fib = calc_fib_target(df_close[ticker])
        
        # Rule of 40 判斷
        r40 = info.get('Rule40')
        r40_badge = "✅ 通過" if r40 and r40 > 40 else ("❌ 未通過" if r40 else "N/A")
        
        with st.expander(f"📊 {ticker} - {trend['status']} | Rule of 40: {r40_badge}", expanded=True):
            k1, k2, k3 = st.columns([2, 1, 1])
            
            with k1: # 技術面
                st.markdown("#### 🎯 技術四角定位")
                c_a, c_b = st.columns(2)
                c_a.write(f"**ATR Target:** ${t_atr:.2f}" if t_atr else "-")
                c_a.write(f"**Monte Carlo:** ${t_mc:.2f}" if t_mc else "-")
                c_b.write(f"**Fibonacci:** ${t_fib:.2f}" if t_fib else "-")
                c_b.write(f"**Analyst:** ${info.get('Target_Mean')}" if info.get('Target_Mean') else "-")
                st.plotly_chart(plot_combo_chart(ticker, df_close, df_vol, trend, ff), use_container_width=True, key=f"ff_{ticker}")

            with k2: # 籌碼面 (Institutional Radar)
                st.markdown("#### 🏦 機構籌碼")
                inst_held = info.get('Inst_Held', 0)
                st.metric("機構持股比", f"{inst_held*100:.1f}%" if inst_held else "N/A", 
                          delta="高度控盤" if inst_held and inst_held > 0.7 else "散戶多")
                st.metric("OBV 資金流", format_number(ff['obv_slope']), "吸籌" if ff['obv_slope']>0 else "出貨")
                st.caption(f"空單比例 (Short Ratio): {info.get('Short_Ratio', 0)}")

            with k3: # 基本面 (Fundamental Scan)
                st.markdown("#### 💎 基本面體質")
                st.metric("Rule of 40", f"{r40:.1f}" if r40 else "N/A", delta=r40_badge)
                st.metric("PEG 估值", f"{info.get('PEG', 0)}", delta="低估" if info.get('PEG') and info.get('PEG') < 1 else "合理/高估", delta_color="inverse")
                st.caption(f"Forward P/E: {info.get('PE', 'N/A')}")
                st.write("**三階段推演:**")
                st.caption(f"2週: ${trend['p_2w']:.2f} | 1月: ${trend['p_1m']:.2f}")

    st.markdown("---")
    
    # --- 4. 總表 ---
    st.subheader("4. 資產配置總表")
    table_data = []
    for ticker in tickers_list:
        if ticker not in df_close.columns: continue
        trend = analyze_trend(df_close[ticker])
        info = adv_data.get(ticker, {})
        ff = calc_fund_flow(df_close[ticker], df_high[ticker], df_low[ticker], df_vol[ticker])
        t_mc = calc_monte_carlo_target(df_close[ticker])
        
        weight = portfolio_dict.get(ticker, 0) / total_value if total_value > 0 else 0
        
        table_data.append({
            "代號": ticker, "權重": f"{weight:.1%}", "現價": f"${trend['p_now']:.2f}",
            "趨勢": trend['status'], "MC目標": f"${t_mc:.2f}" if t_mc else "-",
            "Rule 40": f"{info.get('Rule40', 0):.1f}" if info.get('Rule40') else "-",
            "機構持股": f"{info.get('Inst_Held', 0)*100:.0f}%" if info.get('Inst_Held') else "-",
            "OBV": "流入" if ff['obv_slope']>0 else "流出"
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    # --- D. 白皮書 ---
    st.header("5. 量化模型白皮書 (Alpha 4.0)")
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("### A. RRG 動能輪動\n計算相對於 SPY 的強度與動能。\n* **領先 (綠):** 強度>100, 動能>100 (最強)\n* **轉弱 (黃):** 強度>100, 動能<100 (獲利了結)")
        with c2:
            st.info("### B. 機構籌碼雷達\n結合 **機構持股比** 與 **OBV**。\n* 機構持股 > 70% 代表籌碼鎖定。\n* OBV 向上代表聰明錢進場。")
        with c3:
            st.info("### C. Rule of 40 (SaaS)\n針對科技成長股的黃金法則。\n$$R_{40} = \\text{Revenue Growth} + \\text{Profit Margin}$$\n若 > 40 為優質公司。")

if __name__ == "__main__":
    main()