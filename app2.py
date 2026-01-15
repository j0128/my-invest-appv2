import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 0. 全局設定 ---
st.set_page_config(page_title="Alpha 10.0: 宗師全配版", layout="wide", page_icon="🦅")

st.markdown("""
<style>
    .metric-card {background-color: #0E1117; border: 1px solid #444; border-radius: 5px; padding: 15px; color: white;}
    .bullish {color: #00FF7F; font-weight: bold;}
    .bearish {color: #FF4B4B; font-weight: bold;}
    .neutral {color: #FFD700; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] {gap: 5px;}
    .stTabs [data-baseweb="tab"] {height: 50px; background-color: #1E1E1E; border-radius: 5px 5px 0 0; color: white;}
    .stTabs [aria-selected="true"] {background-color: #00BFFF; color: black;}
</style>
""", unsafe_allow_html=True)

# --- 1. 核心數據引擎 ---
@st.cache_data(ttl=1800)
def fetch_market_data(tickers):
    benchmarks = ['SPY', 'QQQ', '^VIX', '^TNX', 'HYG', 'GC=F', 'HG=F', 'DX-Y.NYB'] 
    all_tickers = list(set(tickers + benchmarks))
    data = {col: {} for col in ['Close', 'Open', 'High', 'Low', 'Volume']}
    
    for i, t in enumerate(all_tickers):
        try:
            df = yf.Ticker(t).history(period="2y", auto_adjust=True)
            if df.empty: continue
            data['Close'][t] = df['Close']
            data['Open'][t] = df['Open']
            data['High'][t] = df['High']
            data['Low'][t] = df['Low']
            data['Volume'][t] = df['Volume']
        except: continue
    
    try:
        return (pd.DataFrame(data['Close']).ffill(), pd.DataFrame(data['High']).ffill(), 
                pd.DataFrame(data['Low']).ffill(), pd.DataFrame(data['Volume']).ffill())
    except: return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

@st.cache_data(ttl=3600*12)
def fetch_fred_macro(api_key):
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

@st.cache_data(ttl=3600*24)
def get_advanced_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'Target_Mean': info.get('targetMeanPrice'), 
            'Forward_PE': info.get('forwardPE'),
            'Trailing_PE': info.get('trailingPE'),
            'PEG': info.get('pegRatio'),
            'Inst_Held': info.get('heldPercentInstitutions'),
            'Insider_Held': info.get('heldPercentInsiders'),
            'Short_Ratio': info.get('shortRatio'),
            'Quick_Ratio': info.get('quickRatio'),
            'Current_Ratio': info.get('currentRatio'),
            'Debt_Equity': info.get('debtToEquity'),
            'ROE': info.get('returnOnEquity'),
            'Rev_Growth': info.get('revenueGrowth'),
            'Profit_Margin': info.get('profitMargins')
        }
    except: return {}

# --- 2. 戰略運算 (AI & Targets) ---
def train_ai_model(target_ticker, df_close, df_vol, days_forecast=22):
    try:
        if target_ticker not in df_close.columns: return None
        df = pd.DataFrame(index=df_close.index)
        df['Close'] = df_close[target_ticker]
        df['Vol'] = df['Close'].pct_change().rolling(20).std()
        if '^VIX' in df_close.columns: df['VIX'] = df_close['^VIX']
        if '^TNX' in df_close.columns: df['TNX'] = df_close['^TNX']
        df['Target'] = df['Close'].shift(-days_forecast)
        df = df.dropna()
        if len(df) < 50: return None
        
        X = df.drop(columns=['Target', 'Close'])
        y = df['Target']
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)
        return model.predict(X.iloc[[-1]])[0]
    except: return None

def calc_targets_composite(ticker, df_close, df_high, df_low, df_vol, f_data, days_forecast=22):
    if ticker not in df_close.columns: return None
    c = df_close[ticker]; h = df_high[ticker]; l = df_low[ticker]
    if len(c) < 100: return None
    
    # 1. ATR (物理)
    try:
        tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        t_atr = c.iloc[-1] + (atr * np.sqrt(days_forecast))
    except: t_atr = None
    
    # 2. MC (機率)
    try:
        mu = c.pct_change().mean()
        t_mc = c.iloc[-1] * ((1 + mu)**days_forecast)
    except: t_mc = None
    
    # 3. Fib (心理)
    try:
        recent = c.iloc[-60:]
        t_fib = recent.max() + (recent.max() - recent.min()) * 0.618 
    except: t_fib = None
    
    # 4. Fund & AI
    t_fund = f_data.get('Target_Mean')
    t_ai = train_ai_model(ticker, df_close, df_vol, days_forecast)
    
    targets = [t for t in [t_atr, t_mc, t_fib, t_ai] if t is not None and not pd.isna(t)]
    t_avg = sum(targets) / len(targets) if targets else None
    
    return {"ATR": t_atr, "MC": t_mc, "Fib": t_fib, "Fund": t_fund, "AI": t_ai, "Avg": t_avg}

def run_backtest_lab(ticker, df_close, df_high, df_low, days_ago=22):
    if ticker not in df_close.columns or len(df_close) < 250: return None
    
    idx_past = len(df_close) - days_ago - 1
    p_now = df_close[ticker].iloc[-1]
    
    # 當時數據
    c_slice = df_close[ticker].iloc[:idx_past+1]
    h_slice = df_high[ticker].iloc[:idx_past+1]
    l_slice = df_low[ticker].iloc[:idx_past+1]
    
    # 重算 ATR (代表技術面)
    tr = pd.concat([h_slice-l_slice], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    past_atr = c_slice.iloc[-1] + (atr * np.sqrt(days_ago))
    
    # 重算 MC
    past_mc = c_slice.iloc[-1] * ((1 + c_slice.pct_change().mean())**days_ago)
    
    # 平均
    past_avg = (past_atr + past_mc) / 2
    err = (past_avg - p_now) / p_now
    
    return {"Past_Pred": past_avg, "Error": err, "Price_Now": p_now}

def analyze_trend_multi(series):
    if series is None or len(series) < 126: return {}
    y = series.iloc[-126:].values.reshape(-1, 1)
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    
    return {
        "p_2w": model.predict([[len(y)+10]])[0].item(),
        "p_1m": model.predict([[len(y)+22]])[0].item(),
        "p_3m": model.predict([[len(y)+66]])[0].item(),
        "p_now": series.iloc[-1],
        "sma200": series.rolling(200).mean().iloc[-1]
    }

def calc_obv(close, volume):
    if volume is None: return None
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

# --- 3. 策略與財務模組 ---
def run_traffic_light_strategy(series):
    """紅綠燈策略回測 (Traffic Light Backtest)"""
    if len(series) < 200: return None
    
    sma200 = series.rolling(200).mean()
    sma50 = series.rolling(50).mean()
    
    df = pd.DataFrame({'Close': series, 'SMA200': sma200, 'SMA50': sma50})
    df['Signal'] = 0
    # 綠燈: 價格 > 200MA (持有)
    df.loc[df['Close'] > df['SMA200'], 'Signal'] = 1
    
    df['Strategy_Ret'] = df['Close'].pct_change() * df['Signal'].shift(1)
    df['BuyHold_Ret'] = df['Close'].pct_change()
    
    cum_strat = (1 + df['Strategy_Ret']).cumprod()
    cum_bh = (1 + df['BuyHold_Ret']).cumprod()
    
    return cum_strat, cum_bh

def calc_coast_fire(current_age, retire_age, current_net_worth, monthly_saving, return_rate, inflation):
    years = retire_age - current_age
    real_rate = (1 + return_rate/100) / (1 + inflation/100) - 1
    data = []
    bal = current_net_worth
    for y in range(years + 1):
        data.append({"Age": current_age + y, "Balance": bal})
        bal = bal * (1 + real_rate) + (monthly_saving * 12)
    return bal, pd.DataFrame(data)

def calc_mortgage(amount, years, rate_pct):
    rate = rate_pct / 100 / 12
    months = years * 12
    pmt = amount * (rate * (1 + rate)**months) / ((1 + rate)**months - 1)
    return pmt

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
    with st.sidebar:
        st.header("⚙️ 設定與資產")
        fred_key = st.secrets.get("FRED_API_KEY", st.text_input("FRED API Key", type="password"))
        default_input = """BTC-USD, 10000\nAMD, 10000\nNVDA, 10000\nTLT, 5000"""
        user_input = st.text_area("持倉清單", default_input, height=150)
        portfolio_dict = parse_input(user_input)
        tickers_list = list(portfolio_dict.keys())
        total_value = sum(portfolio_dict.values())
        st.metric("總資產 (Est.)", f"${total_value:,.0f}")
        
        if st.button("🚀 啟動宗師版", type="primary"): st.session_state['run'] = True

    if not st.session_state.get('run', False): return

    with st.spinner("🦅 正在執行 Alpha 10.0 全系統運算..."):
        df_close, df_high, df_low, df_vol = fetch_market_data(tickers_list)
        df_macro = fetch_fred_macro(fred_key)
        adv_data = {t: get_advanced_info(t) for t in tickers_list}

    if df_close.empty: st.error("❌ 數據獲取失敗"); st.stop()

    # --- TABS ---
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🦅 戰略戰情", "🐋 深度籌碼", "🔍 個股體檢", 
        "🚦 策略回測", "💰 CFO 財報", "🏠 房貸目標"
    ])

    # === TAB 1: 戰略戰情室 (Strategy) ===
    with t1:
        st.subheader("1. 宏觀與預測 (Macro & Forecast)")
        
        # Macro
        vix = df_close['^VIX'].iloc[-1] if '^VIX' in df_close.columns else 0
        tnx = df_close['^TNX'].iloc[-1] if '^TNX' in df_close.columns else 0
        liq = df_macro['Net_Liquidity'].iloc[-1] if df_macro is not None else 0
        try: cg = (df_close['HG=F'].iloc[-1]/df_close['GC=F'].iloc[-1])*1000
        except: cg = 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💧 淨流動性", f"${liq:.2f}T")
        c2.metric("🌪️ VIX", f"{vix:.2f}", delta_color="inverse")
        c3.metric("⚖️ 10年殖利率", f"{tnx:.2f}%")
        c4.metric("🏭 銅金比", f"{cg:.2f}")
        
        st.divider()
        
        # Individual
        for ticker in tickers_list:
            if ticker not in df_close.columns: continue
            
            trend = analyze_trend_multi(df_close[ticker])
            info = adv_data.get(ticker, {})
            targets = calc_targets_composite(ticker, df_close, df_high, df_low, df_vol, info, 22)
            bt = run_backtest_lab(ticker, df_close, df_high, df_low, 22)
            obv = calc_obv(df_close[ticker], df_vol[ticker])
            
            t_avg = f"${targets['Avg']:.2f}" if targets and targets['Avg'] else "-"
            
            with st.expander(f"🦅 {ticker} | 綜合目標: {t_avg}", expanded=True):
                k1, k2, k3 = st.columns([2, 1, 1])
                with k1:
                    st.markdown("#### 📉 價格與 OBV 資金流")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_close.index[-126:], y=df_close[ticker].iloc[-126:], name='Price', line=dict(color='#00FF7F')))
                    if obv is not None:
                        fig.add_trace(go.Scatter(x=df_close.index[-126:], y=obv.iloc[-126:], name='OBV', line=dict(color='#FFD700', width=1), yaxis='y2'))
                    fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), yaxis2=dict(overlaying='y', side='right', showgrid=False))
                    st.plotly_chart(fig, use_container_width=True)
                with k2:
                    st.markdown("#### 🎯 四角定位 (1M)")
                    if targets:
                        st.write(f"**ATR:** ${targets['ATR']:.2f}" if targets['ATR'] else "-")
                        st.write(f"**MC:** ${targets['MC']:.2f}" if targets['MC'] else "-")
                        st.write(f"**Fib:** ${targets['Fib']:.2f}" if targets['Fib'] else "-")
                        st.write(f"**AI:** ${targets['AI']:.2f}" if targets['AI'] else "N/A")
                        st.write(f"**Fund:** ${targets['Fund']}" if targets['Fund'] else "N/A")
                    
                    if bt:
                        st.markdown("#### 🧪 回測實驗室")
                        err = bt['Error']
                        c = "green" if abs(err)<0.05 else "red"
                        st.markdown(f"1月前誤差: <span style='color:{c}'>{err:.1%}</span>", unsafe_allow_html=True)
                with k3:
                    st.markdown("#### 🔮 未來矩陣")
                    st.metric("2週", f"${trend.get('p_2w',0):.2f}")
                    st.metric("1月", f"${trend.get('p_1m',0):.2f}")
                    st.metric("3月", f"${trend.get('p_3m',0):.2f}")

    # === TAB 2: 深度籌碼 (Chips) ===
    with t2:
        st.subheader("🐋 機構與內部人籌碼")
        chip_data = []
        for t in tickers_list:
            info = adv_data.get(t, {})
            inst = info.get('Inst_Held', 0)
            insider = info.get('Insider_Held', 0)
            short = info.get('Short_Ratio', 0)
            chip_data.append({
                "代號": t,
                "機構持股": f"{inst*100:.1f}%" if inst else "-",
                "內部人持股": f"{insider*100:.1f}%" if insider else "-",
                "空單比例": short
            })
        st.dataframe(pd.DataFrame(chip_data), use_container_width=True)
        st.info("💡 邏輯：機構 > 70% 代表籌碼穩定；內部人高代表公司派有信心；空單高代表有軋空機會。")

    # === TAB 3: 個股體檢 (Health) ===
    with t3:
        st.subheader("🔍 財務體質掃描")
        health_data = []
        for t in tickers_list:
            info = adv_data.get(t, {})
            r40 = (info.get('Rev_Growth', 0) + info.get('Profit_Margin', 0)) * 100 if info.get('Rev_Growth') else 0
            
            health_data.append({
                "代號": t,
                "Rule 40 (SaaS)": f"{r40:.1f}",
                "流動比 (>1.5)": info.get('Current_Ratio'),
                "負債/權益 (<1)": info.get('Debt_Equity'),
                "ROE": f"{info.get('ROE', 0)*100:.1f}%" if info.get('ROE') else "-",
                "PEG (<1低估)": info.get('PEG')
            })
        st.dataframe(pd.DataFrame(health_data), use_container_width=True)

    # === TAB 4: 策略回測 (Backtest) ===
    with t4:
        st.subheader("🚦 紅綠燈趨勢策略回測 (Traffic Light)")
        st.caption("策略邏輯：價格 > 200日均線時買進持有；跌破時清倉轉現金。")
        
        for t in tickers_list:
            if t not in df_close.columns: continue
            strat, bh = run_traffic_light_strategy(df_close[t])
            
            if strat is not None:
                ret_strat = strat.iloc[-1] - 1
                ret_bh = bh.iloc[-1] - 1
                
                c1, c2 = st.columns(2)
                c1.metric(f"{t} 策略報酬", f"{ret_strat:.1%}", delta=f"勝過買持 {ret_strat-ret_bh:.1%}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=strat.index, y=strat, name='策略 (Trend)', line=dict(color='#00FF7F')))
                fig.add_trace(go.Scatter(x=bh.index, y=bh, name='買進持有 (Buy&Hold)', line=dict(color='gray', dash='dash')))
                fig.update_layout(title=f"{t} 累計報酬率比較", height=300)
                st.plotly_chart(fig, use_container_width=True)

    # === TAB 5: CFO 財報 (Personal) ===
    with t5:
        st.subheader("💰 個人 CFO 戰情室")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📊 收支表 (Income Statement)")
            inc_salary = st.number_input("月薪資收入", 0, 1000000, 80000)
            inc_passive = st.number_input("月被動收入", 0, 1000000, 10000)
            exp_life = st.number_input("月生活支出", 0, 1000000, 30000)
            exp_debt = st.number_input("月償債支出", 0, 1000000, 15000)
            
            net_saving = (inc_salary + inc_passive) - (exp_life + exp_debt)
            save_rate = net_saving / (inc_salary + inc_passive) if (inc_salary + inc_passive) > 0 else 0
            
            st.metric("月淨儲蓄", f"${net_saving:,.0f}")
            st.metric("儲蓄率", f"{save_rate:.1%}", delta="優秀" if save_rate > 0.3 else "需努力")

        with c2:
            st.markdown("#### 🏦 資產負債表 (Balance Sheet)")
            asset_invest = total_value
            asset_home = st.number_input("自用房產價值", 0, 100000000, 15000000)
            asset_cash = st.number_input("現金存款", 0, 10000000, 500000)
            liab_home = st.number_input("房貸餘額", 0, 100000000, 8000000)
            liab_other = st.number_input("信貸/車貸", 0, 10000000, 0)
            
            total_assets = asset_invest + asset_home + asset_cash
            total_liab = liab_home + liab_other
            net_worth = total_assets - total_liab
            debt_ratio = total_liab / total_assets if total_assets > 0 else 0
            
            st.metric("總淨值 (Net Worth)", f"${net_worth:,.0f}")
            st.metric("負債比", f"{debt_ratio:.1%}", delta="健康" if debt_ratio < 0.5 else "警戒", delta_color="inverse")

    # === TAB 6: 房貸與目標 (Mortgage) ===
    with t6:
        st.subheader("🏠 房貸與 FIRE 規劃")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 房貸試算")
            l_amt = st.number_input("貸款總額", 1000000, 50000000, 10000000)
            l_yr = st.number_input("年限", 10, 40, 30)
            l_rate = st.number_input("利率%", 1.0, 5.0, 2.2)
            pmt = calc_mortgage(l_amt, l_yr, l_rate)
            st.metric("月付金", f"${pmt:,.0f}")
            
        with c2:
            st.markdown("#### Coast FIRE")
            age = st.number_input("現齡", 20, 80, 35)
            r_age = st.number_input("退齡", 40, 90, 60)
            ret, df_fire = calc_coast_fire(age, r_age, net_worth, net_saving, 7.0, 2.5)
            st.metric(f"{r_age}歲預估資產", f"${ret:,.0f}")
            st.plotly_chart(px.area(df_fire, x='Age', y='Balance', title='資產累積'), use_container_width=True)

if __name__ == "__main__":
    main()