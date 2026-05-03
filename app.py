import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import time
from datetime import datetime, timedelta, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="BTC Regime Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
body, .stApp { background-color: #0d1117; color: #e6edf3; }
.metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 16px 20px; margin: 6px 0;
}
.metric-label { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 28px; font-weight: 700; margin-top: 4px; }
.regime-badge {
    display: inline-block; padding: 4px 14px;
    border-radius: 20px; font-size: 13px; font-weight: 600;
}
.section-title {
    color: #8b949e; font-size: 11px; text-transform: uppercase;
    letter-spacing: 1.5px; margin: 20px 0 8px 0;
    border-bottom: 1px solid #21262d; padding-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
BG        = 'https://bitcoin-data.com/v1'
FRED_KEY  = '92f900ea16327ad7cf971aa8ccc0172b'
FRED_BASE = 'https://api.stlouisfed.org/fred/series/observations'
ROUND_TRIP = 0.001
MAX_SHORT  = 0.30

REGIME_COLORS = {
    'EXPANSION':  '#00d4aa',
    'STABLE':     '#4dabf7',
    'TIGHTENING': '#ff6b6b',
    'BLACK_SWAN': '#cc0000',
}
REGIME_WEIGHTS = {
    'EXPANSION':  {'MVRV_Z':25,'NUPL':20,'FnG':15,'SOPR':10,'VIX':5,'DXY':5,'MOM4W':10,'MOM12W':10},
    'STABLE':     {'MVRV_Z':20,'NUPL':15,'FnG':15,'SOPR':10,'VIX':10,'DXY':10,'MOM4W':10,'MOM12W':10},
    'TIGHTENING': {'MVRV_Z':10,'NUPL':10,'FnG':10,'SOPR':5,'VIX':20,'DXY':20,'MOM4W':10,'MOM12W':15},
    'BLACK_SWAN': {'MVRV_Z':5,'NUPL':5,'FnG':5,'SOPR':5,'VIX':30,'DXY':20,'MOM4W':15,'MOM12W':15},
}
SPECS = [
    ('MVRV_Z',-0.5,7.0,True),('NUPL',-0.4,0.75,True),
    ('FnG',0,100,True),('SOPR',0.90,1.05,True),
    ('VIX',10,45,False),('DXY',90,115,True),
    ('MOM4W',-40,60,True),('MOM12W',-60,100,True),
]
PHASE_PARAMS = {
    'BULL_EARLY': {'trailing_stop':0.12,'stop_loss':0.20,'min_hold':21,'short_hold_max':10},
    'BULL_LATE':  {'trailing_stop':0.08,'stop_loss':0.15,'min_hold':14,'short_hold_max':5},
    'BEAR':       {'trailing_stop':0.06,'stop_loss':0.12,'min_hold':7,'short_hold_max':30},
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _get(url, params=None, timeout=12, retries=2):
    for a in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(15); continue
            return r.json() if r.status_code == 200 else None
        except:
            if a < retries - 1: time.sleep(2)
    return None

def _score(val, lo, hi, inverse=False):
    try:
        v = float(val)
        if not np.isfinite(v): return np.nan
        s = max(0.0, min(100.0, (v - lo) / (hi - lo) * 100.0))
        return 100.0 - s if inverse else s
    except:
        return np.nan

def classify_regime(row):
    m2c    = row.get('m2_chg_3m', 0) or 0
    dxy_vs = row.get('dxy_vs_sma50', 0) or 0
    vix    = row.get('VIX', 20) or 20
    hy_c   = row.get('hy_chg_20d', 0) or 0
    vix_vs = row.get('vix_vs_sma50', 0) or 0

    if vix > 35 or vix_vs > 0.5 or hy_c > 100:
        return 'BLACK_SWAN'
    bull, bear = 0, 0
    if m2c > 0.3:     bull += 2
    elif m2c > 0:     bull += 1
    elif m2c < -0.5:  bear += 2
    elif m2c < 0:     bear += 1
    if dxy_vs < -0.01: bull += 1
    elif dxy_vs > 0.02: bear += 2
    if vix < 18:      bull += 1
    elif vix > 25:    bear += 1
    if hy_c < -10:    bull += 1
    elif hy_c > 30:   bear += 2
    net = bull - bear
    if net >= 3:  return 'EXPANSION'
    if net <= -3: return 'TIGHTENING'
    return 'STABLE'

def compute_score(row, regime):
    weights = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS['STABLE'])
    ws, wt = 0.0, 0.0
    for key, lo, hi, inv in SPECS:
        val = row.get(key)
        if val is not None and pd.notna(val):
            sc = _score(val, lo, hi, inv)
            if pd.notna(sc):
                w = weights.get(key, 0)
                ws += sc * w; wt += w
    return ws / wt if wt > 0 else 50.0

def classify_phase(regime, mvrv_z, score, mom12w):
    if regime in ['TIGHTENING', 'BLACK_SWAN']: return 'BEAR'
    if mvrv_z > 4.5 or (score < 32 and mom12w > 40): return 'BULL_LATE'
    return 'BULL_EARLY'

def target_position(score, regime, phase='BULL_EARLY', above_200=True, dd_90d=0):
    ENTRY = {'EXPANSION':55,'STABLE':35,'TIGHTENING':999,'BLACK_SWAN':80}
    EXIT  = {'EXPANSION':10,'STABLE':35,'TIGHTENING':999,'BLACK_SWAN':50}
    entry = ENTRY.get(regime, 55)
    exit_ = EXIT.get(regime, 35)
    if regime == 'TIGHTENING': return -0.10 if score < 30 else 0.0
    if regime == 'BLACK_SWAN': return 0.30 if score >= 80 else -0.22
    if regime == 'EXPANSION':
        return min(1.0, 0.98 * max(0.85, score / 100)) if score >= 30 else 0.88
    if regime == 'STABLE':
        danger = (not above_200) and (dd_90d < -0.20)
        trending_up = above_200 and (score >= 35)
        if trending_up: return 0.75
        if score >= entry:
            base = 0.68
            return min(0.35, base * score / 100) if danger else max(0.58, base * score / 100)
        elif score >= exit_: return 0.10 if danger else 0.29
        return 0.0
    return 0.0

def zone_label(s):
    if s >= 80: return ('CAPITULATION', '#00d4aa')
    if s >= 65: return ('FEAR', '#4dabf7')
    if s >= 55: return ('ANXIETY', '#74c0fc')
    if s >= 45: return ('NEUTRAL', '#8b949e')
    if s >= 35: return ('OPTIMISM', '#ffa94d')
    if s >= 20: return ('GREED', '#ff8c42')
    return ('EUPHORIA', '#ff6b6b')

# ── Data (cached 30 min) ───────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def load_master():
    tickers = {'BTC-USD':'BTC','DX-Y.NYB':'DXY','^IXIC':'Nasdaq','GC=F':'Gold'}
    raw = yf.download(list(tickers.keys()), period='max', progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    close = close[[c for c in tickers.keys() if c in close.columns]]
    close.columns = [tickers[c] for c in close.columns]
    master = close.reset_index()
    master.columns = ['date'] + list(master.columns[1:])
    master['date'] = pd.to_datetime(master['date']).dt.tz_localize(None)
    master = master[master['date'] >= '2019-01-01'].copy()

    for ep, (col, field) in {
        'mvrv-zscore': ('MVRV_Z','mvrvZscore'),
        'mvrv':        ('MVRV','mvrv'),
        'nupl':        ('NUPL','nupl'),
        'sopr':        ('SOPR','sopr'),
    }.items():
        data = _get(f'{BG}/{ep}')
        if data and isinstance(data, list) and len(data) > 0:
            temp = pd.DataFrame(data)
            if 'd' in temp.columns and field in temp.columns:
                temp = temp[['d', field]].rename(columns={'d':'date', field:col})
                temp['date'] = pd.to_datetime(temp['date'])
                temp[col] = pd.to_numeric(temp[col], errors='coerce')
                master = master.merge(temp.dropna(), on='date', how='left')
        time.sleep(0.3)

    fg = _get('https://api.alternative.me/fng/', params={'limit':0,'format':'json'})
    if fg and 'data' in fg:
        fdf = pd.DataFrame(fg['data'])
        fdf['date'] = pd.to_datetime(fdf['timestamp'].astype(int), unit='s').dt.normalize()
        fdf['FnG'] = fdf['value'].astype(int)
        master = master.merge(fdf[['date','FnG']], on='date', how='left')

    for sid, col in [('VIXCLS','VIX'),('M2SL','M2'),('BAMLH0A0HYM2','HY_Spread')]:
        data = _get(FRED_BASE, params={'series_id':sid,'api_key':FRED_KEY,
            'file_type':'json','observation_start':'2019-01-01','sort_order':'asc'})
        if data and 'observations' in data:
            df = pd.DataFrame(data['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df[col] = pd.to_numeric(df['value'], errors='coerce')
            master = master.merge(df[['date', col]].dropna(), on='date', how='left')

    master = master.sort_values('date').reset_index(drop=True)
    for col in ['MVRV_Z','MVRV','NUPL','SOPR','FnG']:
        if col in master.columns: master[col] = master[col].ffill(limit=3)
    for col in ['VIX','M2','HY_Spread']:
        if col in master.columns: master[col] = master[col].ffill(limit=14)
    master['MOM4W']        = master['BTC'].pct_change(20) * 100
    master['MOM12W']       = master['BTC'].pct_change(60) * 100
    master['m2_chg_3m']    = master['M2'].pct_change(63) * 100 if 'M2' in master.columns else 0
    master['dxy_vs_sma50'] = master['DXY'] / master['DXY'].rolling(50).mean() - 1
    master['hy_chg_20d']   = (master['HY_Spread'] - master['HY_Spread'].shift(20)) if 'HY_Spread' in master.columns else 0
    master['vix_vs_sma50'] = (master['VIX'] / master['VIX'].rolling(50).mean() - 1) if 'VIX' in master.columns else 0
    master['btc_200d']     = master['BTC'].rolling(200).mean()
    master['btc_90d_high'] = master['BTC'].rolling(90).max()
    master['btc_dd_90d']   = master['BTC'] / master['btc_90d_high'] - 1
    master['m2_chg_3m']    = master['m2_chg_3m'].fillna(0)
    master['hy_chg_20d']   = master['hy_chg_20d'].fillna(0)
    master['vix_vs_sma50'] = master['vix_vs_sma50'].fillna(0)
    master['regime'] = master.apply(lambda r: classify_regime(r.to_dict()), axis=1)
    master['score']  = master.apply(lambda r: compute_score(r.to_dict(), r['regime']), axis=1)
    master['phase']  = master.apply(
        lambda r: classify_phase(r['regime'], r.get('MVRV_Z') or 1, r['score'], r.get('MOM12W') or 0), axis=1)
    return master[master['BTC'].notna() & master['score'].notna()].copy()

@st.cache_data(ttl=1800, show_spinner=False)
def run_backtest(_master):
    bt = _master.copy().reset_index(drop=True)
    equity = 1.0; position = 0.0; peak_eq = 1.0
    entry_price = 0.0; peak_since_entry = 0.0
    hold_days = 0; in_position = False
    eq_series = []

    stop_cooldown = 0  # days since stop_loss fired

    for i in range(1, len(bt)):
        row  = bt.iloc[i]
        prev = bt.iloc[i - 1]
        btc_ret   = row['BTC'] / prev['BTC'] - 1
        score     = row['score']
        regime    = row['regime']
        phase     = row.get('phase', 'BULL_EARLY')
        above_200 = pd.notna(row.get('btc_200d')) and row['BTC'] > row['btc_200d']
        dd_90     = row.get('btc_dd_90d', 0) if pd.notna(row.get('btc_dd_90d')) else 0

        pp = PHASE_PARAMS.get(phase, PHASE_PARAMS['BULL_EARLY'])
        trailing_stop = pp['trailing_stop']
        stop_loss     = pp['stop_loss']
        min_hold      = pp['min_hold']

        tgt = target_position(score, regime, phase, above_200, dd_90)

        # Trailing stop (per-trade)
        if position > 0 and entry_price > 0:
            peak_since_entry = max(peak_since_entry, row['BTC'])
            if row['BTC'] < peak_since_entry * (1 - trailing_stop):
                tgt = 0.0

        # Portfolio stop-loss: fire once, then cooldown 60 days before re-entry
        if stop_cooldown > 0:
            stop_cooldown -= 1
            tgt = 0.0  # no new positions during cooldown
        elif equity < peak_eq * (1 - stop_loss):
            tgt = 0.0
            stop_cooldown = 30  # reset: wait 30 days

        if in_position and hold_days < min_hold and abs(tgt - position) < 0.3:
            tgt = position

        if abs(tgt - position) > 0.05:
            cost = abs(tgt - position) * ROUND_TRIP
            equity *= (1 - cost)
            if tgt > 0 and position <= 0:
                entry_price = row['BTC']
                peak_since_entry = row['BTC']
                in_position = True; hold_days = 0
            elif tgt == 0:
                in_position = False; hold_days = 0
            position = tgt

        equity *= (1 + position * btc_ret)
        peak_eq = max(peak_eq, equity)
        if in_position: hold_days += 1

        eq_series.append({
            'date': row['date'], 'equity': equity, 'position': position,
            'score': score, 'regime': regime, 'btc_px': row['BTC'],
        })

    eq_df = pd.DataFrame(eq_series)
    eq_df['btc_norm'] = eq_df['btc_px'] / eq_df['btc_px'].iloc[0]
    eq_df['dd'] = eq_df['equity'] / eq_df['equity'].cummax() - 1
    return eq_df

def annual_stats(eq_df):
    rows = []
    eq_df = eq_df.copy()
    eq_df['year'] = pd.to_datetime(eq_df['date']).dt.year
    # need cumulative equity relative to year start
    for yr, g in eq_df.groupby('year'):
        g = g.sort_values('date')
        ret  = g['equity'].iloc[-1] / g['equity'].iloc[0] - 1
        btcr = g['btc_norm'].iloc[-1] / g['btc_norm'].iloc[0] - 1
        dd   = g['dd'].min()
        rows.append({'Year':yr,'Strategy':ret,'BTC (B&H)':btcr,'Max DD':dd,'Alpha':ret-btcr})
    return pd.DataFrame(rows).set_index('Year')

# ══════════════════════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## ₿  BTC Regime-Adaptive Dashboard")
st.markdown(
    f"<span style='color:#8b949e;font-size:13px'>Updated: "
    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</span>",
    unsafe_allow_html=True)

with st.spinner("Fetching data & running backtest…"):
    master = load_master()
    eq_df  = run_backtest(master)

with st.expander("Debug", expanded=False):
    st.write("Master tail:", master[["date","BTC","score","regime","MVRV_Z","NUPL"]].tail(10))
    st.write("Equity tail:", eq_df[["date","equity","position","score","regime"]].tail(10))
    yr_eq = eq_df.copy()
    yr_eq["year"] = pd.to_datetime(yr_eq["date"]).dt.year
    for yr in [2023,2024,2025,2026]:
        g = yr_eq[yr_eq["year"]==yr]
        if len(g)>1:
            ret = g["equity"].iloc[-1]/g["equity"].iloc[0]-1
            st.write(f"{yr}: start={g['equity'].iloc[0]:.4f} end={g['equity'].iloc[-1]:.4f} ret={ret:+.2%} n={len(g)}")

latest  = master.iloc[-1].to_dict()
eq_now  = eq_df.iloc[-1]
cur_regime = latest['regime']
cur_score  = latest['score']
cur_zone, zone_color = zone_label(cur_score)
regime_color = REGIME_COLORS[cur_regime]

# Top KPIs
total_ret = eq_df['equity'].iloc[-1] - 1
this_year = datetime.now().year
ytd_df    = eq_df[pd.to_datetime(eq_df['date']).dt.year == this_year]
ytd_ret   = (ytd_df['equity'].iloc[-1] / ytd_df['equity'].iloc[0] - 1) if len(ytd_df) > 1 else 0
max_dd    = eq_df['dd'].min()
cur_pos   = eq_now['position']
btc_px    = latest['BTC']

def kpi(col, label, value, color=None):
    c = color or '#e6edf3'
    col.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value' style='color:{c}'>{value}</div>
    </div>""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "BTC Price",        f"${btc_px:,.0f}",    '#f7931a')
kpi(c2, "Strategy Total",   f"{total_ret:+.1%}",  '#00d4aa' if total_ret > 0 else '#ff6b6b')
kpi(c3, f"{this_year} YTD", f"{ytd_ret:+.1%}",   '#00d4aa' if ytd_ret > 0 else '#ff6b6b')
kpi(c4, "Max Drawdown",     f"{max_dd:.1%}",      '#ff6b6b')
kpi(c5, "Current Position", f"{cur_pos:+.0%}",   '#4dabf7')

st.markdown("")

col_left, col_right = st.columns([1, 3])

with col_left:
    st.markdown("<div class='section-title'>Current Signal</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>Regime</div>
      <div style='margin-top:8px'>
        <span class='regime-badge' style='background:{regime_color}22;color:{regime_color};border:1px solid {regime_color}44'>
          {cur_regime}
        </span>
      </div>
    </div>
    <div class='metric-card'>
      <div class='metric-label'>Score — {cur_zone}</div>
      <div class='metric-value' style='color:{zone_color}'>{cur_score:.1f}</div>
      <div style='margin-top:8px;background:#21262d;border-radius:6px;height:8px'>
        <div style='width:{cur_score:.0f}%;background:{zone_color};height:8px;border-radius:6px'></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Key Indicators</div>", unsafe_allow_html=True)
    indicators = [
        ('MVRV-Z', latest.get('MVRV_Z'), '{:.2f}'),
        ('NUPL',   latest.get('NUPL'),   '{:.3f}'),
        ('F&G',    latest.get('FnG'),    '{:.0f}'),
        ('SOPR',   latest.get('SOPR'),   '{:.4f}'),
        ('VIX',    latest.get('VIX'),    '{:.1f}'),
        ('DXY',    latest.get('DXY'),    '{:.1f}'),
        ('MOM4W',  latest.get('MOM4W'),  '{:+.1f}%'),
        ('MOM12W', latest.get('MOM12W'), '{:+.1f}%'),
    ]
    for name, val, fmt in indicators:
        if val is not None and pd.notna(val):
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;padding:5px 0;
                        border-bottom:1px solid #21262d;font-size:13px'>
              <span style='color:#8b949e'>{name}</span>
              <span style='color:#e6edf3;font-weight:600'>{fmt.format(val)}</span>
            </div>""", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='section-title'>Equity Curve vs BTC Buy & Hold</div>", unsafe_allow_html=True)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20],
                        vertical_spacing=0.04)

    eq_df['date_ts'] = pd.to_datetime(eq_df['date'])
    dates   = eq_df['date_ts'].values
    regimes = eq_df['regime'].values
    i = 0
    while i < len(dates) - 1:
        j = i + 1
        while j < len(dates) - 1 and regimes[j] == regimes[i]: j += 1
        fig.add_vrect(x0=str(dates[i])[:10], x1=str(dates[j])[:10],
                      fillcolor=REGIME_COLORS.get(regimes[i], 'gray'),
                      opacity=0.07, layer='below', line_width=0)
        i = j

    fig.add_trace(go.Scatter(
        x=eq_df['date_ts'], y=eq_df['equity'],
        name='Strategy', line=dict(color='#00d4aa', width=2.5),
        fill='tozeroy', fillcolor='rgba(0,212,170,0.06)'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=eq_df['date_ts'], y=eq_df['btc_norm'],
        name='BTC B&H', line=dict(color='#f7931a', width=1.5, dash='dot')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=eq_df['date_ts'], y=eq_df['score'],
        name='Score', line=dict(color='#4dabf7', width=1.5)
    ), row=2, col=1)
    fig.add_hline(y=65, line_dash='dash', line_color='#00d4aa', line_width=1, opacity=0.5, row=2, col=1)
    fig.add_hline(y=35, line_dash='dash', line_color='#ff6b6b', line_width=1, opacity=0.5, row=2, col=1)
    fig.add_trace(go.Scatter(
        x=eq_df['date_ts'], y=eq_df['position'],
        name='Position', line=dict(color='#ffa94d', width=1.5),
        fill='tozeroy', fillcolor='rgba(255,169,77,0.1)'
    ), row=3, col=1)

    fig.update_layout(
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(color='#8b949e', size=11),
        legend=dict(orientation='h', x=0, y=1.02, bgcolor='rgba(0,0,0,0)'),
        height=520, margin=dict(l=0, r=0, t=30, b=0),
        hovermode='x unified',
        xaxis=dict(gridcolor='#21262d'),
        xaxis2=dict(gridcolor='#21262d'),
        xaxis3=dict(gridcolor='#21262d'),
        yaxis=dict(gridcolor='#21262d', title='Equity (×)'),
        yaxis2=dict(gridcolor='#21262d', title='Score'),
        yaxis3=dict(gridcolor='#21262d', title='Position'),
    )
    st.plotly_chart(fig, use_container_width=True)

# Annual table
st.markdown("<div class='section-title'>Annual Performance</div>", unsafe_allow_html=True)
ann = annual_stats(eq_df)

def color_ret(val):
    c = '#00d4aa' if val > 0 else '#ff6b6b'
    return f'color: {c}; font-weight: 600'

styled = ann.style\
    .format({'Strategy':'{:+.1%}','BTC (B&H)':'{:+.1%}','Max DD':'{:.1%}','Alpha':'{:+.1%}'})\
    .map(color_ret, subset=['Strategy','BTC (B&H)','Alpha'])

st.dataframe(styled, use_container_width=True)

with st.expander("Drawdown Detail"):
    fig_dd = go.Figure(go.Scatter(
        x=eq_df['date_ts'], y=eq_df['dd'],
        fill='tozeroy', fillcolor='rgba(255,107,107,0.2)',
        line=dict(color='#ff6b6b', width=1.5), name='Drawdown'
    ))
    fig_dd.update_layout(
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(color='#8b949e'), height=250,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(tickformat='.0%', gridcolor='#21262d'),
        xaxis=dict(gridcolor='#21262d'),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

st.markdown(
    "<div style='color:#8b949e;font-size:11px;text-align:right'>"
    "Data: BGeometrics · FRED · yfinance · Alt.me | Cache: 30min</div>",
    unsafe_allow_html=True)
