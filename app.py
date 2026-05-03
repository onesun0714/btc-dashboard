import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import time
from datetime import datetime, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="BTC Regime Dashboard", page_icon="₿", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
body, .stApp { background-color: #0d1117; color: #e6edf3; }
.metric-card { background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 20px;margin:6px 0; }
.metric-label { color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:1px; }
.metric-value { font-size:28px;font-weight:700;margin-top:4px; }
.regime-badge { display:inline-block;padding:4px 14px;border-radius:20px;font-size:13px;font-weight:600; }
.section-title { color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1.5px;
                 margin:20px 0 8px 0;border-bottom:1px solid #21262d;padding-bottom:6px; }
</style>""", unsafe_allow_html=True)

# ── Constants (nb3 V16 원본) ───────────────────────────────────────────────────
BG        = 'https://bitcoin-data.com/v1'
FRED_KEY  = '92f900ea16327ad7cf971aa8ccc0172b'
FRED_BASE = 'https://api.stlouisfed.org/fred/series/observations'
ROUND_TRIP = 0.001
MAX_SHORT  = 0.50

REGIME_COLORS = {'EXPANSION':'#00d4aa','STABLE':'#4dabf7','TIGHTENING':'#ff6b6b','BLACK_SWAN':'#cc0000'}

# nb3 S1 PHASE_PARAMS (원본)
PHASE_PARAMS = {
    'BULL_EARLY': {'trailing_stop':0.20,'stop_loss':0.25,'min_hold':45,'short_hold_max':999},
    'BULL_LATE':  {'trailing_stop':0.08,'stop_loss':0.18,'min_hold':10,'short_hold_max':999},
    'BEAR':       {'trailing_stop':0.07,'stop_loss':0.12,'min_hold':5, 'short_hold_max':30},
}
BULL_FLOOR = {'EXPANSION':0.90,'STABLE':0.50}

ENTRY_SCORE_BY_REGIME = {'EXPANSION':55,'STABLE':35,'TIGHTENING':999,'BLACK_SWAN':80,'SIDEWAYS':999}
EXIT_SCORE_BY_REGIME  = {'EXPANSION':10,'STABLE':35,'TIGHTENING':999,'BLACK_SWAN':50,'SIDEWAYS':999}

REGIME_WEIGHTS = {
    'EXPANSION':  {'MVRV_Z':25,'NUPL':20,'FnG':15,'SOPR':10,'VIX':5,'DXY':5,'MOM4W':10,'MOM12W':10},
    'STABLE':     {'MVRV_Z':20,'NUPL':15,'FnG':15,'SOPR':10,'VIX':10,'DXY':10,'MOM4W':10,'MOM12W':10},
    'TIGHTENING': {'MVRV_Z':10,'NUPL':10,'FnG':10,'SOPR':5,'VIX':20,'DXY':20,'MOM4W':10,'MOM12W':15},
    'BLACK_SWAN': {'MVRV_Z':5,'NUPL':5,'FnG':5,'SOPR':5,'VIX':30,'DXY':20,'MOM4W':15,'MOM12W':15},
}
SPECS = [('MVRV_Z',-0.5,7.0,True),('NUPL',-0.4,0.75,True),('FnG',0,100,True),
         ('SOPR',0.90,1.05,True),('VIX',10,45,False),('DXY',90,115,True),
         ('MOM4W',-40,60,True),('MOM12W',-60,100,True)]

# ── Helpers ────────────────────────────────────────────────────────────────────
def _get(url, params=None, timeout=12, retries=2):
    for a in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429: time.sleep(15); continue
            return r.json() if r.status_code == 200 else None
        except:
            if a < retries-1: time.sleep(2)
    return None

def _score(val, lo, hi, inverse=False):
    try:
        v = float(val)
        if not np.isfinite(v): return np.nan
        s = max(0.0, min(100.0, (v-lo)/(hi-lo)*100.0))
        return 100.0-s if inverse else s
    except: return np.nan

def classify_regime(row):
    m2c    = row.get('m2_chg_3m',0) or 0
    dxy_vs = row.get('dxy_vs_sma50',0) or 0
    vix    = row.get('VIX',20) or 20
    hy_c   = row.get('hy_chg_20d',0) or 0
    vix_vs = row.get('vix_vs_sma50',0) or 0
    if vix > 35 or vix_vs > 0.5 or hy_c > 100: return 'BLACK_SWAN'
    bull, bear = 0, 0
    if m2c > 0.3:       bull += 2
    elif m2c > 0:       bull += 1
    elif m2c < -0.5:    bear += 2
    elif m2c < 0:       bear += 1
    if dxy_vs < -0.01:  bull += 1
    elif dxy_vs > 0.02: bear += 2
    if vix < 18:        bull += 1
    elif vix > 25:      bear += 1
    if hy_c < -10:      bull += 1
    elif hy_c > 30:     bear += 2
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
                ws += sc*w; wt += w
    return ws/wt if wt > 0 else 50.0

def classify_phase(row):
    mom12w = row.get('MOM12W',0) or 0
    score  = row.get('score',50) or 50
    mvrv_z = row.get('MVRV_Z',2) or 2
    regime = row.get('regime','STABLE')
    if regime in ['TIGHTENING','BLACK_SWAN'] and mom12w < -15: return 'BEAR'
    if mvrv_z > 4.5 or (score < 32 and mom12w > 40): return 'BULL_LATE'
    if regime in ['EXPANSION','STABLE'] and mom12w > -10: return 'BULL_EARLY'
    return 'BEAR'

def target_v16(score, regime, impulse=0, wh_div=False,
               above_200=True, dd_90d=0, phase='BULL_EARLY',
               whale_dist=False, etf_bull=False):
    regime = regime if regime in ENTRY_SCORE_BY_REGIME else 'STABLE'
    phase  = phase  if phase  in PHASE_PARAMS else 'BULL_EARLY'
    entry  = ENTRY_SCORE_BY_REGIME.get(regime, 55)
    exit_  = EXIT_SCORE_BY_REGIME.get(regime, 35)
    if regime == 'TIGHTENING': return -0.10 if score < 30 else 0.0
    if regime == 'BLACK_SWAN': return 0.30 if score >= 80 else -0.22
    if regime == 'EXPANSION':
        if whale_dist: return 0.30
        return min(1.0, 0.98*max(0.85, score/100)) if score >= 30 else 0.88
    if regime == 'STABLE':
        danger      = (not above_200) and (dd_90d < -0.20)
        trending_up = above_200 and (score >= 35)
        if whale_dist:  return 0.30
        if etf_bull:    return 0.75 if score >= exit_ else 0.55
        if trending_up: return 0.75
        if score >= entry:
            base = 0.68
            return min(0.35, base*score/100) if danger else max(0.58, base*score/100)
        elif score >= exit_: return 0.10 if danger else 0.29
        return 0.0
    return 0.0

def zone_label(s):
    if s >= 80: return ('CAPITULATION','#00d4aa')
    if s >= 65: return ('FEAR','#4dabf7')
    if s >= 55: return ('ANXIETY','#74c0fc')
    if s >= 45: return ('NEUTRAL','#8b949e')
    if s >= 35: return ('OPTIMISM','#ffa94d')
    if s >= 20: return ('GREED','#ff8c42')
    return ('EUPHORIA','#ff6b6b')

# ── Data load ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def load_master():
    tickers = {'BTC-USD':'BTC','DX-Y.NYB':'DXY','^IXIC':'Nasdaq','GC=F':'Gold'}
    raw = yf.download(list(tickers.keys()), period='max', progress=False, auto_adjust=True)
    close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close[[c for c in tickers if c in close.columns]]
    close.columns = [tickers[c] for c in close.columns]
    master = close.reset_index()
    master.columns = ['date'] + list(master.columns[1:])
    master['date'] = pd.to_datetime(master['date']).dt.tz_localize(None)
    master = master[master['date'] >= '2019-01-01'].copy()

    for ep,(col,field) in {
        'mvrv-zscore':('MVRV_Z','mvrvZscore'),
        'mvrv':('MVRV','mvrv'),
        'nupl':('NUPL','nupl'),
        'sopr':('SOPR','sopr'),
    }.items():
        data = _get(f'{BG}/{ep}')
        if data and isinstance(data,list) and len(data)>0:
            temp = pd.DataFrame(data)
            if 'd' in temp.columns and field in temp.columns:
                temp = temp[['d',field]].rename(columns={'d':'date',field:col})
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

    for sid,col in [('VIXCLS','VIX'),('M2SL','M2'),('BAMLH0A0HYM2','HY_Spread')]:
        data = _get(FRED_BASE, params={'series_id':sid,'api_key':FRED_KEY,
            'file_type':'json','observation_start':'2019-01-01','sort_order':'asc'})
        if data and 'observations' in data:
            df = pd.DataFrame(data['observations'])
            df['date'] = pd.to_datetime(df['date'])
            df[col] = pd.to_numeric(df['value'], errors='coerce')
            master = master.merge(df[['date',col]].dropna(), on='date', how='left')

    master = master.sort_values('date').reset_index(drop=True)
    for col in ['MVRV_Z','MVRV','NUPL','SOPR','FnG']:
        if col in master.columns: master[col] = master[col].ffill(limit=3)
    for col in ['VIX','M2','HY_Spread']:
        if col in master.columns: master[col] = master[col].ffill(limit=14)

    master['MOM4W']         = master['BTC'].pct_change(20)*100
    master['MOM12W']        = master['BTC'].pct_change(60)*100
    master['m2_chg_3m']     = master['M2'].pct_change(63)*100 if 'M2' in master.columns else 0
    master['dxy_vs_sma50']  = master['DXY']/master['DXY'].rolling(50).mean()-1
    master['hy_chg_20d']    = master['HY_Spread']-master['HY_Spread'].shift(20) if 'HY_Spread' in master.columns else 0
    master['vix_vs_sma50']  = master['VIX']/master['VIX'].rolling(50).mean()-1 if 'VIX' in master.columns else 0
    master['btc_200d']      = master['BTC'].rolling(200).mean()
    master['btc_90d_high']  = master['BTC'].rolling(90).max()
    master['btc_dd_90d']    = master['BTC']/master['btc_90d_high']-1
    master['btc_5d_ret']    = master['BTC'].pct_change(5)*100
    master['impulse_count'] = master['btc_5d_ret'].rolling(90).apply(lambda x:(x>15).sum(), raw=True).fillna(0)

    for col in ['m2_chg_3m','hy_chg_20d','vix_vs_sma50']:
        master[col] = master[col].fillna(0)

    master['regime']    = master.apply(lambda r: classify_regime(r.to_dict()), axis=1)
    master['score']     = master.apply(lambda r: compute_score(r.to_dict(), r['regime']), axis=1)
    master['phase']     = master.apply(classify_phase, axis=1)
    master['whale_div'] = False
    if 'SOPR' in master.columns and 'FnG' in master.columns:
        master['whale_div'] = (master['SOPR'] > 1.02) & (master['FnG'] > 75)

    return master[master['BTC'].notna() & master['score'].notna()].copy()

# ── Backtest: nb3 V16 원본 로직 ───────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def run_backtest(_master):
    bt = _master.dropna(subset=['score','BTC']).copy().reset_index(drop=True)

    equity = 1.0; position = 0.0; peak_eq = 1.0
    entry_price = 0.0; peak_since_entry = 0.0
    hold_days = 0; short_hold_days = 0; in_position = False
    last_precrisis = -999
    eq_series = []

    for i in range(1, len(bt)):
        row  = bt.iloc[i]
        prev = bt.iloc[i-1]
        btc_ret  = row['BTC'] / prev['BTC'] - 1
        score    = row['score']
        regime   = row.get('regime','STABLE')
        btc_px   = row['BTC']
        above_200 = pd.notna(row.get('btc_200d')) and btc_px > row['btc_200d']
        dd_90    = row.get('btc_dd_90d',0) if pd.notna(row.get('btc_dd_90d')) else 0
        impulse  = row.get('impulse_count',0) if pd.notna(row.get('impulse_count')) else 0
        wh_div   = bool(row.get('whale_div',False))
        phase    = row.get('phase','BULL_EARLY')

        pp            = PHASE_PARAMS.get(phase, PHASE_PARAMS['BULL_EARLY'])
        TRAILING_STOP = pp['trailing_stop']
        STOP_LOSS     = pp['stop_loss']
        MIN_HOLD_DAYS = pp['min_hold']
        SHORT_HOLD_MAX= pp.get('short_hold_max',999)

        # equity 먼저 업데이트 (nb3 원본 순서)
        equity *= (1 + position * btc_ret)
        if position != 0: hold_days += 1
        if position < 0:  short_hold_days += 1
        else:             short_hold_days = 0
        if position > 0 and btc_px > peak_since_entry: peak_since_entry = btc_px
        peak_eq = max(peak_eq, equity)

        # 숏 강제 청산
        if position < 0 and short_hold_days > SHORT_HOLD_MAX:
            equity *= (1 - abs(position)*ROUND_TRIP)
            position = 0.0; entry_price = 0.0; hold_days = 0
            short_hold_days = 0; in_position = False
            eq_series.append({'date':row['date'],'equity':equity,'position':0.0,
                              'score':score,'regime':regime,'btc_px':btc_px})
            continue

        stop = False; stype = None

        if position > 0 and entry_price > 0:
            # PRE_CRISIS 헤지
            if position > 0.40 and hold_days > last_precrisis + 30:
                vix_val      = row.get('VIX',20) if pd.notna(row.get('VIX')) else 20
                vix_prev_val = prev.get('VIX',20) if pd.notna(prev.get('VIX')) else 20
                pre_crisis = False
                if (not above_200) and dd_90 < -0.15: pre_crisis = True
                if vix_prev_val < 23 and vix_val >= 23: pre_crisis = True
                if pre_crisis:
                    desired = position * 0.5
                    equity *= (1 - (position-desired)*ROUND_TRIP)
                    position = desired
                    last_precrisis = hold_days

            # Stop 조건 (nb3 원본)
            if regime == 'BLACK_SWAN':
                stop = True; stype = 'BLACK_SWAN'
            elif regime == 'TIGHTENING':
                if hold_days >= MIN_HOLD_DAYS and btc_px < entry_price*(1-STOP_LOSS):
                    stop = True; stype = 'HARD_STOP'
                elif hold_days >= MIN_HOLD_DAYS and peak_since_entry > entry_price:
                    if btc_px < peak_since_entry*(1-TRAILING_STOP):
                        stop = True; stype = 'TRAIL_STOP'
            elif phase == 'BULL_LATE':
                if hold_days >= MIN_HOLD_DAYS and peak_since_entry > entry_price:
                    if btc_px < peak_since_entry*(1-TRAILING_STOP):
                        stop = True; stype = 'TRAIL_STOP'
            elif regime == 'STABLE':
                if hold_days >= 30 and btc_px < entry_price*(1-0.15):
                    stop = True; stype = 'STABLE_STOP'

        if stop:
            equity *= (1 - ROUND_TRIP)
            position = 0.0; entry_price = 0.0; hold_days = 0
            peak_since_entry = 0.0; short_hold_days = 0; in_position = False
            eq_series.append({'date':row['date'],'equity':equity,'position':0.0,
                              'score':score,'regime':regime,'btc_px':btc_px})
            continue

        # Rebalance: 20일(bull) / 7일(bear)마다
        rebalance_days = 20 if regime in ['EXPANSION','STABLE'] else 7

        if i % rebalance_days == 0:
            target = target_v16(score, regime, impulse, wh_div,
                                above_200, dd_90, phase,
                                whale_dist=False, etf_bull=False)

            if position > 0:
                if regime in ['EXPANSION','STABLE']:
                    desired = max(target, BULL_FLOOR[regime])
                    if regime == 'EXPANSION': desired = max(desired, 0.90)
                    if abs(desired-position) > 0.05:
                        equity *= (1 - abs(desired-position)*ROUND_TRIP)
                        position = desired
                else:
                    if target <= 0.0:
                        equity *= (1 - abs(position)*ROUND_TRIP)
                        position = 0.0; entry_price = 0.0; hold_days = 0
                        peak_since_entry = 0.0; short_hold_days = 0; in_position = False
                        if target < -0.08:
                            equity *= (1 - abs(target)*ROUND_TRIP)
                            position = target; entry_price = btc_px
                            hold_days = 0; short_hold_days = 0; in_position = True

            elif position < 0:
                if regime in ['EXPANSION','STABLE'] or score >= 55 or above_200:
                    equity *= (1 - abs(position)*ROUND_TRIP)
                    position = 0.0; entry_price = 0.0; hold_days = 0
                    short_hold_days = 0; in_position = False
                else:
                    desired = target
                    if abs(desired-position) > 0.05:
                        equity *= (1 - abs(desired-position)*ROUND_TRIP)
                        position = desired

            else:  # 포지션 없음
                if regime in ['EXPANSION','STABLE']:
                    desired = max(target, BULL_FLOOR[regime])
                    if regime == 'EXPANSION': desired = max(desired, 0.90)
                    if desired > 0.08:
                        equity *= (1 - desired*ROUND_TRIP)
                        position = desired; entry_price = btc_px
                        peak_since_entry = btc_px; hold_days = 0; in_position = True
                else:
                    if target < -0.08 and not above_200:
                        equity *= (1 - abs(target)*ROUND_TRIP)
                        position = target; entry_price = btc_px
                        hold_days = 0; short_hold_days = 0; in_position = True

        eq_series.append({'date':row['date'],'equity':equity,'position':position,
                          'score':score,'regime':regime,'btc_px':btc_px})

    eq_df = pd.DataFrame(eq_series)
    eq_df['btc_norm'] = eq_df['btc_px'] / eq_df['btc_px'].iloc[0]
    eq_df['dd'] = eq_df['equity'] / eq_df['equity'].cummax() - 1
    return eq_df

def annual_stats(eq_df):
    rows = []
    eq_df = eq_df.copy()
    eq_df['year'] = pd.to_datetime(eq_df['date']).dt.year
    for yr, g in eq_df.groupby('year'):
        g = g.sort_values('date')
        ret  = g['equity'].iloc[-1] / g['equity'].iloc[0] - 1
        btcr = g['btc_norm'].iloc[-1] / g['btc_norm'].iloc[0] - 1
        dd   = g['dd'].min()
        rows.append({'Year':yr,'Strategy':ret,'BTC (B&H)':btcr,'Max DD':dd,'Alpha':ret-btcr})
    return pd.DataFrame(rows).set_index('Year')

# ── Render ─────────────────────────────────────────────────────────────────────
st.markdown("## ₿  BTC Regime-Adaptive Dashboard")
st.markdown(f"<span style='color:#8b949e;font-size:13px'>Updated: "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</span>",
            unsafe_allow_html=True)

with st.spinner("Fetching data & running backtest…"):
    master = load_master()
    eq_df  = run_backtest(master)

latest     = master.iloc[-1].to_dict()
eq_now     = eq_df.iloc[-1]
cur_regime = latest['regime']
cur_score  = latest['score']
cur_zone, zone_color = zone_label(cur_score)
regime_color = REGIME_COLORS[cur_regime]

total_ret = eq_df['equity'].iloc[-1] - 1
this_year = datetime.now().year
ytd_df    = eq_df[pd.to_datetime(eq_df['date']).dt.year == this_year]
ytd_ret   = (ytd_df['equity'].iloc[-1]/ytd_df['equity'].iloc[0]-1) if len(ytd_df)>1 else 0
max_dd    = eq_df['dd'].min()
cur_pos   = eq_now['position']
btc_px    = latest['BTC']

def kpi(col, label, value, color=None):
    c = color or '#e6edf3'
    col.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value' style='color:{c}'>{value}</div>
    </div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5 = st.columns(5)
kpi(c1,"BTC Price",       f"${btc_px:,.0f}",   '#f7931a')
kpi(c2,"Strategy Total",  f"{total_ret:+.1%}", '#00d4aa' if total_ret>0 else '#ff6b6b')
kpi(c3,f"{this_year} YTD",f"{ytd_ret:+.1%}",  '#00d4aa' if ytd_ret>0 else '#ff6b6b')
kpi(c4,"Max Drawdown",    f"{max_dd:.1%}",     '#ff6b6b')
kpi(c5,"Current Position",f"{cur_pos:+.0%}",  '#4dabf7')

st.markdown("")
col_left, col_right = st.columns([1,3])

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
    </div>""", unsafe_allow_html=True)

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
        ('Phase',  latest.get('phase'),  '{}'),
    ]
    for name, val, fmt in indicators:
        if val is not None and (isinstance(val,str) or pd.notna(val)):
            st.markdown(f"""<div style='display:flex;justify-content:space-between;padding:5px 0;
                        border-bottom:1px solid #21262d;font-size:13px'>
              <span style='color:#8b949e'>{name}</span>
              <span style='color:#e6edf3;font-weight:600'>{fmt.format(val)}</span>
            </div>""", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='section-title'>Equity Curve vs BTC Buy & Hold</div>", unsafe_allow_html=True)
    fig = make_subplots(rows=3,cols=1,shared_xaxes=True,
                        row_heights=[0.55,0.25,0.20],vertical_spacing=0.04)
    eq_df['date_ts'] = pd.to_datetime(eq_df['date'])
    dates = eq_df['date_ts'].values; regimes = eq_df['regime'].values
    i = 0
    while i < len(dates)-1:
        j = i+1
        while j < len(dates)-1 and regimes[j]==regimes[i]: j += 1
        fig.add_vrect(x0=str(dates[i])[:10],x1=str(dates[j])[:10],
                      fillcolor=REGIME_COLORS.get(regimes[i],'gray'),
                      opacity=0.07,layer='below',line_width=0)
        i = j

    fig.add_trace(go.Scatter(x=eq_df['date_ts'],y=eq_df['equity'],name='Strategy',
        line=dict(color='#00d4aa',width=2.5),fill='tozeroy',
        fillcolor='rgba(0,212,170,0.06)'),row=1,col=1)
    fig.add_trace(go.Scatter(x=eq_df['date_ts'],y=eq_df['btc_norm'],name='BTC B&H',
        line=dict(color='#f7931a',width=1.5,dash='dot')),row=1,col=1)
    fig.add_trace(go.Scatter(x=eq_df['date_ts'],y=eq_df['score'],name='Score',
        line=dict(color='#4dabf7',width=1.5)),row=2,col=1)
    fig.add_hline(y=65,line_dash='dash',line_color='#00d4aa',line_width=1,opacity=0.5,row=2,col=1)
    fig.add_hline(y=35,line_dash='dash',line_color='#ff6b6b',line_width=1,opacity=0.5,row=2,col=1)
    fig.add_trace(go.Scatter(x=eq_df['date_ts'],y=eq_df['position'],name='Position',
        line=dict(color='#ffa94d',width=1.5),fill='tozeroy',
        fillcolor='rgba(255,169,77,0.1)'),row=3,col=1)

    fig.update_layout(paper_bgcolor='#0d1117',plot_bgcolor='#0d1117',
        font=dict(color='#8b949e',size=11),
        legend=dict(orientation='h',x=0,y=1.02,bgcolor='rgba(0,0,0,0)'),
        height=520,margin=dict(l=0,r=0,t=30,b=0),hovermode='x unified',
        xaxis=dict(gridcolor='#21262d'),xaxis2=dict(gridcolor='#21262d'),
        xaxis3=dict(gridcolor='#21262d'),
        yaxis=dict(gridcolor='#21262d',title='Equity (×)'),
        yaxis2=dict(gridcolor='#21262d',title='Score'),
        yaxis3=dict(gridcolor='#21262d',title='Position'))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<div class='section-title'>Annual Performance</div>", unsafe_allow_html=True)
ann = annual_stats(eq_df)
styled = ann.style\
    .format({'Strategy':'{:+.1%}','BTC (B&H)':'{:+.1%}','Max DD':'{:.1%}','Alpha':'{:+.1%}'})\
    .map(lambda v: f'color:{"#00d4aa" if v>0 else "#ff6b6b"};font-weight:600',
         subset=['Strategy','BTC (B&H)','Alpha'])
st.dataframe(styled, use_container_width=True)

with st.expander("Drawdown Detail"):
    fig_dd = go.Figure(go.Scatter(x=eq_df['date_ts'],y=eq_df['dd'],
        fill='tozeroy',fillcolor='rgba(255,107,107,0.2)',
        line=dict(color='#ff6b6b',width=1.5),name='Drawdown'))
    fig_dd.update_layout(paper_bgcolor='#0d1117',plot_bgcolor='#0d1117',
        font=dict(color='#8b949e'),height=250,margin=dict(l=0,r=0,t=10,b=0),
        yaxis=dict(tickformat='.0%',gridcolor='#21262d'),
        xaxis=dict(gridcolor='#21262d'))
    st.plotly_chart(fig_dd, use_container_width=True)

st.markdown("<div style='color:#8b949e;font-size:11px;text-align:right'>"
            "Data: BGeometrics · FRED · yfinance · Alt.me | Cache: 30min | Logic: nb3 V16</div>",
            unsafe_allow_html=True)
