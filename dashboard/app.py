"""
Smart Soil AI - Final Dashboard
Run:  python -m streamlit run dashboard/app.py

Tabs : Live | Last Updated | Saved / History
Modes: Manual demo (sidebar sliders)  |  ESP32 sensors (commented, enable later)
Model: LSTM, input (24 hours x [temperature, humidity, rainfall, soil_moisture]) -> next hour moisture
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "data" / "soil_moisture.csv"
MODEL_PATH = BASE / "models" / "soil_moisture_lstm.keras"
PRED_PATH = BASE / "results" / "predictions.csv"
FEATURES = ["temperature", "humidity", "rainfall", "soil_moisture"]  # exact training order
WINDOW = 24            # hours of history the LSTM needs
STEPS = 6              # forecast horizon shown on chart (hours)
RAIN_MM = 1.0          # rainfall >= this counts as "rain detected"
TANK_MIN = 10          # pump is locked out below this water level (%)

# ---- Real sensors (keep commented until manual demo is stable) --------------
# USE_SENSORS = False
# ESP32_URL = "http://192.168.1.50/data"   # ESP32 must return JSON, see read_esp32()
#
# import requests
# def read_esp32():
#     d = requests.get(ESP32_URL, timeout=2).json()
#     return {"moisture": d["soil_moisture"], "temperature": d["temperature"],
#             "humidity": d["humidity"], "rainfall": d["rainfall"],
#             "soil_temp": d.get("soil_temp", d["temperature"]),
#             "light": d.get("light", 0), "ph": d.get("ph", 7.0),
#             "water": d.get("water_level", 100)}

st.set_page_config(page_title="Smart Soil AI", page_icon="🌱", layout="wide")

# ----------------------------------------------------------------------------
# Styling (dark agricultural monitor, following the reference image)
# ----------------------------------------------------------------------------
st.markdown("""<style>
.stApp{background:radial-gradient(circle at 20% 0%,#10204a 0%,#0a1230 55%,#070c22 100%);color:#e8eefc}
[data-testid="stSidebar"]{background:#0b1436}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.2rem;max-width:1500px}
h1,h2,h3,p,label,span,div{color:#e8eefc}
.hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.hdr h1{margin:0;font-size:1.9rem}.hdr small{color:#8fa3d6}
.badge{padding:6px 16px;border-radius:20px;font-size:.8rem;font-weight:600;border:1px solid}
.b-ok{background:#0e3b35;color:#2df0b0;border-color:#2df0b0}.b-warn{background:#3b2f0e;color:#ffc94d;border-color:#ffc94d}
.card{background:linear-gradient(160deg,#16265a,#0f1b45);border:1px solid #23387a;border-radius:16px;padding:12px 14px;height:150px;margin-bottom:12px}
.ct{font-size:.72rem;letter-spacing:.06em;color:#a9b9e8;font-weight:700}
.cv{font-size:2rem;font-weight:800;line-height:1.1}.cu{font-size:.85rem;color:#a9b9e8;margin-left:3px}
.top{display:flex;justify-content:space-between;align-items:flex-start}
.ring{width:46px;height:46px;border-radius:50%;background:conic-gradient(var(--c) calc(var(--p)*1%),rgba(255,255,255,.12) 0);
-webkit-mask:radial-gradient(farthest-side,transparent 66%,#000 68%);mask:radial-gradient(farthest-side,transparent 66%,#000 68%)}
.spark{width:100%;height:38px;margin-top:6px}
.panel{background:linear-gradient(160deg,#16265a,#0f1b45);border:1px solid #23387a;border-radius:16px;padding:14px;margin-bottom:12px}
.big{font-size:2.2rem;font-weight:800;text-align:center}
.pill{display:block;text-align:center;border-radius:8px;padding:4px;font-weight:700;font-size:.8rem;margin:6px 0}
.map{display:grid;grid-template-columns:repeat(6,1fr);gap:6px}
.zone{border-radius:6px;height:38px;display:flex;align-items:center;justify-content:center;font-size:.72rem;color:#06122b;font-weight:700}
.alert{display:flex;gap:10px;font-size:.85rem;padding:5px 0;border-bottom:1px solid #1d2f68}
.dot{width:9px;height:9px;border-radius:50%;margin-top:5px;flex:none}
.banner{border-radius:14px;padding:14px 18px;margin-bottom:12px;font-weight:700;font-size:1.05rem;border:1px solid}
.n-bad{background:#4a1220;border-color:#ff4d6d}.n-warn{background:#43340c;border-color:#ffc94d}.n-ok{background:#0e3b35;border-color:#2df0b0}
.stButton>button{width:100%;border-radius:8px;font-weight:700}
</style>""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Data + model
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    if not CSV_PATH.exists():
        return None
    df = pd.read_csv(CSV_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:  # keep demo alive even if TF/model is missing
        return f"{type(e).__name__}: {e}"


df = load_data()
model = load_model()
MODEL_OK = not isinstance(model, str)

if df is None or len(df) < WINDOW:
    st.error(f"Dataset not found or too short: {CSV_PATH}. Run src/generate_data.py first.")
    st.stop()


@st.cache_data(show_spinner=False)
def run_forecast(win_bytes: bytes, steps: int):
    """Recursive multi-step forecast. Temp/humidity/rain are held at the latest values."""
    w = np.frombuffer(win_bytes, dtype="float32").reshape(WINDOW, 4).copy()
    out = []
    for _ in range(steps):
        if MODEL_OK:
            p = float(model.predict(w[None], verbose=0)[0, 0])
        else:  # fallback so the demo never breaks: gentle drift toward recent trend
            p = float(w[-1, 3] + (w[-1, 3] - w[-4, 3]) / 3 - 0.2 * w[-1, 0] / 30 + 0.5 * w[-1, 2])
        p = float(np.clip(p, 0, 100))
        out.append(p)
        nxt = w[-1].copy()
        nxt[3] = p
        w = np.vstack([w[1:], nxt])
    return out


@st.cache_data(show_spinner=False)
def model_check(n=100):
    """MAE of the loaded model on the last n windows of the CSV (sanity check for scaling/order)."""
    if not MODEL_OK or len(df) < WINDOW + n:
        return None
    v = df[FEATURES].to_numpy("float32")
    X = np.stack([v[i - WINDOW:i] for i in range(len(v) - n, len(v))])
    y = v[len(v) - n:, 3]
    p = model.predict(X, verbose=0).ravel()
    return float(np.mean(np.abs(p - y)))


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def now_str():
    return datetime.now().strftime("%H:%M:%S")


def log(level, text):
    st.session_state.alerts.insert(0, (now_str(), level, text))
    del st.session_state.alerts[30:]


def set_pump(on, why):
    s = st.session_state
    if s.pump != on:
        s.pump = on
        s.pump_since = datetime.now()
        log("info" if on else "ok", f"Pump {'ON' if on else 'OFF'} ({why})")


def start_pump():
    st.session_state.mode = "MANUAL"
    set_pump(True, "manual start")


def stop_pump():
    st.session_state.mode = "MANUAL"
    set_pump(False, "manual stop")


def init_state():
    s = st.session_state
    if "init" in s:
        return
    tail = df[FEATURES].tail(40).reset_index(drop=True)
    rng = np.random.default_rng(7)
    s.hist = {
        "moisture": list(tail["soil_moisture"]), "temperature": list(tail["temperature"]),
        "humidity": list(tail["humidity"]), "rainfall": list(tail["rainfall"]),
        "soil_temp": list(tail["temperature"] - 1.5),
        "light": list(np.clip(300 + rng.normal(0, 40, 40).cumsum(), 50, 900)),
        "ph": list(np.clip(6.5 + rng.normal(0, 0.03, 40).cumsum(), 5.5, 7.5)),
        "water": list(np.clip(60 + rng.normal(0, 1, 40).cumsum(), 30, 95)),
    }
    s.log_rows, s.alerts, s.flags = [], [], {}
    s.pump, s.pump_since, s.last_key = False, datetime.now(), None
    s.zone_offsets = np.random.default_rng(3).normal(0, 7, 24)
    s.init = True


init_state()
S = st.session_state
last = df[FEATURES].iloc[-1]

# ----------------------------------------------------------------------------
# Sidebar: manual demo controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Demo controls")
    st.radio("Pump mode", ["AUTO", "MANUAL"], key="mode", horizontal=True)
    st.subheader("Irrigation rules")
    thr = st.slider("Water-required threshold (%)", 5, 60, 30)
    target = st.slider("Stop-irrigation target (%)", thr + 5, 90, max(55, thr + 5))
    st.subheader("Sensor values")
    r = {
        "moisture": st.slider("Soil moisture (%)", 0.0, 100.0, float(round(last["soil_moisture"], 1)), 0.5),
        "temperature": st.slider("Air temperature (°C)", 0.0, 50.0, float(round(last["temperature"], 1)), 0.5),
        "humidity": st.slider("Humidity (%)", 0.0, 100.0, float(round(last["humidity"], 1)), 0.5),
        "rainfall": st.slider("Rainfall (mm)", 0.0, 50.0, float(round(last["rainfall"], 1)), 0.5),
        "soil_temp": st.slider("Soil temperature (°C)", 0.0, 50.0, float(round(last["temperature"] - 1.5, 1)), 0.5),
        "light": st.slider("Light (lux)", 0, 2000, 300, 10),
        "ph": st.slider("Soil pH", 4.0, 9.0, 6.5, 0.05),
        "water": st.slider("Water tank level (%)", 0.0, 100.0, 60.0, 1.0),
    }
    # Sensor hook: uncomment together with read_esp32() above
    # if USE_SENSORS:
    #     r.update(read_esp32())
    st.caption("Sensors are disabled. Values come from these sliders.")

raining = r["rainfall"] >= RAIN_MM
tank_ok = r["water"] >= TANK_MIN

# ----------------------------------------------------------------------------
# Forecast + decision logic
# ----------------------------------------------------------------------------
win = np.vstack([
    df[FEATURES].tail(WINDOW - 1).to_numpy("float32"),
    np.array([[r["temperature"], r["humidity"], r["rainfall"], r["moisture"]]], dtype="float32"),
])
fc = run_forecast(win.tobytes(), STEPS)
next_hr = fc[0]

if S.mode == "AUTO":
    if raining:
        set_pump(False, "auto: rain detected")
    elif not tank_ok:
        set_pump(False, "auto: tank low")
    elif r["moisture"] < thr:
        set_pump(True, "auto: low moisture")
    elif r["moisture"] >= target:
        set_pump(False, "auto: target reached")
elif S.pump and not tank_ok:
    set_pump(False, "tank empty")

low = r["moisture"] < thr
if low and raining:
    level, headline, advice = "warn", "LOW MOISTURE, RAIN DETECTED", "Irrigation held while it rains."
elif low:
    level, headline, advice = "bad", "WATER REQUIRED", f"Moisture {r['moisture']:.1f}% is below {thr}%. Start irrigation."
elif next_hr < thr:
    level, headline, advice = "warn", "MOISTURE FALLING", f"LSTM expects {next_hr:.1f}% next hour, below {thr}%. Prepare irrigation."
else:
    level, headline, advice = "ok", "SOIL HEALTHY", "No irrigation needed right now."

# edge-triggered alerts (logged once when a condition starts / clears)
def flag(name, cond, on_level, on_text, off_text=None):
    was = S.flags.get(name, False)
    if cond and not was:
        log(on_level, on_text)
        if on_level == "bad":
            st.toast(on_text, icon="💧")
    elif was and not cond and off_text:
        log("ok", off_text)
    S.flags[name] = cond

flag("low", low and not raining, "bad", f"Water required: moisture {r['moisture']:.1f}% < {thr}%", "Moisture back above threshold")
flag("soon", (not low) and next_hr < thr, "warn", f"Forecast {next_hr:.1f}% next hour, below {thr}%")
flag("tank", not tank_ok, "bad", f"Water tank low ({r['water']:.0f}%). Pump locked", "Water tank refilled")
flag("ph", not (5.5 <= r["ph"] <= 7.5), "warn", f"Soil pH out of range: {r['ph']:.2f}", "Soil pH back in range")
flag("rain", raining, "info", f"Rain detected ({r['rainfall']:.1f} mm)")

# history for sparklines / last-updated log
key = tuple(r.values()) + (S.pump,)
if key != S.last_key:
    for k, v in r.items():
        S.hist[k].append(v)
        del S.hist[k][:-60]
    S.log_rows.append({"time": now_str(), **r, "pump": "ON" if S.pump else "OFF",
                       "mode": S.mode, "status": headline, "forecast_1h": round(next_hr, 2)})
    del S.log_rows[:-200]
    S.last_key = key
    S.last_update = datetime.now()

# ----------------------------------------------------------------------------
# HTML helpers
# ----------------------------------------------------------------------------
def spark(vals, color, w=150, h=38):
    v = list(vals)[-40:]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    rg = (hi - lo) or 1
    pts = " ".join(f"{i * w / (len(v) - 1):.1f},{h - 3 - (x - lo) / rg * (h - 6):.1f}" for i, x in enumerate(v))
    return f'<svg class="spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none"><polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/></svg>'


def card(title, key, unit, color, lo, hi, fmt="{:.1f}"):
    v = S.hist[key]
    val = r[key]
    pct = float(np.clip((val - lo) / (hi - lo) * 100, 0, 100))
    arrow = "▲" if len(v) < 2 or v[-1] >= v[-2] else "▼"
    return (f'<div class="card"><div class="top"><div><div class="ct">{title}</div>'
            f'<div class="cv" style="color:{color}">{fmt.format(val)}<span class="cu">{unit}</span> '
            f'<span style="font-size:.7rem;color:{color}">{arrow}</span></div></div>'
            f'<div class="ring" style="--p:{pct};--c:{color}"></div></div>{spark(v, color)}</div>')


def zone_html():
    cells = ""
    for i in range(24):
        m = float(np.clip(r["moisture"] + S.zone_offsets[i], 0, 100))
        cells += f'<div class="zone" style="background:hsl({int(m * 1.2)},80%,50%)">{m:.0f}</div>'
    return f'<div class="map">{cells}</div>'


def alerts_html():
    if not S.alerts:
        return '<div style="color:#8fa3d6">No alerts yet.</div>'
    col = {"bad": "#ff4d6d", "warn": "#ffc94d", "ok": "#2df0b0", "info": "#4da3ff"}
    return "".join(f'<div class="alert"><div class="dot" style="background:{col[l]}"></div>'
                   f'<div style="color:#8fa3d6">{t}</div><div>{m}</div></div>' for t, l, m in S.alerts[:8])


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    f'<div class="hdr"><div><h1>🌱 Smart Soil AI</h1><small>Field Overview · Greenhouse A · LSTM forecasting</small></div>'
    f'<div><span class="badge b-warn">● MANUAL DEMO</span> &nbsp; '
    f'<small>{datetime.now().strftime("%a %d %b %Y · %H:%M:%S")}</small></div></div>', unsafe_allow_html=True)

tab_live, tab_last, tab_hist = st.tabs(["Live", "Last updated", "Saved / History"])

# ----------------------------------------------------------------------------
# Tab 1: Live
# ----------------------------------------------------------------------------
with tab_live:
    st.markdown(f'<div class="banner n-{level}">{"💧 " if level == "bad" else ""}{headline}'
                f'<div style="font-weight:400;font-size:.9rem">{advice}</div></div>', unsafe_allow_html=True)
    left, right = st.columns([3.2, 1.4])

    with left:
        row1 = st.columns(4)
        row1[0].markdown(card("SOIL MOISTURE", "moisture", "%", "#2df0b0", 0, 100), unsafe_allow_html=True)
        row1[1].markdown(card("AIR TEMPERATURE", "temperature", "°C", "#2df0b0", 0, 50), unsafe_allow_html=True)
        row1[2].markdown(card("HUMIDITY", "humidity", "%", "#2df0b0", 0, 100), unsafe_allow_html=True)
        row1[3].markdown(card("SOIL TEMPERATURE", "soil_temp", "°C", "#2df0b0", 0, 50), unsafe_allow_html=True)
        row2 = st.columns(4)
        row2[0].markdown(card("LIGHT", "light", "lux", "#ffc94d", 0, 2000, "{:.0f}"), unsafe_allow_html=True)
        row2[1].markdown(card("WATER LEVEL", "water", "%", "#2df0b0", 0, 100), unsafe_allow_html=True)
        row2[2].markdown(card("SOIL PH", "ph", "pH", "#2df0b0", 4, 9, "{:.2f}"), unsafe_allow_html=True)
        rc = "#4da3ff" if raining else "#8fa3d6"
        row2[3].markdown(f'<div class="card"><div class="ct">RAIN</div><div class="cv" style="color:{rc};margin-top:28px">'
                         f'{"YES" if raining else "NO"}</div><div class="cu">{r["rainfall"]:.1f} mm</div></div>',
                         unsafe_allow_html=True)

        st.markdown("**Soil moisture: actual vs LSTM forecast**" + ("" if MODEL_OK else "  (model not loaded, fallback estimate)"))
        idx = list(range(-(WINDOW - 1), STEPS + 1))
        actual = list(win[:, 3]) + [np.nan] * STEPS
        pred = [np.nan] * (WINDOW - 1) + [r["moisture"]] + fc
        st.line_chart(pd.DataFrame({"Actual": actual, "LSTM forecast": pred, "Threshold": [thr] * len(idx)}, index=idx),
                      color=["#4da3ff", "#2df0b0", "#ff4d6d"], height=260)
        st.caption("X axis: hours from now. Next hour forecast: %.1f%%" % next_hr)

    with right:
        tank_note = "" if tank_ok else '<span class="pill" style="background:#4a1220;color:#ff4d6d">TANK LOW - PUMP LOCKED</span>'
        run = int((datetime.now() - S.pump_since).total_seconds()) if S.pump else 0
        pill_c = {"bad": "#ff4d6d", "warn": "#ffc94d", "ok": "#2df0b0"}[level]
        st.markdown(
            f'<div class="panel"><div class="ct">IRRIGATION CONTROL</div><div class="big" style="color:{pill_c}">{r["moisture"]:.1f}%</div>'
            f'<span class="pill" style="border:1px solid {pill_c};color:{pill_c}">{"WATER" if low else "MONITOR"}</span>'
            f'<div style="text-align:center;font-size:.85rem">Pump <b>{"ON" if S.pump else "OFF"}</b> · {S.mode}'
            f'<br><span style="color:#8fa3d6">{"runtime %dm %02ds" % (run // 60, run % 60) if S.pump else "idle"}</span></div>{tank_note}</div>',
            unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        b1.button("START", on_click=start_pump)
        b2.button("STOP", on_click=stop_pump)
        st.markdown(f'<div class="panel"><div class="ct">FIELD MAP · 24 ZONES</div>{zone_html()}'
                    f'<div style="height:6px;border-radius:4px;margin-top:8px;background:linear-gradient(90deg,#f0432b,#f0e02b,#2bf05a)"></div>'
                    f'<div style="display:flex;justify-content:space-between;font-size:.7rem;color:#8fa3d6"><span>dry</span><span>wet</span></div></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="panel"><div class="ct">LIVE ALERTS</div>{alerts_html()}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Tab 2: Last updated
# ----------------------------------------------------------------------------
with tab_last:
    st.subheader("Latest reading")
    st.caption(f"Updated {S.last_update.strftime('%d %b %Y, %H:%M:%S')} · source: manual demo controls")
    c = st.columns(4)
    c[0].metric("Soil moisture", f"{r['moisture']:.1f} %")
    c[1].metric("Next hour (LSTM)", f"{next_hr:.1f} %", f"{next_hr - r['moisture']:+.1f} %")
    c[2].metric("Status", headline.title())
    c[3].metric("Pump", f"{'ON' if S.pump else 'OFF'} ({S.mode})")
    c = st.columns(4)
    c[0].metric("Air temperature", f"{r['temperature']:.1f} °C")
    c[1].metric("Humidity", f"{r['humidity']:.1f} %")
    c[2].metric("Rainfall", f"{r['rainfall']:.1f} mm")
    c[3].metric("Water tank", f"{r['water']:.0f} %")
    st.subheader("Recent readings (this session)")
    log_df = pd.DataFrame(S.log_rows).iloc[::-1]
    st.dataframe(log_df.head(30))
    st.download_button("Download session log (CSV)", log_df.to_csv(index=False), "session_log.csv", "text/csv")

# ----------------------------------------------------------------------------
# Tab 3: Saved / history
# ----------------------------------------------------------------------------
with tab_hist:
    st.subheader("Saved dataset")
    a, b, c_ = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Interval", "1 hour")
    mae = model_check()
    c_.metric("Model check (last 100 windows, MAE)", "n/a" if mae is None else f"{mae:.2f}")
    if mae is not None and mae > 10:
        st.warning("Model MAE is high on the saved data. Check that train_model.py used the same feature order and units (no /100 or scaling).")
    n = st.slider("Hours to show", 24, min(2000, len(df)), min(240, len(df)), 24)
    view = df.tail(n)
    xcol = view["timestamp"] if "timestamp" in view.columns else view.index
    st.line_chart(view[FEATURES].set_index(xcol), height=300)
    st.subheader("Model predictions")
    if PRED_PATH.exists():
        pdf = pd.read_csv(PRED_PATH)
        num = pdf.select_dtypes("number")
        st.line_chart(num.tail(300), height=260)
        st.dataframe(pdf.tail(50))
    else:
        st.info("results/predictions.csv not found. Run src/evaluate_model.py.")
    imgs = [BASE / "results" / f for f in ("actual_vs_predicted.png", "prediction_error.png", "correlation.png")]
    imgs = [p for p in imgs if p.exists()]
    if imgs:
        st.subheader("Training plots")
        for col, p in zip(st.columns(len(imgs)), imgs):
            col.image(str(p), caption=p.stem.replace("_", " "))