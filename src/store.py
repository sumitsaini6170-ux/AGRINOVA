
"""SQLite storage shared by backend/server.py (ESP32 data in) and dashboard/app.py."""
import sqlite3
import time
from contextlib import contextmanager
 
import pandas as pd
 
from src.config import DB_PATH, FLOW_LPM
 
COLS = ["moisture", "temperature", "humidity", "rainfall", "soil_temp", "light", "ph", "water"]
 
 
@contextmanager
def _c():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=5)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()
 
 
def init():
    with _c() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS readings(ts REAL, moisture REAL, temperature REAL, humidity REAL, rainfall REAL,
            soil_temp REAL, light REAL, ph REAL, water REAL, source TEXT);
        CREATE TABLE IF NOT EXISTS events(ts REAL, level TEXT, text TEXT);
        CREATE TABLE IF NOT EXISTS pump_runs(start REAL, stop REAL, litres REAL, mode TEXT);
        CREATE TABLE IF NOT EXISTS command(id INTEGER PRIMARY KEY, pump INTEGER, ts REAL);
        INSERT OR IGNORE INTO command VALUES(1, 0, 0);
        """)
 
 
def insert_reading(r, source):
    with _c() as c:
        c.execute("INSERT INTO readings VALUES(?,?,?,?,?,?,?,?,?,?)", (time.time(), *[r[k] for k in COLS], source))
 
 
def latest_reading():
    with _c() as c:
        row = c.execute("SELECT * FROM readings WHERE source='esp32' ORDER BY ts DESC LIMIT 1").fetchone()
    return dict(row) if row else None
 
 
def recent(n=500):
    with _c() as c:
        df = pd.read_sql_query("SELECT * FROM readings ORDER BY ts DESC LIMIT ?", c, params=(n,))
    df["time"] = pd.to_datetime(df["ts"], unit="s")
    return df.iloc[::-1].reset_index(drop=True)
 
 
def log_event(level, text):
    with _c() as c:
        c.execute("INSERT INTO events VALUES(?,?,?)", (time.time(), level, text))
 
 
def events(n=100):
    with _c() as c:
        df = pd.read_sql_query("SELECT * FROM events ORDER BY ts DESC LIMIT ?", c, params=(n,))
    df["time"] = pd.to_datetime(df["ts"], unit="s")
    return df
 
 
def pump_start(mode):
    with _c() as c:
        c.execute("INSERT INTO pump_runs VALUES(?,NULL,NULL,?)", (time.time(), mode))
 
 
def pump_stop():
    now = time.time()
    with _c() as c:
        c.execute("UPDATE pump_runs SET stop=?, litres=(?-start)/60.0*? WHERE stop IS NULL", (now, now, FLOW_LPM))
 
 
def runs():
    with _c() as c:
        df = pd.read_sql_query("SELECT * FROM pump_runs ORDER BY start DESC", c)
    open_ = df["stop"].isna()
    df.loc[open_, "litres"] = (time.time() - df.loc[open_, "start"]) / 60.0 * FLOW_LPM
    df["started"] = pd.to_datetime(df["start"], unit="s")
    return df
 
 
def set_command(on):
    with _c() as c:
        c.execute("UPDATE command SET pump=?, ts=? WHERE id=1", (int(on), time.time()))
 
 
def get_command():
    with _c() as c:
        return int(c.execute("SELECT pump FROM command WHERE id=1").fetchone()["pump"])
 

