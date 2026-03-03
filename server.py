# """
# PEA Signal — Backend de données réelles
#
# Installe les dépendances :
# pip install fastapi uvicorn yfinance requests pandas
#
# Lance le serveur :
# python server.py
#
# L'API écoute sur http://localhost:8000
# Le dashboard HTML se connecte automatiquement à cette adresse.
# """

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime, timedelta
import math

app = FastAPI(title="PEA Signal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Univers PEA ──────────────────────────────────────────────────────────────

WATCHLIST = [
    {"ticker": "MC.PA",      "name": "LVMH",               "sector": "Luxe",              "exchange": "EPA"},
    {"ticker": "AIR.PA",     "name": "Airbus",              "sector": "Aéronautique",      "exchange": "EPA"},
    {"ticker": "ASML.AS",    "name": "ASML Holding",        "sector": "Semi-conducteurs",  "exchange": "AEX"},
    {"ticker": "SAN.PA",     "name": "Sanofi",              "sector": "Santé",             "exchange": "EPA"},
    {"ticker": "TTE.PA",     "name": "TotalEnergies",       "sector": "Énergie",           "exchange": "EPA"},
    {"ticker": "BNP.PA",     "name": "BNP Paribas",         "sector": "Banques",           "exchange": "EPA"},
    {"ticker": "SAF.PA",     "name": "Safran",              "sector": "Aéronautique",      "exchange": "EPA"},
    {"ticker": "DG.PA",      "name": "Vinci",               "sector": "Infrastructure",    "exchange": "EPA"},
    {"ticker": "SIE.DE",     "name": "Siemens",             "sector": "Industriels",       "exchange": "XETRA"},
    {"ticker": "ALV.DE",     "name": "Allianz",             "sector": "Assurance",         "exchange": "XETRA"},
    {"ticker": "NOVO-B.CO",  "name": "Novo Nordisk",        "sector": "Pharma",            "exchange": "CSE"},
    {"ticker": "RMS.PA",     "name": "Hermès",              "sector": "Luxe",              "exchange": "EPA"},
    {"ticker": "OR.PA",      "name": "L'Oréal",             "sector": "Cosmétiques",       "exchange": "EPA"},
    {"ticker": "CS.PA",      "name": "AXA",                 "sector": "Assurance",         "exchange": "EPA"},
    {"ticker": "KER.PA",     "name": "Kering",              "sector": "Luxe",              "exchange": "EPA"},
]

INDICES = {
    "es50":   "^STOXX50E",
    "cac":    "^FCHI",
    "dax":    "^GDAXI",
    "eurusd": "EURUSD=X",
    "bund":   "^TNX",   # proxy US10Y — Bund non dispo yfinance, on indique la source
}

SECTOR_ETFS = {
    "Technologie":   "DXGE.DE",
    "Aéronautique":  "AIR.PA",
    "Banques":       "SX7P.DE",
    "Industriels":   "SIE.DE",
    "Luxe":          "MC.PA",
    "Énergie":       "TTE.PA",
    "Santé":         "SAN.PA",
    "Assurance":     "ALV.DE",
    "Immobilier":    "URW.AS",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def safe_float(v, decimals=2):
    try:
        f = float(v)
        return round(f, decimals) if not math.isnan(f) else None
    except:
        return None

def pct_change(series: pd.Series, periods: int) -> float | None:
    if len(series) < periods + 1:
        return None
    v_now = series.iloc[-1]
    v_old = series.iloc[-periods]
    if v_old == 0:
        return None
    return round((v_now / v_old - 1) * 100, 2)

def weinstein_stage(hist: pd.DataFrame) -> int:
    """
    Heuristique stage Weinstein sur MA30 hebdo.
    Stage 1: price autour MA30, flat
    Stage 2: price > MA30 montante
    Stage 3: price autour MA30, plate après hausse
    Stage 4: price < MA30 descendante
    """
    if len(hist) < 35:
        return 1
    close = hist["Close"]
    ma30 = close.rolling(30).mean()
    last = close.iloc[-1]
    ma_last = ma30.iloc[-1]
    ma_slope = ma30.iloc[-1] - ma30.iloc[-5]   # pente sur 5 semaines
    above = last > ma_last

    if above and ma_slope > 0:
        return 2
    if above and ma_slope <= 0:
        return 3
    if not above and ma_slope < 0:
        return 4
    return 1  # sous MA mais pente neutre → base

def volume_trend(hist: pd.DataFrame) -> str:
    if len(hist) < 20:
        return "neutre"
    vol = hist["Volume"]
    recent = vol.iloc[-4:].mean()
    older  = vol.iloc[-20:-4].mean()
    if older == 0:
        return "neutre"
    ratio = recent / older
    # check si le volume monte avec le prix ou contre
    price_up = hist["Close"].iloc[-1] > hist["Close"].iloc[-4]
    if ratio > 1.3 and price_up:
        return "accumulation"
    if ratio > 1.5:
        return "fort"
    if ratio < 0.7 and not price_up:
        return "distribution"
    return "neutre"

def relative_strength(ticker_hist: pd.DataFrame, bench_hist: pd.DataFrame, weeks=13) -> int:
    """RS score 0-100 par rapport au benchmark (STOXX50)."""
    try:
        t_ret = pct_change(ticker_hist["Close"], weeks)
        b_ret = pct_change(bench_hist["Close"], weeks)
        if t_ret is None or b_ret is None:
            return 50
        diff = t_ret - b_ret
        # Normalise dans [0,100] centré sur 50
        score = 50 + diff * 2
        return max(0, min(100, round(score)))
    except:
        return 50

def conviction_score(stage, rs, vol, chg1m) -> int:
    s = 0
    if stage == 2: s += 2
    elif stage == 1: s += 1
    if rs >= 75: s += 1
    if rs >= 85: s += 1
    if vol in ("accumulation", "fort"): s += 1
    if chg1m and chg1m >= 5: s += 0.5
    return min(5, round(s))

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    return FileResponse("Index.html")

@app.get("/macro")
def get_macro():
    tickers = list(INDICES.values())
    data = yf.download(tickers, period="5d", interval="1d", progress=False, auto_adjust=True)
    result = {}
    for key, sym in INDICES.items():
        try:
            close = data["Close"][sym].dropna()
            last  = safe_float(close.iloc[-1])
            prev  = safe_float(close.iloc[-2]) if len(close) >= 2 else last
            chg   = round((last - prev) / prev * 100, 2) if prev else 0
            # Format affiché
            if key == "eurusd":
                display = f"{last:.4f}"
            elif key == "bund":
                display = f"{last:.2f}%"
            elif last and last > 1000:
                display = f"{last:,.0f}".replace(",", " ")
            else:
                display = f"{last:.2f}"
            result[key] = {"val": display, "chg": chg, "raw": last}
        except Exception as e:
            result[key] = {"val": "—", "chg": 0, "raw": None}

    # Déduction régime simple
    stoxx_chg = result.get("es50", {}).get("chg", 0) or 0
    if stoxx_chg >= 0.5:
        regime = "bull"
    elif stoxx_chg <= -0.5:
        regime = "bear"
    else:
        regime = "neutral"

    result["regime"] = regime
    result["timestamp"] = datetime.now().strftime("%H:%M")
    return result

@app.get("/watchlist")
def get_watchlist():
    bench = yf.download("^STOXX50E", period="6mo", interval="1wk", progress=False, auto_adjust=True)

    results = []
    for item in WATCHLIST:
        t = item["ticker"]
        try:
            hist = yf.download(t, period="2y", interval="1wk", progress=False, auto_adjust=True)
            if hist.empty:
                continue

            close = hist["Close"]
            last_price = safe_float(close.iloc[-1])
            if last_price is None:
                continue

            chg1w = pct_change(close, 1)
            chg1m = pct_change(close, 4)
            hi52  = safe_float(close.iloc[-52:].max()) if len(close) >= 52 else safe_float(close.max())
            lo52  = safe_float(close.iloc[-52:].min()) if len(close) >= 52 else safe_float(close.min())

            stage = weinstein_stage(hist)
            vol   = volume_trend(hist)
            rs    = relative_strength(hist, bench)

            # Market cap
            info = yf.Ticker(t).fast_info
            mc_raw = getattr(info, "market_cap", None)
            if mc_raw and mc_raw > 1e12:
                mktcap = f"{mc_raw/1e12:.1f}T"
            elif mc_raw and mc_raw > 1e9:
                mktcap = f"{mc_raw/1e9:.0f}B"
            else:
                mktcap = "—"

            score = conviction_score(stage, rs, vol, chg1m)

            results.append({
                **item,
                "price":  last_price,
                "chg1w":  chg1w,
                "chg1m":  chg1m,
                "hi52":   hi52,
                "lo52":   lo52,
                "stage":  stage,
                "vol":    vol,
                "rs":     rs,
                "score":  score,
                "mktcap": mktcap,
            })
        except Exception as e:
            print(f"Erreur {t}: {e}")
            continue

    return results

@app.get("/chart/{ticker}")
def get_chart(ticker: str, period: str = "6mo"):
    hist = yf.download(ticker, period=period, interval="1wk", progress=False, auto_adjust=True)
    if hist.empty:
        return {"error": "no data"}

    close = hist["Close"]
    volume = hist["Volume"]

    # MA20 hebdo
    ma20 = close.rolling(20).mean()

    labels  = [d.strftime("%d/%m") for d in hist.index]
    prices  = [safe_float(v) for v in close]
    volumes = [int(v) if not math.isnan(v) else 0 for v in volume]
    ma20_v  = [safe_float(v) for v in ma20]

    return {
        "labels":  labels,
        "prices":  prices,
        "volumes": volumes,
        "ma20":    ma20_v,
    }

@app.get("/sectors")
def get_sectors():
    results = []
    period_weeks = 4
    for name, ticker in SECTOR_ETFS.items():
        try:
            hist = yf.download(ticker, period="3mo", interval="1wk", progress=False, auto_adjust=True)
            perf = pct_change(hist["Close"], period_weeks) if not hist.empty else None
            results.append({"name": name, "perf": perf or 0.0})
        except:
            results.append({"name": name, "perf": 0.0})

    results.sort(key=lambda x: x["perf"], reverse=True)
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
