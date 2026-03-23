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

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import yfinance as yf
import httpx
import pandas as pd
from datetime import datetime, timedelta
import math
import os
import csv

app = FastAPI(title="PEA Signal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Universe loader (reads CSV files next to server.py) ──────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_watchlist() -> list[dict]:
    path = os.path.join(BASE_DIR, "watchlist.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]

def _load_indices() -> dict:
    path = os.path.join(BASE_DIR, "indices.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return {row["key"]: row["ticker"] for row in csv.DictReader(f)}

def _load_sector_etfs() -> dict:
    path = os.path.join(BASE_DIR, "sector_etfs.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return {row["name"]: row["ticker"] for row in csv.DictReader(f)}

def load_universe():
    global WATCHLIST, INDICES, SECTOR_ETFS
    WATCHLIST    = _load_watchlist()
    INDICES      = _load_indices()
    SECTOR_ETFS  = _load_sector_etfs()
    print(f"Universe loaded: {len(WATCHLIST)} tickers, {len(INDICES)} indices, {len(SECTOR_ETFS)} sector ETFs")

load_universe()

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
    return FileResponse("index.html")

@app.get("/health")
def health():
    """Lightweight health check for Render — never fetches external data."""
    return {"status": "ok"}

@app.post("/reload")
def reload_universe():
    """Reload watchlist/indices/sector_etfs from CSV files without restarting."""
    load_universe()
    return {"status": "reloaded", "tickers": len(WATCHLIST), "indices": len(INDICES), "sectors": len(SECTOR_ETFS)}

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
    tickers = [item["ticker"] for item in WATCHLIST]
    try:
        raw = yf.download(
            tickers + ["^STOXX50E"],
            period="1y", interval="1wk",
            progress=False, auto_adjust=True, threads=False,
        )
    except Exception as e:
        print(f"Watchlist batch download failed: {e}")
        raise HTTPException(status_code=503, detail="Yahoo Finance unavailable")

    if raw.empty:
        print("Watchlist batch download returned empty DataFrame")
        raise HTTPException(status_code=503, detail="Yahoo Finance returned no data")

    close_df  = raw["Close"]
    volume_df = raw["Volume"]
    # Guard: single-ticker edge case returns a Series
    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame()
    if isinstance(volume_df, pd.Series):
        volume_df = volume_df.to_frame()
    print("Watchlist columns:", list(close_df.columns))

    # Daily data for chg1d (5-day download)
    daily_close = pd.DataFrame()
    try:
        daily_raw = yf.download(
            tickers + ["^STOXX50E"],
            period="5d", interval="1d",
            progress=False, auto_adjust=True, threads=False,
        )
        if not daily_raw.empty:
            daily_close = daily_raw["Close"]
            if isinstance(daily_close, pd.Series):
                daily_close = daily_close.to_frame()
    except Exception as e:
        print(f"Daily download for chg1d failed (non-fatal): {e}")

    bench = pd.DataFrame()
    if "^STOXX50E" in close_df.columns:
        bench = pd.DataFrame({"Close": close_df["^STOXX50E"].dropna()})

    results = []
    for item in WATCHLIST:
        t = item["ticker"]
        try:
            if t not in close_df.columns:
                print(f"  {t}: not found in batch data")
                continue

            close  = close_df[t].dropna()
            volume = volume_df[t].dropna() if t in volume_df.columns else pd.Series(dtype=float)

            if len(close) < 5:
                continue

            hist = pd.DataFrame({"Close": close, "Volume": volume}).dropna(subset=["Close"])

            last_price = safe_float(close.iloc[-1])
            if last_price is None:
                continue

            chg1w = pct_change(close, 1)
            chg1m = pct_change(close, 4)
            chg3m = pct_change(close, 13)
            hi52  = safe_float(close.iloc[-52:].max()) if len(close) >= 52 else safe_float(close.max())
            lo52  = safe_float(close.iloc[-52:].min()) if len(close) >= 52 else safe_float(close.min())

            # 1-day change from daily data
            chg1d = None
            if not daily_close.empty and t in daily_close.columns:
                dc = daily_close[t].dropna()
                if len(dc) >= 2:
                    chg1d = round((float(dc.iloc[-1]) / float(dc.iloc[-2]) - 1) * 100, 2)

            stage = weinstein_stage(hist)
            vol   = volume_trend(hist)
            rs    = relative_strength(hist, bench)
            score = conviction_score(stage, rs, vol, chg1m)

            results.append({
                **item,
                "price":  last_price,
                "chg1d":  chg1d,
                "chg1w":  chg1w,
                "chg1m":  chg1m,
                "chg3m":  chg3m,
                "hi52":   hi52,
                "lo52":   lo52,
                "stage":  stage,
                "vol":    vol,
                "rs":     rs,
                "score":  score,
                "mktcap": "—",
            })
        except Exception as e:
            print(f"Erreur {t}: {e}")
            continue

    print(f"Watchlist returning {len(results)} rows")
    if not results:
        raise HTTPException(status_code=503, detail="No ticker data available from Yahoo Finance")
    return results

@app.get("/chart/{ticker}")
def get_chart(ticker: str, period: str = "6mo"):
    hist = yf.download(ticker, period=period, interval="1wk", progress=False, auto_adjust=True)
    if hist.empty:
        raise HTTPException(status_code=404, detail="no data")

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

@app.get("/debug")
def debug():
    """Lightweight diagnostic — open in any browser to see what yfinance returns."""
    tickers = [item["ticker"] for item in WATCHLIST]
    results = {"watchlist_tickers": tickers, "checks": []}
    try:
        raw = yf.download(
            tickers[:3],           # only first 3 to be fast
            period="1mo", interval="1wk",
            progress=False, auto_adjust=True, threads=False,
        )
        results["raw_empty"] = raw.empty
        results["raw_shape"] = list(raw.shape)
        close = raw["Close"]
        results["close_type"] = type(close).__name__
        results["close_columns"] = list(close.columns) if hasattr(close, "columns") else "Series"
        for t in tickers[:3]:
            if hasattr(close, "columns") and t in close.columns:
                s = close[t].dropna()
                results["checks"].append({"ticker": t, "ok": True, "rows": len(s), "last": round(float(s.iloc[-1]), 2) if len(s) else None})
            else:
                results["checks"].append({"ticker": t, "ok": False, "reason": "not in columns"})
    except Exception as e:
        results["error"] = str(e)
    return results

@app.get("/sectors")
def get_sectors():
    tickers = list(SECTOR_ETFS.values())
    try:
        raw = yf.download(
            tickers, period="1y", interval="1wk",
            progress=False, auto_adjust=True, threads=False,
        )
    except Exception as e:
        print(f"Sectors batch download failed: {e}")
        return []

    close_df = raw["Close"]
    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame()

    results = []
    for name, ticker in SECTOR_ETFS.items():
        try:
            if ticker not in close_df.columns:
                print(f"  Sector {name} ({ticker}): NOT FOUND")
                results.append({"name": name, "ticker": ticker,
                                "perf1w": 0.0, "perf1m": 0.0, "perf3m": 0.0,
                                "zscore": 0.0, "signal": "—"})
                continue

            close = close_df[ticker].dropna()
            print(f"  Sector {name} ({ticker}): {len(close)} weeks")

            if len(close) < 5:
                results.append({"name": name, "ticker": ticker,
                                "perf1w": 0.0, "perf1m": 0.0, "perf3m": 0.0,
                                "zscore": 0.0, "signal": "—"})
                continue

            perf1w = pct_change(close, 1)  or 0.0
            perf1m = pct_change(close, 4)  or 0.0
            perf3m = pct_change(close, 13) or 0.0

            # Z-score: how unusual is the current 4W return vs its own 1Y history?
            rolling_4w = close.pct_change(4).dropna() * 100
            if len(rolling_4w) >= 8:
                mu, sigma = rolling_4w.mean(), rolling_4w.std()
                zscore = round((rolling_4w.iloc[-1] - mu) / sigma, 2) if sigma > 0 else 0.0
            else:
                zscore = 0.0

            if zscore >= 1.5:
                signal = "momentum"
            elif zscore <= -1.5:
                signal = "oversold"
            elif abs(zscore) >= 1.0:
                signal = "attention"
            else:
                signal = "neutre"

            results.append({
                "name": name, "ticker": ticker,
                "perf1w": round(perf1w, 2), "perf1m": round(perf1m, 2), "perf3m": round(perf3m, 2),
                "zscore": zscore, "signal": signal,
            })
        except Exception as e:
            print(f"  Sector {name} ({ticker}): error — {e}")
            results.append({"name": name, "ticker": ticker,
                            "perf1w": 0.0, "perf1m": 0.0, "perf3m": 0.0,
                            "zscore": 0.0, "signal": "—"})

    # Most abnormal (highest |z-score|) first — those are the actionable signals
    results.sort(key=lambda x: abs(x["zscore"]), reverse=True)
    return results

@app.post("/analyze")
async def analyze(request: Request):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY non configurée sur le serveur."}
    body = await request.json()

    async def stream():
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
