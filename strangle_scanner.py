#!/usr/bin/env python3
"""
Daily Short Strangle Scanner
============================
Screens a fixed ETF watchlist for 35-50 DTE, 10-15 delta short strangles and
writes a dated Markdown report (optionally emailed).

SCREENING TOOL ONLY - not an order, not an investment instruction.

Usage
-----
  python strangle_scanner.py              # only runs 30-60 min after the open on NYSE trading days
  python strangle_scanner.py --force      # run now, ignoring calendar/time (for testing)
  python strangle_scanner.py --no-email   # skip email even if SMTP is configured

Data source: Yahoo Finance via yfinance (free; option quotes can be ~15 min
delayed). To use true real-time quotes, replace `get_chain()` / `spot_price()`
with your broker's API (IBKR, Tastytrade, Schwab, ...). Everything else stays.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq
from scipy.stats import norm

ET = ZoneInfo("America/New_York")
NAN = float("nan")

# =============================================================================
# CONFIG - edit these to match your own short-strangle rules
# =============================================================================
TICKERS = ["SPY", "QQQ", "IWM", "GLD", "SLV", "XLE", "XLF", "EEM", "USO", "SMH", "EWZ"]
# Add "TLT" above if you want it scanned; it will be flagged if premium is thin.

DTE_MIN, DTE_MAX, DTE_TARGET = 35, 50, 45
DELTA_MIN, DELTA_MAX, DELTA_TARGET = 0.10, 0.15, 0.12
RUN_WINDOW_MIN = (30, 60)          # minutes after the opening bell
CORR_LOOKBACK = 60                 # trading days for correlation with SPY

RULES = dict(
    ivr_trade=50,          # IV Rank needed for TRADE CANDIDATE
    ivr_watch=30,          # IV Rank needed for WATCH
    edge_trade=2.0,        # ATM IV - HV20 (vol points) needed for TRADE
    credit_pct_trade=1.0,  # strangle credit as % of underlying for TRADE
    credit_pct_low=0.6,    # below this -> "low premium" flag
    spread_ok=10.0,        # leg bid/ask width as % of mid -> OK liquidity
    spread_fair=20.0,      #                                  -> FAIR liquidity
    oi_ok=500,             # min open interest per leg -> OK
    oi_fair=100,           #                           -> FAIR
    corr_cluster=0.80,     # corr with SPY above this = same risk bucket
)

# CBOE volatility indices used as an IV-Rank proxy until the script has
# collected enough of its own daily ATM-IV history (iv_history.csv).
VOL_INDEX = {
    "SPY": "^VIX", "QQQ": "^VXN", "IWM": "^RVX", "GLD": "^GVZ", "SLV": "^VXSLV",
    "XLE": "^VXXLE", "EEM": "^VXEEM", "USO": "^OVX", "EWZ": "^VXEWZ",
}
OWN_LOG_PREFERRED = 120            # days of own IV history before preferring it

BASE = Path(__file__).resolve().parent
REPORT_DIR = BASE / "reports"
IV_LOG = BASE / "iv_history.csv"


# =============================================================================
# Calendar gate
# =============================================================================
def market_gate(now: dt.datetime) -> tuple[bool, str]:
    import pandas_market_calendars as mcal

    sched = mcal.get_calendar("NYSE").schedule(start_date=now.date(), end_date=now.date())
    if sched.empty:
        return False, f"{now:%a %Y-%m-%d} is not a US trading day (weekend/holiday). No report."
    opened = sched.iloc[0]["market_open"].tz_convert(ET)
    mins = (now - opened).total_seconds() / 60
    lo, hi = RUN_WINDOW_MIN
    if not lo <= mins <= hi:
        return False, f"Outside run window: {mins:.0f} min after open (window {lo}-{hi}). No report."
    return True, ""


# =============================================================================
# Black-Scholes helpers (no dividend term - fine for screening)
# =============================================================================
def bs_price(S, K, T, r, sig, call):
    if T <= 0 or sig <= 0:
        return max(0.0, (S - K) if call else (K - S))
    st = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / st
    d2 = d1 - st
    if call:
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sig, call):
    if T <= 0 or not np.isfinite(sig) or sig <= 0:
        return NAN
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return norm.cdf(d1) if call else norm.cdf(d1) - 1.0


def implied_vol(price, S, K, T, r, call):
    intrinsic = max(0.0, (S - K) if call else (K - S))
    if T <= 0 or price <= intrinsic + 1e-6:
        return NAN
    try:
        return brentq(lambda s: bs_price(S, K, T, r, s, call) - price, 1e-3, 5.0, maxiter=200)
    except (ValueError, RuntimeError):
        return NAN


def regt_naked(S, K, prem, call):
    """Typical Reg-T naked short option requirement per contract (USD)."""
    otm = max(0.0, K - S) if call else max(0.0, S - K)
    a = prem + 0.20 * S - otm
    b = prem + 0.10 * (S if call else K)
    return max(a, b) * 100


# =============================================================================
# Data access (swap these two for a broker API if you want real-time quotes)
# =============================================================================
def spot_price(tk: yf.Ticker) -> float:
    try:
        p = tk.fast_info["last_price"]
        if p and p > 0:
            return float(p)
    except Exception:
        pass
    h = tk.history(period="1d", interval="1m")
    return float(h["Close"].dropna().iloc[-1])


def get_chain(tk: yf.Ticker, expiry: str):
    ch = tk.option_chain(expiry)
    cols = ["strike", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
    calls = ch.calls[cols].copy()
    puts = ch.puts[cols].copy()
    for df in (calls, puts):
        df[["volume", "openInterest"]] = df[["volume", "openInterest"]].fillna(0)
        df[["bid", "ask", "impliedVolatility"]] = df[["bid", "ask", "impliedVolatility"]].fillna(0)
    return calls, puts


def risk_free() -> float:
    try:
        v = float(yf.Ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1]) / 100
        return v if 0 < v < 0.2 else 0.04
    except Exception:
        return 0.04


def download_closes(symbols, period="1y") -> pd.DataFrame:
    df = yf.download(symbols, period=period, auto_adjust=True, progress=False, threads=True)
    closes = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    if not isinstance(df.columns, pd.MultiIndex):
        closes.columns = symbols[:1]
    return closes


# =============================================================================
# Analytics
# =============================================================================
def hv(closes: pd.Series, n: int) -> float:
    r = np.log(closes / closes.shift(1)).dropna().tail(n)
    return float(r.std(ddof=1) * math.sqrt(252) * 100) if len(r) >= int(n * 0.8) else NAN


def rank_stats(series: pd.Series, current: float, src: str):
    s = series.dropna().astype(float)
    if len(s) < 60 or not np.isfinite(current):
        return None
    lo, hi = s.min(), s.max()
    ivr = 100 * (current - lo) / (hi - lo) if hi > lo else NAN
    ivp = 100 * (s < current).mean()
    return dict(ivr=float(np.clip(ivr, 0, 100)), ivp=float(ivp), src=src)


def iv_rank(sym, atm_iv_pct, vol_hist: pd.DataFrame, log_df: pd.DataFrame):
    own = log_df.loc[log_df["ticker"] == sym, "atm_iv"] if not log_df.empty else pd.Series(dtype=float)
    own_n = len(own)
    if own_n >= OWN_LOG_PREFERRED:
        return rank_stats(pd.concat([own, pd.Series([atm_iv_pct])]), atm_iv_pct, f"own log {own_n}d")
    vix = VOL_INDEX.get(sym)
    if vix and vix in vol_hist.columns and vol_hist[vix].dropna().size >= 120:
        s = vol_hist[vix].dropna()
        out = rank_stats(s, float(s.iloc[-1]), f"{vix} 1y proxy")
        if out:
            return out
    if own_n >= 60:
        return rank_stats(pd.concat([own, pd.Series([atm_iv_pct])]), atm_iv_pct, f"own log {own_n}d")
    return None


def atm_iv(calls, puts, S, T, r) -> float:
    ivs = []
    for df, call in ((calls, True), (puts, False)):
        d = df[(df.bid > 0) & (df.ask > 0)]
        if d.empty:
            continue
        row = d.iloc[(d.strike - S).abs().argsort()].iloc[0]
        iv = implied_vol((row.bid + row.ask) / 2, S, row.strike, T, r, call)
        if not np.isfinite(iv) and row.impliedVolatility > 0.01:
            iv = float(row.impliedVolatility)
        if np.isfinite(iv):
            ivs.append(iv)
    return float(np.mean(ivs)) if ivs else NAN


def pick_leg(df, S, T, r, call, fallback_iv):
    d = df[(df.bid > 0) & (df.ask > 0)]
    d = d[d.strike > S] if call else d[d.strike < S]
    if d.empty:
        return None
    d = d.copy()
    d["mid"] = (d.bid + d.ask) / 2
    d["iv_calc"] = [implied_vol(m, S, k, T, r, call) for m, k in zip(d.mid, d.strike)]
    yahoo_iv = d.impliedVolatility.where(d.impliedVolatility > 0.01)
    d["iv_use"] = d.iv_calc.fillna(yahoo_iv).fillna(fallback_iv)
    d["delta"] = [bs_delta(S, k, T, r, s, call) for k, s in zip(d.strike, d.iv_use)]
    d = d[np.isfinite(d.delta)]
    if d.empty:
        return None
    d["absd"] = d.delta.abs()
    band = d[(d.absd >= DELTA_MIN) & (d.absd <= DELTA_MAX)]
    pool = band if not band.empty else d
    row = pool.iloc[(pool.absd - DELTA_TARGET).abs().argsort()].iloc[0]
    return dict(
        strike=float(row.strike), delta=float(row.delta), bid=float(row.bid), ask=float(row.ask),
        mid=float(row.mid), oi=int(row.openInterest), vol=int(row.volume),
        spread_pct=float(100 * (row.ask - row.bid) / row.mid), in_band=not band.empty,
    )


def liquidity_grade(p, c):
    R = RULES
    worst_spread = max(p["spread_pct"], c["spread_pct"])
    min_oi = min(p["oi"], c["oi"])
    if worst_spread <= R["spread_ok"] and min_oi >= R["oi_ok"]:
        return "OK"
    if worst_spread <= R["spread_fair"] and min_oi >= R["oi_fair"]:
        return "FAIR"
    return "POOR"


def scan_ticker(sym, closes, vol_hist, r, today, log_df) -> dict:
    out = dict(ticker=sym)
    try:
        tk = yf.Ticker(sym)
        S = spot_price(tk)
        out["price"] = S
        exps = []
        for e in tk.options:
            dte = (dt.date.fromisoformat(e) - today).days
            if DTE_MIN <= dte <= DTE_MAX:
                exps.append((abs(dte - DTE_TARGET), dte, e))
        if not exps:
            out["error"] = f"no expiry in {DTE_MIN}-{DTE_MAX} DTE"
            return out
        _, dte, expiry = sorted(exps)[0]
        T = dte / 365.0
        calls, puts = get_chain(tk, expiry)
        iv = atm_iv(calls, puts, S, T, r)
        p = pick_leg(puts, S, T, r, False, iv)
        c = pick_leg(calls, S, T, r, True, iv)
        if p is None or c is None or not np.isfinite(iv):
            out["error"] = "chain incomplete (no two-sided quotes)"
            return out

        px = closes[sym].dropna()
        spy = closes["SPY"].dropna()
        rets = np.log(pd.concat([px, spy], axis=1, keys=["x", "spy"]).dropna()).diff().dropna()
        corr = float(rets.tail(CORR_LOOKBACK).corr().iloc[0, 1]) if sym != "SPY" else 1.0

        credit = p["mid"] + c["mid"]
        # Reg-T strangle: the larger side's requirement + premium of the other side
        put_req = regt_naked(S, p["strike"], p["mid"], False)
        call_req = regt_naked(S, c["strike"], c["mid"], True)
        margin = put_req + c["mid"] * 100 if put_req >= call_req else call_req + p["mid"] * 100
        hv20, hv30 = hv(px, 20), hv(px, 30)
        ivp = iv * 100
        rk = iv_rank(sym, ivp, vol_hist, log_df)

        out.update(
            expiry=expiry, dte=dte, atm_iv=ivp, hv20=hv20, hv30=hv30,
            iv_minus_hv20=ivp - hv20 if np.isfinite(hv20) else NAN,
            ivr=rk["ivr"] if rk else NAN, ivp=rk["ivp"] if rk else NAN,
            ivr_src=rk["src"] if rk else "n/a",
            exp_move=S * iv * math.sqrt(T), exp_move_pct=100 * iv * math.sqrt(T),
            put=p, call=c, credit=credit, credit_pct=100 * credit / S,
            margin=margin, rom=100 * credit * 100 / margin if margin else NAN,
            liquidity=liquidity_grade(p, c), corr_spy=corr,
        )
    except Exception as e:  # keep the report going if one name fails
        out["error"] = f"data error: {type(e).__name__}: {e}"[:160]
    return out


# =============================================================================
# Classification
# =============================================================================
def classify(row: dict) -> tuple[str, str, list[str]]:
    """Returns (status, main_reason, flags)."""
    R = RULES
    flags: list[str] = []
    if row.get("error"):
        return "AVOID", row["error"], flags

    ivr, edge, cp, liq = row["ivr"], row["iv_minus_hv20"], row["credit_pct"], row["liquidity"]
    ivr_known = np.isfinite(ivr)
    in_band = row["put"]["in_band"] and row["call"]["in_band"]
    low_prem = cp < R["credit_pct_low"]

    if low_prem:
        flags.append(f"LOW PREMIUM ({cp:.2f}% of underlying)")
    if not in_band:
        flags.append("no strike inside 10-15Δ on one side; nearest used")
    if not ivr_known:
        flags.append("IV Rank unavailable - confirm on broker before entry")
    if np.isfinite(edge) and edge < 0:
        flags.append("IV below HV20 (no volatility risk premium)")

    if liq == "POOR":
        return "AVOID", "Poor option liquidity (wide markets or thin open interest)", flags
    if low_prem and (not ivr_known or ivr < R["ivr_watch"]):
        return "AVOID", "Premium too thin for the tail risk of a naked strangle", flags

    edge_ok = np.isfinite(edge) and edge >= R["edge_trade"]
    if (ivr_known and ivr >= R["ivr_trade"] and edge_ok and cp >= R["credit_pct_trade"]
            and liq == "OK" and in_band):
        return "TRADE CANDIDATE", (
            f"IVR {ivr:.0f}, IV {edge:+.1f} pts over HV20, credit {cp:.2f}% with good liquidity"), flags

    if np.isfinite(edge) and edge < 0:
        return "WAIT", "Implied vol is below realized - selling vol here has no edge", flags
    if ivr_known and ivr < R["ivr_watch"]:
        return "WAIT", f"IV Rank only {ivr:.0f} - premium is cheap relative to its own history", flags

    bits = []
    if ivr_known:
        bits.append(f"IVR {ivr:.0f}")
    if np.isfinite(edge):
        bits.append(f"IV-HV20 {edge:+.1f}")
    bits.append(f"credit {cp:.2f}%")
    missing = []
    if ivr_known and ivr < R["ivr_trade"]:
        missing.append(f"IVR<{R['ivr_trade']}")
    if not edge_ok:
        missing.append(f"edge<{R['edge_trade']}")
    if cp < R["credit_pct_trade"]:
        missing.append(f"credit<{R['credit_pct_trade']}%")
    if liq != "OK":
        missing.append(f"liquidity {liq}")
    if not ivr_known:
        missing.append("IVR unconfirmed")
    return "WATCH", f"{', '.join(bits)}; short of trade rules on: {', '.join(missing) or 'strike band'}", flags


def score(row: dict, status: str) -> float:
    if row.get("error"):
        return -99
    w = {"TRADE CANDIDATE": 3, "WATCH": 2, "WAIT": 0.5, "AVOID": -5}[status]
    ivr = row["ivr"] if np.isfinite(row["ivr"]) else 40
    edge = row["iv_minus_hv20"] if np.isfinite(row["iv_minus_hv20"]) else 0
    liq = {"OK": 0.5, "FAIR": 0, "POOR": -1}[row["liquidity"]]
    return w + 2 * ivr / 100 + np.clip(edge, -5, 10) / 5 + min(row["credit_pct"], 3) + liq


# =============================================================================
# Report
# =============================================================================
def f(x, fmt="{:.1f}", na="n/a"):
    return fmt.format(x) if isinstance(x, (int, float)) and np.isfinite(x) else na


def build_report(rows, closes, now) -> str:
    L = []
    L.append(f"# Short Strangle Scanner — {now:%a %d %b %Y, %H:%M} ET\n")
    L.append(f"_Screen: {DTE_MIN}–{DTE_MAX} DTE (target {DTE_TARGET}), short strikes "
             f"{DELTA_MIN:.2f}–{DELTA_MAX:.2f} delta. Quotes: Yahoo Finance (may be ~15 min delayed). "
             f"Screening report only — not an order or investment instruction._\n")

    counts = pd.Series([r["status"] for r in rows]).value_counts()
    L.append("**Summary:** " + " · ".join(f"{k}: {v}" for k, v in counts.items()) + "\n")

    # Highlights
    ranked = sorted([r for r in rows if r["status"] != "AVOID"], key=lambda r: -r["score"])[:4]
    L.append("## Most relevant today\n")
    if not ranked:
        L.append("Nothing meets the rules today. No trade is the trade.\n")
    for r in ranked:
        L.append(f"**{r['ticker']} — {r['status']}.** {r['reason']}. "
                 f"{r['put']['strike']:g}P / {r['call']['strike']:g}C {r['expiry']} ({r['dte']} DTE) "
                 f"for ~${r['credit']:.2f} credit; expected move ±${r['exp_move']:.2f} "
                 f"({r['exp_move_pct']:.1f}%).\n")

    # Table 1 - volatility
    L.append("## Volatility picture\n")
    L.append("| Ticker | Status | Price | Expiry (DTE) | ATM IV | IVR / IVP (source) | HV20 | HV30 | IV−HV20 | Exp. move | Corr SPY |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("error"):
            L.append(f"| {r['ticker']} | {r['status']} | {f(r.get('price', NAN), '{:.2f}')} | — | — | — | — | — | — | — | — |")
            continue
        L.append(
            f"| {r['ticker']} | {r['status']} | {r['price']:.2f} | {r['expiry']} ({r['dte']}) | "
            f"{f(r['atm_iv'])}% | {f(r['ivr'], '{:.0f}')} / {f(r['ivp'], '{:.0f}')} ({r['ivr_src']}) | "
            f"{f(r['hv20'])}% | {f(r['hv30'])}% | {f(r['iv_minus_hv20'], '{:+.1f}')} | "
            f"±{r['exp_move']:.2f} ({r['exp_move_pct']:.1f}%) | {f(r['corr_spy'], '{:.2f}')} |")

    # Table 2 - structure
    L.append("\n## Strangle structure & liquidity\n")
    L.append("| Ticker | Short put (Δ) | Short call (Δ) | Credit | Credit % | Est. margin* | Credit/margin | OI P / C | Vol P / C | Bid-ask % P / C | Liquidity |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("error"):
            L.append(f"| {r['ticker']} | — | — | — | — | — | — | — | — | — | {r['error']} |")
            continue
        p, c = r["put"], r["call"]
        L.append(
            f"| {r['ticker']} | {p['strike']:g} ({p['delta']:.2f}) | {c['strike']:g} ({c['delta']:.2f}) | "
            f"${r['credit']:.2f} | {r['credit_pct']:.2f}% | ${r['margin']:,.0f} | {f(r['rom'])}% | "
            f"{p['oi']:,} / {c['oi']:,} | {p['vol']:,} / {c['vol']:,} | "
            f"{p['spread_pct']:.0f}% / {c['spread_pct']:.0f}% | {r['liquidity']} |")
    L.append("\n_*Typical Reg-T naked estimate per 1-lot. Your broker's (or portfolio-margin) figure will differ._\n")

    # Flags
    flagged = [r for r in rows if r["flags"]]
    if flagged:
        L.append("## Flags\n")
        for r in flagged:
            L.append(f"- **{r['ticker']}:** " + "; ".join(r["flags"]))
        L.append("")

    # Concentration
    L.append("## Concentration & correlation\n")
    idx = [t for t in ("SPY", "QQQ", "IWM") if t in closes.columns]
    rets = np.log(closes[idx]).diff().dropna().tail(CORR_LOOKBACK)
    cm = rets.corr()
    pairs = [f"{a}/{b} {cm.loc[a, b]:.2f}" for i, a in enumerate(idx) for b in idx[i + 1:]]
    L.append(f"{CORR_LOOKBACK}-day return correlations: " + ", ".join(pairs) + ".")
    active_idx = [r["ticker"] for r in rows if r["ticker"] in idx and r["status"] in ("TRADE CANDIDATE", "WATCH")]
    if len(active_idx) >= 2:
        L.append(f"\n**Warning:** {', '.join(active_idx)} are all live ideas but move largely as one "
                 f"US-equity bet. Treat them as a single position for sizing, or pick one.")
    else:
        L.append("\nOnly one (or none) of SPY/QQQ/IWM is active, so index concentration is limited today.")
    cluster = [f"{r['ticker']} ({r['corr_spy']:.2f})" for r in rows
               if not r.get("error") and r["ticker"] not in idx and r["corr_spy"] >= RULES["corr_cluster"]]
    if cluster:
        L.append(f"\nAlso tightly tied to SPY (≥{RULES['corr_cluster']:.2f}): {', '.join(cluster)}.")
    diversifiers = [r["ticker"] for r in rows
                    if not r.get("error") and r["ticker"] not in idx and r["corr_spy"] < 0.4 and r["status"] != "AVOID"]
    if diversifiers:
        L.append(f"\nLow-correlation alternatives today: {', '.join(diversifiers)}.")

    L.append("\n---\n_Before acting: confirm IV Rank and margin on your broker, check ex-dividend dates "
             "(early-assignment risk on short calls) and scheduled macro events inside the expiry window "
             "(FOMC, CPI, OPEC for USO/XLE). Screening report only._")
    return "\n".join(L)


def send_email(subject: str, md_text: str):
    host, to = os.getenv("SMTP_HOST"), os.getenv("REPORT_TO")
    if not host or not to:
        print("Email not configured (SMTP_HOST / REPORT_TO) - skipping.")
        return
    import markdown

    html = markdown.markdown(md_text, extensions=["tables"])
    html = ("<html><body style='font-family:Arial,sans-serif;font-size:13px'>"
            "<style>table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 6px}</style>"
            f"{html}</body></html>")
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, os.getenv("SMTP_USER", to), to
    msg.attach(MIMEText(md_text, "plain"))
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587"))) as s:
        s.starttls()
        if os.getenv("SMTP_USER"):
            s.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS", ""))
        s.sendmail(msg["From"], to.split(","), msg.as_string())
    print(f"Emailed report to {to}")


def append_iv_log(rows, today):
    new = pd.DataFrame([{"date": today.isoformat(), "ticker": r["ticker"], "atm_iv": round(r["atm_iv"], 3)}
                        for r in rows if not r.get("error")])
    if new.empty:
        return
    old = pd.read_csv(IV_LOG) if IV_LOG.exists() else pd.DataFrame(columns=new.columns)
    old = old[old["date"] != today.isoformat()]
    pd.concat([old, new]).to_csv(IV_LOG, index=False)


# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore trading-day / time window")
    ap.add_argument("--no-email", action="store_true")
    args = ap.parse_args()

    now = dt.datetime.now(ET)
    if not args.force:
        ok, msg = market_gate(now)
        if not ok:
            print(msg)
            return 0

    today = now.date()
    print(f"Scanning {len(TICKERS)} tickers at {now:%Y-%m-%d %H:%M} ET ...")
    symbols = sorted(set(TICKERS) | {"SPY"})
    closes = download_closes(symbols)
    vol_syms = sorted({VOL_INDEX[t] for t in TICKERS if t in VOL_INDEX})
    try:
        vol_hist = download_closes(vol_syms)
    except Exception:
        vol_hist = pd.DataFrame()
    log_df = pd.read_csv(IV_LOG) if IV_LOG.exists() else pd.DataFrame(columns=["date", "ticker", "atm_iv"])
    r = risk_free()

    rows = []
    for sym in TICKERS:
        row = scan_ticker(sym, closes, vol_hist, r, today, log_df)
        row["status"], row["reason"], row["flags"] = classify(row)
        row["score"] = score(row, row["status"])
        rows.append(row)
        print(f"  {sym:4s} {row['status']:16s} {row['reason']}")

    report = build_report(rows, closes, now)
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{today.isoformat()}.md"
    path.write_text(report, encoding="utf-8")
    append_iv_log(rows, today)
    print(f"Report written to {path}")

    if not args.no_email:
        n_trade = sum(r["status"] == "TRADE CANDIDATE" for r in rows)
        send_email(f"Strangle scan {today:%d %b}: {n_trade} trade candidate(s)", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
