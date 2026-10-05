# Daily Short Strangle Scanner

Scans SPY, QQQ, IWM, GLD, SLV, XLE, XLF, EEM, USO, SMH, EWZ every US trading day,
~35 minutes after the open, for 35–50 DTE / 10–15 delta short strangles.
Screening report only — not an order or investment instruction.

## Setup (≈10 minutes)
1. Create a **private** GitHub repo and upload these files (keep the `.github/workflows` folder).
2. *Optional email:* Repo → Settings → Secrets and variables → Actions → add
   `SMTP_HOST`, `SMTP_PORT` (587), `SMTP_USER`, `SMTP_PASS`, `REPORT_TO`.
   For Gmail use `smtp.gmail.com` and an App Password.
3. Actions tab → enable workflows → "Daily Short Strangle Scanner" → **Run workflow** to test now.

Reports land in `reports/YYYY-MM-DD.md` (and in your inbox if email is set).

## Rules
All thresholds live in the `RULES` block at the top of `strangle_scanner.py`.
Defaults: TRADE CANDIDATE = IVR ≥ 50, ATM IV ≥ HV20 + 2 pts, credit ≥ 1.0% of underlying,
both strikes inside 10–15Δ, good liquidity. WATCH = partly there. WAIT = IVR < 30 or IV < HV.
AVOID = poor liquidity, or thin premium (< 0.6%) without high IVR, or missing data.

## Data notes
- Yahoo option quotes can be ~15 min delayed; deltas are computed from mid prices (Black-Scholes).
- IV Rank: CBOE vol-index proxy (^VIX, ^VXN, ^RVX, ^GVZ, ^OVX, ...) until the scanner has logged
  120 days of its own ATM IV in `iv_history.csv`; then it uses the ETF's own history.
  XLF and SMH have no vol index, so their IVR shows n/a for the first ~60 trading days.
- Margin is a Reg-T naked estimate; your broker's number will differ.
- For real-time data, replace `spot_price()` and `get_chain()` with your broker's API.
