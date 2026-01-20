# Fundamental Check

Rules-based intraday fundamental bias scoring tool for US equities.

## Usage

```bash
python fundamental_check.py
```

## Example output (AAPL)

```
Ticker: AAPL
Final Bias Score: +3 / 5
Bias Label: Bullish (Long-biased)
Bucket Breakdown:
  - Growth: +1 | Revenue YoY strong at 12%. EPS YoY modest at 8%.
  - Profitability/Quality: +2 | Operating margin strong at 29%. Margins improving. Free cash flow positive.
  - Balance Sheet: +0 | Net debt/EBITDA healthy at 1.5x. Interest coverage solid at 12.0x.
  - Valuation: +0 | Forward P/E reasonable at 23.4x.
Confidence: High
Final Verdict: Supports longs
```
