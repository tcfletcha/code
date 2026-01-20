"""
Intraday fundamental bias check for US equities.

This module provides a rules-based, transparent scoring system intended to
confirm or block intraday long/short scalp setups. It is NOT for entry timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yfinance as yf


# =========================
# Config: thresholds/weights
# =========================
CONFIG = {
    "growth": {
        "revenue_yoy_positive": 0.05,
        "revenue_yoy_strong": 0.15,
        "eps_yoy_positive": 0.05,
        "eps_yoy_strong": 0.20,
    },
    "profitability": {
        "operating_margin_good": 0.15,
        "operating_margin_excellent": 0.25,
        "fcf_positive_years": 2,
    },
    "balance_sheet": {
        "net_debt_to_ebitda_good": 2.0,
        "interest_coverage_good": 6.0,
    },
    "valuation": {
        "forward_pe_expensive": 28.0,
        "forward_pe_cheap": 14.0,
        "premium_to_sector_warn": 1.4,
    },
    "confidence": {
        "high_min_fields": 10,
        "medium_min_fields": 6,
    },
}


@dataclass
class BucketScore:
    score: int
    reason: str


@dataclass
class FundamentalSnapshot:
    revenue_yoy: Optional[float]
    eps_yoy: Optional[float]
    operating_margin: Optional[float]
    operating_margin_trend: Optional[float]
    fcf_recent: Optional[float]
    fcf_trend: Optional[float]
    net_debt: Optional[float]
    ebitda: Optional[float]
    interest_coverage: Optional[float]
    forward_pe: Optional[float]
    sector_forward_pe: Optional[float]
    available_fields: int


class DataProvider:
    """Abstractable data layer for fundamentals."""

    def get_fundamentals(self, ticker: str) -> FundamentalSnapshot:
        raise NotImplementedError


class YFinanceProvider(DataProvider):
    """YFinance implementation for fundamental metrics."""

    def get_fundamentals(self, ticker: str) -> FundamentalSnapshot:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.get_info()
        financials = yf_ticker.get_financials()
        cashflow = yf_ticker.get_cashflow()
        balance_sheet = yf_ticker.get_balance_sheet()

        available_fields = 0

        def count_field(value: Optional[float]) -> None:
            nonlocal available_fields
            if value is not None:
                available_fields += 1

        revenue_yoy = _calc_yoy_growth(financials, "Total Revenue")
        count_field(revenue_yoy)
        eps_yoy = _calc_yoy_growth(financials, "Diluted EPS")
        count_field(eps_yoy)

        operating_margin = info.get("operatingMargins")
        count_field(operating_margin)
        operating_margin_trend = _calc_margin_trend(financials)
        count_field(operating_margin_trend)

        fcf_recent, fcf_trend = _calc_fcf_metrics(cashflow)
        count_field(fcf_recent)
        count_field(fcf_trend)

        total_debt = _safe_get(balance_sheet, "Total Debt")
        cash = _safe_get(balance_sheet, "Cash And Cash Equivalents")
        net_debt = None
        if total_debt is not None and cash is not None:
            net_debt = total_debt - cash
        count_field(net_debt)

        ebitda = info.get("ebitda")
        count_field(ebitda)
        interest_coverage = info.get("interestCoverage")
        count_field(interest_coverage)

        forward_pe = info.get("forwardPE")
        count_field(forward_pe)
        sector_forward_pe = info.get("sectorForwardPE")
        count_field(sector_forward_pe)

        return FundamentalSnapshot(
            revenue_yoy=revenue_yoy,
            eps_yoy=eps_yoy,
            operating_margin=operating_margin,
            operating_margin_trend=operating_margin_trend,
            fcf_recent=fcf_recent,
            fcf_trend=fcf_trend,
            net_debt=net_debt,
            ebitda=ebitda,
            interest_coverage=interest_coverage,
            forward_pe=forward_pe,
            sector_forward_pe=sector_forward_pe,
            available_fields=available_fields,
        )


def _safe_get(frame, label: str) -> Optional[float]:
    if frame is None or frame.empty:
        return None
    if label not in frame.index:
        return None
    series = frame.loc[label]
    if series.dropna().empty:
        return None
    return float(series.dropna().iloc[0])


def _calc_yoy_growth(frame, label: str) -> Optional[float]:
    if frame is None or frame.empty or label not in frame.index:
        return None
    series = frame.loc[label].dropna()
    if len(series) < 2:
        return None
    latest, prior = series.iloc[0], series.iloc[1]
    if prior == 0:
        return None
    return (latest - prior) / abs(prior)


def _calc_margin_trend(financials) -> Optional[float]:
    if financials is None or financials.empty:
        return None
    revenue = financials.loc["Total Revenue"].dropna() if "Total Revenue" in financials.index else None
    operating_income = (
        financials.loc["Operating Income"].dropna() if "Operating Income" in financials.index else None
    )
    if revenue is None or operating_income is None or len(revenue) < 2 or len(operating_income) < 2:
        return None
    latest_margin = operating_income.iloc[0] / revenue.iloc[0] if revenue.iloc[0] else None
    prior_margin = operating_income.iloc[1] / revenue.iloc[1] if revenue.iloc[1] else None
    if latest_margin is None or prior_margin is None:
        return None
    return latest_margin - prior_margin


def _calc_fcf_metrics(cashflow) -> Tuple[Optional[float], Optional[float]]:
    if cashflow is None or cashflow.empty:
        return None, None
    if "Free Cash Flow" not in cashflow.index:
        return None, None
    series = cashflow.loc["Free Cash Flow"].dropna()
    if series.empty:
        return None, None
    recent = float(series.iloc[0])
    trend = None
    if len(series) >= 2 and series.iloc[1] != 0:
        trend = (series.iloc[0] - series.iloc[1]) / abs(series.iloc[1])
    return recent, trend


def _confidence_label(available_fields: int) -> str:
    if available_fields >= CONFIG["confidence"]["high_min_fields"]:
        return "High"
    if available_fields >= CONFIG["confidence"]["medium_min_fields"]:
        return "Medium"
    return "Low"


def score_growth(snapshot: FundamentalSnapshot) -> BucketScore:
    score = 0
    reasons: List[str] = []

    rev = snapshot.revenue_yoy
    eps = snapshot.eps_yoy
    if rev is None:
        reasons.append("Revenue YoY unavailable.")
    else:
        if rev >= CONFIG["growth"]["revenue_yoy_strong"]:
            score += 1
            reasons.append(f"Revenue YoY strong at {rev:.0%}.")
        elif rev >= CONFIG["growth"]["revenue_yoy_positive"]:
            score += 0
            reasons.append(f"Revenue YoY modest at {rev:.0%}.")
        else:
            score -= 1
            reasons.append(f"Revenue YoY weak at {rev:.0%}.")

    if eps is None:
        reasons.append("EPS YoY unavailable.")
    else:
        if eps >= CONFIG["growth"]["eps_yoy_strong"]:
            score += 1
            reasons.append(f"EPS YoY strong at {eps:.0%}.")
        elif eps >= CONFIG["growth"]["eps_yoy_positive"]:
            reasons.append(f"EPS YoY modest at {eps:.0%}.")
        else:
            score -= 1
            reasons.append(f"EPS YoY weak at {eps:.0%}.")

    score = max(-2, min(2, score))
    reason = " ".join(reasons) if reasons else "Growth signals mixed."
    return BucketScore(score=score, reason=reason)


def score_profitability(snapshot: FundamentalSnapshot) -> BucketScore:
    score = 0
    reasons: List[str] = []

    margin = snapshot.operating_margin
    margin_trend = snapshot.operating_margin_trend
    if margin is None:
        reasons.append("Operating margin unavailable.")
    else:
        if margin >= CONFIG["profitability"]["operating_margin_excellent"]:
            score += 1
            reasons.append(f"Operating margin strong at {margin:.0%}.")
        elif margin >= CONFIG["profitability"]["operating_margin_good"]:
            reasons.append(f"Operating margin ok at {margin:.0%}.")
        else:
            score -= 1
            reasons.append(f"Operating margin weak at {margin:.0%}.")

    if margin_trend is not None:
        if margin_trend > 0:
            score += 1
            reasons.append("Margins improving.")
        elif margin_trend < 0:
            score -= 1
            reasons.append("Margins compressing.")

    fcf_recent = snapshot.fcf_recent
    fcf_trend = snapshot.fcf_trend
    if fcf_recent is None:
        reasons.append("Free cash flow unavailable.")
    elif fcf_recent > 0:
        score += 1
        reasons.append("Free cash flow positive.")
        if fcf_trend is not None:
            if fcf_trend > 0:
                score += 1
                reasons.append("Free cash flow improving.")
            elif fcf_trend < 0:
                score -= 1
                reasons.append("Free cash flow slowing.")
    else:
        score -= 1
        reasons.append("Free cash flow negative.")

    score = max(-2, min(2, score))
    reason = " ".join(reasons) if reasons else "Profitability signals mixed."
    return BucketScore(score=score, reason=reason)


def score_balance_sheet(snapshot: FundamentalSnapshot) -> BucketScore:
    score = 0
    reasons: List[str] = []
    net_debt = snapshot.net_debt
    ebitda = snapshot.ebitda

    if net_debt is None or ebitda is None or ebitda == 0:
        reasons.append("Net debt/EBITDA unavailable.")
    else:
        leverage = net_debt / ebitda
        if leverage <= CONFIG["balance_sheet"]["net_debt_to_ebitda_good"]:
            score += 1
            reasons.append(f"Net debt/EBITDA healthy at {leverage:.1f}x.")
        else:
            score -= 1
            reasons.append(f"Net debt/EBITDA elevated at {leverage:.1f}x.")

    coverage = snapshot.interest_coverage
    if coverage is None:
        reasons.append("Interest coverage unavailable.")
    else:
        if coverage >= CONFIG["balance_sheet"]["interest_coverage_good"]:
            score += 1
            reasons.append(f"Interest coverage solid at {coverage:.1f}x.")
        else:
            score -= 1
            reasons.append(f"Interest coverage weak at {coverage:.1f}x.")

    score = max(-1, min(1, score))
    reason = " ".join(reasons) if reasons else "Balance sheet signals mixed."
    return BucketScore(score=score, reason=reason)


def score_valuation(snapshot: FundamentalSnapshot) -> BucketScore:
    score = 0
    reasons: List[str] = []
    forward_pe = snapshot.forward_pe
    sector_pe = snapshot.sector_forward_pe

    if forward_pe is None:
        reasons.append("Forward P/E unavailable.")
    else:
        if forward_pe <= CONFIG["valuation"]["forward_pe_cheap"]:
            score += 1
            reasons.append(f"Forward P/E cheap at {forward_pe:.1f}x.")
        elif forward_pe >= CONFIG["valuation"]["forward_pe_expensive"]:
            score -= 1
            reasons.append(f"Forward P/E expensive at {forward_pe:.1f}x.")
        else:
            reasons.append(f"Forward P/E reasonable at {forward_pe:.1f}x.")

    if forward_pe is not None and sector_pe is not None and sector_pe > 0:
        premium = forward_pe / sector_pe
        if premium >= CONFIG["valuation"]["premium_to_sector_warn"]:
            score -= 1
            reasons.append(f"Valuation premium vs sector at {premium:.1f}x.")

    score = max(-2, min(2, score))
    reason = " ".join(reasons) if reasons else "Valuation signals mixed."
    return BucketScore(score=score, reason=reason)


def bias_label(score: int) -> str:
    if score >= 4:
        return "Strong Bullish (Long-only)"
    if score >= 2:
        return "Bullish (Long-biased)"
    if score <= -4:
        return "Strong Bearish (Short-only)"
    if score <= -2:
        return "Bearish (Short-biased)"
    return "Neutral (Reduce size / optional skip)"


def final_verdict(score: int) -> str:
    if score >= 2:
        return "Supports longs"
    if score <= -2:
        return "Supports shorts"
    return "Do not trade intraday"


def fundamental_check(ticker: str, provider: Optional[DataProvider] = None) -> Dict[str, object]:
    """Run the fundamental bias check and return a structured result."""
    provider = provider or YFinanceProvider()
    snapshot = provider.get_fundamentals(ticker)

    growth = score_growth(snapshot)
    profitability = score_profitability(snapshot)
    balance = score_balance_sheet(snapshot)
    valuation = score_valuation(snapshot)

    total = growth.score + profitability.score + balance.score + valuation.score
    total = max(-5, min(5, total))

    label = bias_label(total)
    confidence = _confidence_label(snapshot.available_fields)

    return {
        "ticker": ticker.upper(),
        "score": total,
        "label": label,
        "confidence": confidence,
        "verdict": final_verdict(total),
        "buckets": {
            "Growth": growth,
            "Profitability/Quality": profitability,
            "Balance Sheet": balance,
            "Valuation": valuation,
        },
    }


def print_summary(result: Dict[str, object]) -> None:
    """Print a trader-friendly summary."""
    print(f"Ticker: {result['ticker']}")
    print(f"Final Bias Score: {result['score']:+d} / 5")
    print(f"Bias Label: {result['label']}")
    print("Bucket Breakdown:")
    buckets: Dict[str, BucketScore] = result["buckets"]
    for name, bucket in buckets.items():
        print(f"  - {name}: {bucket.score:+d} | {bucket.reason}")
    print(f"Confidence: {result['confidence']}")
    print(f"Final Verdict: {result['verdict']}")


if __name__ == "__main__":
    example = fundamental_check("AAPL")
    print_summary(example)
