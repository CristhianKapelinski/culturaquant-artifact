"""Per-item rarity: Wikidata sitelink count + pt.wikipedia log10 monthly pageviews."""

from __future__ import annotations

import math
from statistics import mean

from fetch import pageviews


def log_pageviews(title: str) -> float:
    """Mean of log10(monthly views + 1) over the fixed 2024 window.

    Returns 0.0 if the article has no pageview data (treated as maximally rare).
    """
    data = pageviews(title)
    if not data:
        return 0.0
    months = [it.get("views", 0) for it in data.get("items", [])]
    if not months:
        return 0.0
    return round(mean(math.log10(v + 1) for v in months), 4)


def rarity_band(sitelinks: int) -> str:
    """Coarse band used for item-by-item control matching.

    Most long-tail items sit at 0-1 sitelinks; we bucket finely at the bottom.
    """
    if sitelinks <= 0:
        return "sl0"
    if sitelinks == 1:
        return "sl1"
    if sitelinks <= 3:
        return "sl2-3"
    return "sl4+"
