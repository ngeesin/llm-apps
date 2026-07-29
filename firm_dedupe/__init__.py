"""firm_dedupe: group name variants of investment firms / investors.

Public API
----------
- group_firms:        add group id + canonical label columns to a DataFrame.
- FirmDeduper:        reusable configurable engine (fit once, reuse mapping).
- normalize_name:     the normalization step, exposed for inspection/testing.
- DEFAULT_ALIASES:    starter alias dictionary for hard acronym cases.
"""

from .dedupe import (
    DEFAULT_ALIASES,
    FirmDeduper,
    group_firms,
    normalize_name,
)

__all__ = [
    "DEFAULT_ALIASES",
    "FirmDeduper",
    "group_firms",
    "normalize_name",
]

__version__ = "0.1.0"
