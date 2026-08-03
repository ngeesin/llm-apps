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

# GeminiAliasResolver lives in a submodule that lazily imports google-genai, so
# importing the name here does not require the package to be installed. It only
# fails if you actually instantiate the resolver without google-genai present.
from .llm import DEFAULT_MODEL, GeminiAliasResolver

__all__ = [
    "DEFAULT_ALIASES",
    "DEFAULT_MODEL",
    "FirmDeduper",
    "GeminiAliasResolver",
    "group_firms",
    "normalize_name",
]

__version__ = "0.1.0"
