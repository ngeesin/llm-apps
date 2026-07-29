"""Group name variants of the same investment firm / investor.

The column you get in the wild mixes several *kinds* of variation, and each
kind needs a different technique — no single string metric catches them all:

    1. Case / punctuation / whitespace   "PiMCO"  ->  "pimco"
       ................................  handled by *normalization*.

    2. Legal + geographic qualifiers      "BlackRock", "BlackRock Australia",
       ...............................     "BlackRock (UK)", "Blackrock Inc."
       These share a stable core once you strip the suffix, so normalization
       + *fuzzy matching* on the remaining tokens groups them.

    3. Acronyms and expansions            "GS" <-> "Goldman Sachs",
       ...........................         "PIMCO" <-> "Pacific Investment
                                           Management Company"
       These are NOT string-similar, so fuzzy matching cannot help. Two
       signals catch them: an *acronym heuristic* (initials of a multi-word
       name) and a *curated alias dictionary* for the ones the heuristic
       can't derive (PIMCO includes the "Co" of Company — not pure initials).

Every signal proposes "these two raw strings are the same firm". We feed all
of those pairings into a union-find structure, which merges them transitively
into clusters. Each cluster gets one stable integer id and one human-readable
canonical label.

Design goals: works on any iterable of strings or a pandas Series, tunable
threshold, an extensible alias dictionary, and no hard dependency beyond
pandas (rapidfuzz is used when present, with a stdlib fallback otherwise).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Optional fast fuzzy backend. rapidfuzz is ~100x faster and has a better
# token-set metric; fall back to the stdlib so the module always imports.
# --------------------------------------------------------------------------- #
try:
    from rapidfuzz import fuzz as _rf_fuzz

    _HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - exercised only without rapidfuzz
    import difflib

    _HAVE_RAPIDFUZZ = False


def _similarity(a: str, b: str) -> float:
    """Return a 0-100 token-order-insensitive similarity between two strings."""
    if _HAVE_RAPIDFUZZ:
        # token_set_ratio ignores word order and duplicate/extra tokens, so
        # "morgan stanley" vs "stanley morgan co" still scores high.
        return _rf_fuzz.token_set_ratio(a, b)
    # Stdlib fallback: sort tokens to gain order-insensitivity, then ratio.
    a_sorted = " ".join(sorted(a.split()))
    b_sorted = " ".join(sorted(b.split()))
    return difflib.SequenceMatcher(None, a_sorted, b_sorted).ratio() * 100.0


# --------------------------------------------------------------------------- #
# Vocabulary used during normalization.
# --------------------------------------------------------------------------- #

# Legal entity / structure words. Dropped because they are noise for identity:
# "BlackRock" and "BlackRock Inc." are the same firm.
_LEGAL_SUFFIXES = frozenset(
    {
        "inc", "incorporated", "corp", "corporation", "co", "company",
        "llc", "llp", "lp", "plc", "ltd", "limited", "gmbh", "ag", "sa",
        "nv", "bv", "spa", "pty", "pte", "ab", "as", "oyj", "kk",
        "lp", "l.p", "s.a", "n.v",
    }
)

# Geographic qualifiers. A regional arm ("BlackRock Australia") is treated as
# the same firm for grouping purposes. Kept separate from legal suffixes so a
# caller who wants to keep regional entities distinct can drop just this set.
_GEO_QUALIFIERS = frozenset(
    {
        "usa", "us", "uk", "eu", "emea", "apac", "global", "international",
        "australia", "australian", "america", "americas", "american",
        "asia", "europe", "european", "britain", "british", "england",
        "canada", "canadian", "china", "chinese", "france", "french",
        "germany", "german", "india", "indian", "ireland", "irish",
        "italy", "italian", "japan", "japanese", "korea", "korean",
        "netherlands", "dutch", "singapore", "spain", "spanish",
        "switzerland", "swiss", "hong", "kong", "hongkong", "luxembourg",
        "brazil", "mexico", "russia", "russian", "sweden", "swedish",
    }
)

# Common but low-signal descriptors. Removing them helps fuzzy matching focus
# on the distinctive part of the name (the brand), e.g. so "Bridgewater" and
# "Bridgewater Associates" collapse. NOT removed before acronym derivation.
_GENERIC_TERMS = frozenset(
    {
        "the", "and", "of", "for", "group", "holdings", "holding",
        "partners", "partner", "associates", "associate", "capital",
        "management", "asset", "assets", "investment", "investments",
        "investors", "advisors", "advisers", "advisory", "fund", "funds",
        "financial", "finance", "securities", "bank", "banc", "trust",
    }
)

# Punctuation -> space, so "black-rock" and "black rock" tokenize alike.
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_PAREN_RE = re.compile(r"\([^)]*\)")


def _strip_accents(text: str) -> str:
    """Fold accented characters to ASCII so 'Société' == 'Societe'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_name(
    name: object,
    *,
    drop_geo: bool = True,
    drop_generic: bool = True,
) -> str:
    """Reduce a raw firm name to a comparable canonical key.

    Steps: cast to str, strip accents, lowercase, remove parenthetical asides
    ("(UK)"), replace punctuation with spaces, then drop legal suffixes,
    optionally geographic qualifiers, and optionally generic descriptors.

    A blank / NaN input returns "" so callers can treat empty keys specially.

    >>> normalize_name("BlackRock (UK) Ltd.")
    'blackrock'
    >>> normalize_name("Pacific Investment Management Company")
    'pacific'
    """
    if name is None:
        return ""
    text = str(name)
    # pandas NaN arrives as the literal float nan -> "nan"; guard it.
    if text.strip().lower() in {"", "nan", "none", "null"}:
        return ""

    text = _strip_accents(text).lower()
    text = _PAREN_RE.sub(" ", text)      # "(uk)" -> " "
    text = _PUNCT_RE.sub(" ", text)      # drop &, ., -, / etc.

    tokens = _WS_RE.sub(" ", text).strip().split()

    drop = set(_LEGAL_SUFFIXES)
    if drop_geo:
        drop |= _GEO_QUALIFIERS
    if drop_generic:
        drop |= _GENERIC_TERMS

    kept = [t for t in tokens if t not in drop]

    # If we stripped everything (e.g. name was only generic words), fall back
    # to the pre-generic tokens so we never produce an empty key from a real
    # name — better to group loosely than to merge all such rows together.
    if not kept:
        kept = [t for t in tokens if t not in _LEGAL_SUFFIXES] or tokens

    return " ".join(kept)


def _acronym_of(tokens: Sequence[str]) -> str:
    """First letter of each token: ['goldman','sachs'] -> 'gs'."""
    return "".join(t[0] for t in tokens if t)


# --------------------------------------------------------------------------- #
# A starter alias dictionary for acronyms/expansions the heuristic can't derive.
# Keys and values are matched case-insensitively; extend this for your dataset.
# Every raw string that maps here joins the group of its canonical value.
# --------------------------------------------------------------------------- #
DEFAULT_ALIASES: Dict[str, str] = {
    "pimco": "Pacific Investment Management Company",
    "pacific investment management company": "Pacific Investment Management Company",
    "gs": "Goldman Sachs",
    "goldman": "Goldman Sachs",
    "goldman sachs": "Goldman Sachs",
    "jpm": "JPMorgan",
    "jp morgan": "JPMorgan",
    "jpmorgan": "JPMorgan",
    "j.p. morgan": "JPMorgan",
    "ms": "Morgan Stanley",
    "morgan stanley": "Morgan Stanley",
    "bofa": "Bank of America",
    "boa": "Bank of America",
    "bank of america": "Bank of America",
    "brk": "Berkshire Hathaway",
    "berkshire": "Berkshire Hathaway",
    "berkshire hathaway": "Berkshire Hathaway",
    "blk": "BlackRock",
    "blackrock": "BlackRock",
    "db": "Deutsche Bank",
    "deutsche bank": "Deutsche Bank",
    "ubs": "UBS",
    "cs": "Credit Suisse",
    "credit suisse": "Credit Suisse",
    "vanguard": "Vanguard",
    "kkr": "KKR",
    "kohlberg kravis roberts": "KKR",
}


# --------------------------------------------------------------------------- #
# Union-find (disjoint set) to merge pairings transitively into clusters.
# --------------------------------------------------------------------------- #
class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self._parent: Dict[str, str] = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression.
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> Dict[str, List[str]]:
        clusters: Dict[str, List[str]] = defaultdict(list)
        for item in self._parent:
            clusters[self.find(item)].append(item)
        return clusters


class FirmDeduper:
    """Reusable, configurable investment-firm name deduplicator.

    Parameters
    ----------
    threshold:
        Fuzzy similarity (0-100) at/above which two normalized names are
        considered the same firm. 88 is a good default: it merges typos and
        qualifier variants while keeping genuinely different firms apart.
        Raise it if you see over-merging; lower it to catch more variants.
    aliases:
        Mapping of variant -> canonical name for acronyms/expansions that no
        heuristic can derive. Defaults to :data:`DEFAULT_ALIASES`. Pass a dict
        (merged on top of the defaults) or ``{}`` to disable and start clean.
    use_acronyms:
        If True, treat a short all-initials token as matching the multi-word
        name it abbreviates (e.g. "GS" ~ "Goldman Sachs").
    drop_geo, drop_generic:
        Forwarded to :func:`normalize_name` to control how aggressively the
        key is reduced before comparison.
    """

    def __init__(
        self,
        *,
        threshold: float = 88.0,
        aliases: Optional[Dict[str, str]] = None,
        use_acronyms: bool = True,
        drop_geo: bool = True,
        drop_generic: bool = True,
    ) -> None:
        self.threshold = threshold
        self.use_acronyms = use_acronyms
        self.drop_geo = drop_geo
        self.drop_generic = drop_generic
        if aliases is None:
            self.aliases = dict(DEFAULT_ALIASES)
        else:
            merged = dict(DEFAULT_ALIASES)
            merged.update(aliases)
            self.aliases = merged
        # Normalize alias keys once for lookup.
        self._alias_lookup = {
            self._norm(k): v for k, v in self.aliases.items()
        }

    # -- internal helpers ---------------------------------------------------
    def _norm(self, name: object) -> str:
        return normalize_name(
            name, drop_geo=self.drop_geo, drop_generic=self.drop_generic
        )

    def _cluster(self, raw_values: Sequence[str]) -> Dict[str, int]:
        """Core algorithm. Returns raw value -> group id (dense, 0-based)."""
        uniques = list(dict.fromkeys(raw_values))  # de-dup, keep order
        uf = _UnionFind(uniques)

        # Precompute the normalized key for every unique raw value.
        norm: Dict[str, str] = {v: self._norm(v) for v in uniques}

        # Signal 1: identical normalized key -> same firm.
        by_key: Dict[str, List[str]] = defaultdict(list)
        for v in uniques:
            by_key[norm[v]].append(v)
        for members in by_key.values():
            first = members[0]
            for other in members[1:]:
                uf.union(first, other)

        # Signal 2: curated alias dictionary. Everything whose normalized key
        # resolves to the same canonical name is unioned via a synthetic node.
        alias_anchor: Dict[str, str] = {}
        for v in uniques:
            canonical = self._alias_lookup.get(norm[v])
            if canonical is None:
                continue
            anchor = alias_anchor.setdefault(canonical, v)
            uf.union(anchor, v)

        # Work on one representative per normalized key for the O(n^2) passes
        # below — cheaper, and avoids re-deciding identical strings.
        reps = [members[0] for members in by_key.values() if members[0]]

        # Signal 3: acronym heuristic. A short representative that is exactly
        # the initials of a longer multi-word representative -> union them.
        if self.use_acronyms:
            multiword = [(r, norm[r].split()) for r in reps if " " in norm[r]]
            for r in reps:
                key = norm[r]
                if " " in key or not (1 < len(key) <= 6):
                    continue  # only treat compact single tokens as acronyms
                for other, toks in multiword:
                    if len(toks) >= 2 and _acronym_of(toks) == key:
                        uf.union(r, other)

        # Signal 4: fuzzy match on normalized keys. O(reps^2) but reps is the
        # count of *distinct* names, not rows, so it stays small in practice.
        for i in range(len(reps)):
            ni = norm[reps[i]]
            if not ni:
                continue
            for j in range(i + 1, len(reps)):
                nj = norm[reps[j]]
                if not nj:
                    continue
                if _similarity(ni, nj) >= self.threshold:
                    uf.union(reps[i], reps[j])

        # Assign dense, deterministic group ids (ordered by first appearance).
        root_to_id: Dict[str, int] = {}
        value_to_id: Dict[str, int] = {}
        for v in uniques:
            root = uf.find(v)
            if root not in root_to_id:
                root_to_id[root] = len(root_to_id)
            value_to_id[v] = root_to_id[root]
        return value_to_id

    def _canonical_labels(
        self, raw_values: Sequence[str], value_to_id: Dict[str, int]
    ) -> Dict[int, str]:
        """Pick one display label per group: the most frequent raw value,
        tie-broken by the longest (usually the fullest, most descriptive form).
        """
        counts_by_group: Dict[int, Counter] = defaultdict(Counter)
        for v in raw_values:
            counts_by_group[value_to_id[v]][v] += 1

        labels: Dict[int, str] = {}
        for gid, counter in counts_by_group.items():
            # sort by (frequency desc, length desc, alphabetical) for stability
            best = max(
                counter.items(),
                key=lambda kv: (kv[1], len(str(kv[0])), str(kv[0])),
            )
            labels[gid] = str(best[0])
        return labels

    # -- public API ---------------------------------------------------------
    def fit(self, values: Iterable[object]) -> "FirmDeduper":
        """Learn groupings from an iterable of raw names."""
        raw = ["" if v is None else str(v) for v in values]
        self._value_to_id = self._cluster(raw)
        self._labels = self._canonical_labels(raw, self._value_to_id)
        return self

    def group_id(self, value: object) -> int:
        """Group id for a single raw value (-1 if unseen during fit)."""
        return self._value_to_id.get("" if value is None else str(value), -1)

    def canonical(self, value: object) -> str:
        """Canonical label for a single raw value ("" if unseen)."""
        gid = self.group_id(value)
        return self._labels.get(gid, "")

    def mapping(self) -> Dict[str, str]:
        """Raw value -> canonical label, for every value seen during fit."""
        return {v: self._labels[gid] for v, gid in self._value_to_id.items()}


def group_firms(
    df: "pd.DataFrame",
    column: str,
    *,
    group_id_col: str = "firm_group",
    canonical_col: str = "firm_canonical",
    threshold: float = 88.0,
    aliases: Optional[Dict[str, str]] = None,
    use_acronyms: bool = True,
    drop_geo: bool = True,
    drop_generic: bool = True,
) -> "pd.DataFrame":
    """Add group-id and canonical-name columns to ``df`` for ``column``.

    Returns a copy of ``df`` with two new columns:

    - ``group_id_col``  (default ``"firm_group"``): a dense integer id shared
      by all rows judged to be the same firm.
    - ``canonical_col`` (default ``"firm_canonical"``): a single human-readable
      label chosen for each group.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"investor": ["PiMCO", "pimco", "GS", "Goldman Sachs",
    ...                                 "BlackRock", "BlackRock (UK)"]})
    >>> out = group_firms(df, "investor")
    >>> out.groupby("firm_group")["investor"].apply(list).tolist()
    [['PiMCO', 'pimco'], ['GS', 'Goldman Sachs'], ['BlackRock', 'BlackRock (UK)']]
    """
    if column not in df.columns:
        raise KeyError(
            f"column {column!r} not found; available: {list(df.columns)}"
        )

    deduper = FirmDeduper(
        threshold=threshold,
        aliases=aliases,
        use_acronyms=use_acronyms,
        drop_geo=drop_geo,
        drop_generic=drop_generic,
    )
    deduper.fit(df[column].tolist())

    result = df.copy()
    result[group_id_col] = df[column].map(deduper.group_id)
    result[canonical_col] = df[column].map(deduper.canonical)
    return result
