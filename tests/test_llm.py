"""Tests for the LLM alias-resolution layer, using a stub resolver (no network).

Run with:  python tests/test_llm.py   (or under pytest)
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firm_dedupe import FirmDeduper, group_firms  # noqa: E402


class StubResolver:
    """Stand-in for GeminiAliasResolver: returns a fixed name->canonical map."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def resolve(self, names):
        self.calls.append(list(names))
        return {n: self.mapping[n] for n in names if n in self.mapping}


class RaisingResolver:
    def resolve(self, names):
        raise RuntimeError("simulated API failure")


# --------------------------------------------------------------------------- #
# The resolver supplies world knowledge the deterministic layers cannot.
# --------------------------------------------------------------------------- #
def test_resolver_merges_expansion_that_fuzzy_cannot():
    # "PIMCO" normalizes to "pimco"; the full name normalizes to "pacific"
    # (generic words dropped) -> no shared tokens, no acronym match. Only the
    # LLM can link them. Static aliases disabled so we know it's the resolver.
    stub = StubResolver(
        {
            "PIMCO": "PIMCO",
            "Pacific Investment Management Company": "PIMCO",
        }
    )
    d = FirmDeduper(aliases={}, use_acronyms=False, resolver=stub).fit(
        ["PIMCO", "Pacific Investment Management Company"]
    )
    assert d.group_id("PIMCO") == d.group_id(
        "Pacific Investment Management Company"
    )
    # resolver was called on the cluster representatives
    assert stub.calls, "resolver.resolve was never called"


def test_resolver_merges_acronym_with_aliases_and_heuristic_off():
    stub = StubResolver({"GS": "Goldman Sachs", "Goldman Sachs & Co.": "Goldman Sachs"})
    d = FirmDeduper(aliases={}, use_acronyms=False, resolver=stub).fit(
        ["GS", "Goldman Sachs & Co."]
    )
    assert d.group_id("GS") == d.group_id("Goldman Sachs & Co.")


def test_prefer_llm_label_uses_canonical_name():
    stub = StubResolver(
        {"GS": "Goldman Sachs", "goldman sachs & co": "Goldman Sachs"}
    )
    df = pd.DataFrame({"investor": ["GS", "goldman sachs & co"]})
    out = group_firms(df, "investor", aliases={}, use_acronyms=False, resolver=stub)
    assert set(out["firm_canonical"]) == {"Goldman Sachs"}


def test_prefer_llm_label_false_falls_back_to_frequency():
    stub = StubResolver({"GS": "Goldman Sachs", "Goldman Sachs Group": "Goldman Sachs"})
    df = pd.DataFrame({"investor": ["GS", "Goldman Sachs Group", "GS"]})
    out = group_firms(
        df, "investor", aliases={}, use_acronyms=False,
        resolver=stub, prefer_llm_label=False,
    )
    # With LLM label disabled, the most-frequent raw value ("GS") wins.
    assert set(out["firm_canonical"]) == {"GS"}


def test_resolver_does_not_over_merge_distinct_firms():
    stub = StubResolver(
        {"BlackRock": "BlackRock", "Blackstone": "Blackstone"}
    )
    d = FirmDeduper(aliases={}, resolver=stub).fit(["BlackRock", "Blackstone"])
    assert d.group_id("BlackRock") != d.group_id("Blackstone")


def test_resolver_failure_degrades_gracefully():
    # A resolver that raises must not crash the pipeline; deterministic result
    # (case variants still merged) must survive.
    df = pd.DataFrame({"investor": ["BlackRock", "blackrock", "Blackstone"]})
    out = group_firms(df, "investor", resolver=RaisingResolver())
    canon = dict(zip(out["investor"], out["firm_group"]))
    assert canon["BlackRock"] == canon["blackrock"]
    assert canon["BlackRock"] != canon["Blackstone"]


def test_hallucinated_names_are_ignored():
    # Resolver returns a name that was never in the input -> must be dropped.
    class BadResolver:
        def resolve(self, names):
            return {"TOTALLY MADE UP": "Ghost Capital", names[0]: "Real Co"}

    d = FirmDeduper(aliases={}, use_acronyms=False, resolver=BadResolver()).fit(
        ["Real Co Partners", "Something Else"]
    )
    # No crash; the bogus mapping key simply had no effect.
    assert d.group_id("Real Co Partners") != -1


def test_no_resolver_is_unchanged_behaviour():
    df = pd.DataFrame({"investor": ["PiMCO", "pimco"]})
    out = group_firms(df, "investor")  # no llm, no resolver
    assert out["firm_group"].nunique() == 1


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
