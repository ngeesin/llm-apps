"""Tests for firm_dedupe. Run with:  python -m pytest tests/ -q  (or plain python)."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firm_dedupe import FirmDeduper, group_firms, normalize_name  # noqa: E402


# --------------------------------------------------------------------------- #
# normalize_name
# --------------------------------------------------------------------------- #
def test_normalize_case_and_punctuation():
    assert normalize_name("PiMCO") == "pimco"
    assert normalize_name("pimco") == "pimco"
    assert normalize_name("Black-Rock") == "blackrock" or \
        normalize_name("Black-Rock") == "black rock"


def test_normalize_strips_geo_and_legal():
    assert normalize_name("BlackRock (UK) Ltd.") == "blackrock"
    assert normalize_name("BlackRock Australia") == "blackrock"
    assert normalize_name("BlackRock") == "blackrock"


def test_normalize_accents():
    assert normalize_name("Société Générale") == normalize_name("Societe Generale")


def test_normalize_blank():
    assert normalize_name("") == ""
    assert normalize_name(None) == ""
    assert normalize_name(float("nan")) == ""


# --------------------------------------------------------------------------- #
# grouping behaviour
# --------------------------------------------------------------------------- #
def _groups(df, col="investor"):
    out = group_firms(df, col)
    return {
        gid: sorted(sub[col].tolist())
        for gid, sub in out.groupby("firm_group")
    }


def test_case_variants_group():
    df = pd.DataFrame({"investor": ["PiMCO", "pimco", "PIMCO"]})
    groups = _groups(df)
    assert len(groups) == 1


def test_geographic_variants_group():
    df = pd.DataFrame(
        {"investor": ["BlackRock", "BlackRock Australia", "BlackRock (UK)"]}
    )
    groups = _groups(df)
    assert len(groups) == 1, groups


def test_acronym_heuristic_groups_goldman():
    # "GS" == initials of "Goldman Sachs" -> caught by the acronym heuristic
    # even with the alias dict disabled.
    d = FirmDeduper(aliases={}).fit(["GS", "Goldman Sachs"])
    assert d.group_id("GS") == d.group_id("Goldman Sachs")


def test_alias_dictionary_groups_pimco_expansion():
    # PIMCO's expansion is not derivable by acronym rules -> needs the alias.
    d = FirmDeduper().fit(["PIMCO", "Pacific Investment Management Company"])
    assert d.group_id("PIMCO") == d.group_id(
        "Pacific Investment Management Company"
    )


def test_distinct_firms_stay_separate():
    df = pd.DataFrame({"investor": ["BlackRock", "Goldman Sachs", "PIMCO"]})
    groups = _groups(df)
    assert len(groups) == 3, groups


def test_end_to_end_mixed_column():
    df = pd.DataFrame(
        {
            "investor": [
                "PiMCO",
                "pacific investment management company",
                "pimco",
                "Goldman sachs",
                "GS",
                "BlackRock",
                "BlackRock Australia",
                "BlackRock (uk)",
            ]
        }
    )
    out = group_firms(df, "investor")
    canon = dict(zip(out["investor"], out["firm_canonical"]))
    # three real firms
    assert out["firm_group"].nunique() == 3, out
    # all PIMCO variants share one canonical label
    assert (
        canon["PiMCO"]
        == canon["pimco"]
        == canon["pacific investment management company"]
    )
    # all BlackRock variants share one canonical label
    assert canon["BlackRock"] == canon["BlackRock Australia"] == canon["BlackRock (uk)"]
    # Goldman variants share one
    assert canon["Goldman sachs"] == canon["GS"]


def test_custom_alias_extends_defaults():
    d = FirmDeduper(aliases={"tr": "T. Rowe Price", "t rowe price": "T. Rowe Price"})
    d.fit(["TR", "T. Rowe Price"])
    assert d.group_id("TR") == d.group_id("T. Rowe Price")
    # defaults still present
    assert "goldman sachs" in {normalize_name(k) for k in d.aliases}


def test_missing_column_raises():
    df = pd.DataFrame({"investor": ["PIMCO"]})
    try:
        group_firms(df, "nope")
    except KeyError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")


def test_preserves_row_order_and_nan():
    df = pd.DataFrame({"investor": ["PIMCO", None, "pimco", "GS"]})
    out = group_firms(df, "investor")
    assert len(out) == 4
    assert list(df["investor"].fillna("X")) == list(out["investor"].fillna("X"))


if __name__ == "__main__":
    # Allow running without pytest installed.
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
