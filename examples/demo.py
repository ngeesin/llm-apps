"""Runnable demo of firm_dedupe on a messy investor column.

    python examples/demo.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firm_dedupe import group_firms  # noqa: E402


def main() -> None:
    df = pd.DataFrame(
        {
            "investor": [
                "PiMCO",
                "pacific investment management company",
                "pimco",
                "Goldman sachs",
                "GS",
                "Goldman Sachs & Co.",
                "BlackRock",
                "BlackRock Australia",
                "BlackRock (uk)",
                "Blackrock Inc.",
                "JP Morgan",
                "JPMorgan Chase",
                "J.P. Morgan",
                "Vanguard Group",
                "The Vanguard Group, Inc.",
                "Bridgewater",
                "Bridgewater Associates",
                None,
            ]
        }
    )

    out = group_firms(df, "investor")

    pd.set_option("display.max_colwidth", 45)
    pd.set_option("display.width", 120)

    print("Row-level result (original value -> group id + canonical label):\n")
    print(out.to_string(index=False))

    print("\n\nGrouped duplicates:\n")
    for gid, sub in out.groupby("firm_group"):
        label = sub["firm_canonical"].iloc[0]
        variants = ", ".join(repr(v) for v in sub["investor"].tolist())
        print(f"  [{gid}] {label}")
        print(f"        {variants}")

    print(
        f"\n{len(df)} raw rows collapsed into "
        f"{out['firm_group'].nunique()} distinct firms."
    )


if __name__ == "__main__":
    main()
