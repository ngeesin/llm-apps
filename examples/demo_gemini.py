"""Live demo of the Gemini-backed alias resolver.

Groups a column of acronyms/expansions WITHOUT any hand-curated aliases -- the
Gemini resolver supplies the world knowledge. Requires:

    pip install google-genai
    export GEMINI_API_KEY=...          # your Gemini API key
    python examples/demo_gemini.py

If GEMINI_API_KEY is not set, the script explains what to do and exits cleanly.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from firm_dedupe import group_firms  # noqa: E402


def main() -> None:
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        print(
            "GEMINI_API_KEY is not set.\n"
            "  export GEMINI_API_KEY=your-key   then re-run this script.\n"
            "  (See examples/demo.py for the deterministic, no-API demo.)"
        )
        return

    df = pd.DataFrame(
        {
            "investor": [
                "PiMCO",
                "Pacific Investment Management Company",
                "pimco",
                "GS",
                "Goldman Sachs",
                "Goldman Sachs & Co.",
                "BlackRock",
                "BlackRock (UK)",
                "Blackstone",           # different firm -- must NOT merge with BlackRock
                "JPM",
                "J.P. Morgan",
                "Berkshire",
                "Berkshire Hathaway",
            ]
        }
    )

    # aliases={} disables the hand-curated dictionary, so any acronym/expansion
    # grouping you see is the LLM's doing, not a lookup table.
    out = group_firms(df, "investor", llm=True, aliases={})

    print("Grouped with Gemini (no hand-curated aliases):\n")
    for gid, sub in out.groupby("firm_group"):
        label = sub["firm_canonical"].iloc[0]
        variants = ", ".join(repr(v) for v in sub["investor"].tolist())
        print(f"  [{gid}] {label}")
        print(f"        {variants}")

    print(
        f"\n{len(df)} rows -> {out['firm_group'].nunique()} firms. "
        "Inspect the raw->canonical mapping to audit merges:\n"
    )
    for raw, canon in dict(zip(out["investor"], out["firm_canonical"])).items():
        print(f"    {raw!r:45} -> {canon!r}")


if __name__ == "__main__":
    main()
