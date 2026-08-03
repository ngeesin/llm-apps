# firm_dedupe — group name variants of the same investment firm / investor

Turn a messy column like this…

```
PiMCO
pacific investment management company
pimco
Goldman sachs
GS
BlackRock
BlackRock Australia
BlackRock (uk)
```

…into groups of the same underlying firm, each with a stable id and one
canonical label.

```python
import pandas as pd
from firm_dedupe import group_firms

out = group_firms(df, "investor")
#  adds two columns:
#    firm_group      -> integer id shared by all rows of the same firm
#    firm_canonical  -> one human-readable label per group
```

## Why one trick isn't enough

The variants in a real column come in **three different flavours**, and each
needs a different technique — this is the core insight behind the design:

| Flavour | Example | Technique |
|---|---|---|
| Case / punctuation / spacing | `PiMCO` ↔ `pimco` | **Normalization** |
| Legal + geographic qualifiers | `BlackRock`, `BlackRock Australia`, `BlackRock (UK)`, `Blackrock Inc.` | Normalization (strip suffixes) + **fuzzy matching** |
| Acronyms & expansions | `GS` ↔ `Goldman Sachs`, `PIMCO` ↔ `Pacific Investment Management Company` | **Acronym heuristic** + **alias dictionary** |

Acronyms are the hard part: `GS` and `Goldman Sachs` share almost no
characters, so *no* string-similarity metric will ever link them. They need
either an initials heuristic (`GS` = first letters of `Goldman Sachs`) or, when
even that fails (`PIMCO` isn't pure initials — it borrows the *Co* from
*Company*), a small curated lookup table.

## How it works

Each technique proposes pairs of raw strings that are "the same firm". All
pairs are fed into a **union-find** structure that merges them *transitively*
into clusters — so `GS → Goldman Sachs → Goldman Sachs & Co.` all land in one
group even though no single signal linked the first to the last.

```
raw names
   │
   ▼
1. normalize      lowercase · strip accents · drop "(uk)" · drop punctuation
   │              · drop legal suffixes (Inc/Ltd/LLC…) · drop geo (Australia/UK…)
   │              · drop generic words (Group/Capital/Management…)
   ▼
2. signals ──►  exact key match  ·  alias dictionary  ·  acronym heuristic  ·  fuzzy match
   │
   ▼
3. union-find    merge all proposed pairs into clusters
   │
   ▼
4. label         pick the most-frequent (tie: longest) raw value per cluster
```

The O(n²) fuzzy/acronym passes run over the **distinct** normalized names, not
the raw rows, so a column with 100k rows but 800 unique names stays fast.

## Usage

### Quick, one call

```python
from firm_dedupe import group_firms

out = group_firms(df, "investor")               # defaults are sensible
out = group_firms(df, "investor", threshold=92) # stricter: less merging
```

### Reusable engine (fit once, apply to new data)

```python
from firm_dedupe import FirmDeduper

deduper = FirmDeduper(threshold=88).fit(df["investor"])
deduper.canonical("GS")          # -> "Goldman Sachs & Co."
deduper.group_id("pimco")        # -> 0
deduper.mapping()                # -> {raw value: canonical label, ...}
```

### Teaching it your firms

The acronyms only your dataset knows about go in the alias dictionary. It is
*merged on top of* the built-in defaults, so you only add what's missing:

```python
group_firms(df, "investor", aliases={
    "tr":            "T. Rowe Price",
    "t rowe price":  "T. Rowe Price",
    "wf":            "Wells Fargo",
})
```

## LLM-augmented alias resolution (Gemini)

The hand-curated alias dictionary is the one part that needs *world knowledge* —
and it doesn't scale: someone has to type every `GS → Goldman Sachs` pair. You
can hand that job to Gemini instead, which knows these firms:

```python
out = group_firms(df, "investor", llm=True, aliases={})   # aliases={} = no manual table
```

Requires the optional dependency and a key:

```bash
pip install google-genai
export GEMINI_API_KEY=...
```

**How it fits in.** The LLM does **not** replace the whole algorithm — only the
alias step. The deterministic layers (normalization, acronym heuristic, fuzzy
matching) still run first and do the cheap work; Gemini is then called **once, on
the distinct cluster representatives** — not on every row — so cost scales with
the number of unique names, not the size of the column. On Flash-Lite-tier
pricing that's a few cents even for large columns (and the Gemini Batch API is
~50% cheaper for async jobs). It uses structured JSON output at `temperature=0`,
and when it names a firm, that official name becomes the group's `firm_canonical`
label.

**It fails safe.** If the API errors out, the resolver contributes nothing and
you get the deterministic grouping instead of a crash. You can also inject your
own resolver (any object with `resolve(names) -> {name: canonical}`) — useful for
testing, caching, or swapping providers:

```python
from firm_dedupe import FirmDeduper
FirmDeduper(resolver=my_resolver, aliases={}).fit(df["investor"])
```

## Tuning cheatsheet

| Parameter | Default | Raise it / turn on | Lower it / turn off |
|---|---|---|---|
| `threshold` | `88` | Fewer false merges (stricter) | Catch more typos/variants |
| `aliases` | `DEFAULT_ALIASES` | Add your firms' acronyms | `{}` to start clean |
| `use_acronyms` | `True` | Link `GS`↔`Goldman Sachs` | Off if it over-merges |
| `drop_geo` | `True` | Merge regional arms | `False` to keep `BlackRock UK` distinct |
| `drop_generic` | `True` | Merge `Bridgewater`↔`Bridgewater Associates` | `False` for finer distinctions |
| `llm` | `False` | `True` for Gemini alias resolution (needs key) | keep `False` for offline/deterministic |
| `prefer_llm_label` | `True` | Use the LLM's official firm name as the label | `False` to keep the most-frequent raw value |

## Design notes & honest limitations

- **Distinct legal entities stay split by default.** `JPMorgan Chase` is *not*
  auto-merged with `J.P. Morgan` because "Chase" is a real distinguishing
  token. If you want them together, add an alias — the tool errs toward
  *not* over-merging.
- **Fuzzy matching can't invent knowledge.** Any acronym that isn't derivable
  from initials must go in the alias dictionary *or* the Gemini layer; that's a
  deliberate, auditable seam rather than a black box.
- **The LLM can be wrong.** Gemini may occasionally over-merge (two different
  funds sharing a name) or mis-canonicalize. The full raw→canonical map is
  inspectable via `FirmDeduper.mapping()` — review it for high-stakes use. Note
  that with the LLM on, results depend on an external API and can drift across
  model versions even at `temperature=0`; leave `llm=False` when you need a
  fully offline, reproducible run.
- **Blank / NaN values** are grouped together into their own cluster and never
  merged with a real firm.
- **Determinism:** group ids are assigned by first appearance, so the same
  input always yields the same ids (deterministic layers only).
- **Dependencies:** `pandas` is required; `rapidfuzz` (faster fuzzy metric) and
  `google-genai` (the Gemini layer) are optional — the core works without both.

## Run it

```bash
pip install -r requirements.txt
python examples/demo.py         # end-to-end deterministic demo (no API)
python examples/demo_gemini.py  # Gemini alias resolution (needs GEMINI_API_KEY)
python tests/test_dedupe.py     # deterministic test suite
python tests/test_llm.py        # LLM-layer tests (stubbed, no network)
```
