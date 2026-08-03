"""LLM-backed alias resolution for firm_dedupe (Google Gemini).

The deterministic layers in :mod:`firm_dedupe.dedupe` cannot link names that
share no characters -- ``GS`` <-> ``Goldman Sachs`` or ``PIMCO`` <-> ``Pacific
Investment Management Company``. Those require *world knowledge*, which is
exactly what a language model provides. This module supplies that knowledge as
a drop-in replacement for the hand-curated alias dictionary.

The contract is one method::

    resolver.resolve(names: list[str]) -> dict[str, str]

mapping each input name to a canonical firm name. Names that map to the same
canonical string are merged by the deduper. Any object implementing that method
works -- tests inject a stub so no network is needed; :class:`GeminiAliasResolver`
is the real, Gemini-backed implementation.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("firm_dedupe.llm")

# Default model: a Flash-Lite-tier model is plenty for this simple grouping /
# canonicalization task and is the cheapest option. As of 2026-08
# "gemini-2.5-flash-lite" is live (retires 2026-10-16); its successor is
# "gemini-3.1-flash-lite". Override via the constructor if the id has moved on.
DEFAULT_MODEL = "gemini-2.5-flash-lite"

# JSON schema Gemini must conform to (OpenAPI subset). A top-level array of
# {name, canonical_name} objects -- no prose, no invented fields.
_RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "canonical_name": {"type": "string"},
        },
        "required": ["name", "canonical_name"],
    },
}

_PROMPT = """\
You are an expert at entity resolution for investment firms and investors.

You are given a JSON list of raw name strings taken from a spreadsheet column.
They contain acronyms, abbreviations, expansions, and regional variants of the
SAME underlying firms (e.g. "GS", "Goldman Sachs" and "Goldman Sachs & Co." are
one firm; "PIMCO" and "Pacific Investment Management Company" are one firm).

For EACH name in the input, return an object {{"name": <the exact input string>,
"canonical_name": <the firm's canonical name>}}.

Rules:
- Assign the SAME canonical_name to every string that refers to the same firm,
  and DIFFERENT canonical_name to firms that are genuinely different.
- Use the firm's common, official name as canonical_name (e.g. "Goldman Sachs",
  "PIMCO", "BlackRock"). Be consistent: identical firm -> byte-identical string.
- Only group names you are confident refer to the same entity. When unsure,
  keep a name on its own (canonical_name = the name itself). Do NOT merge names
  just because they look similar.
- Only use the names provided. Do NOT invent names that are not in the input,
  and return exactly one object per input name.

Input names:
{names_json}
"""

# Above this many unique names in one call, split into chunks. Cross-chunk
# merging still happens automatically because merging keys on the canonical
# string, and a well-known firm yields the same canonical text in every chunk.
_DEFAULT_MAX_BATCH = 400


class GeminiAliasResolver:
    """Resolve firm-name variants to canonical names using Google Gemini.

    Parameters
    ----------
    model:
        Gemini model id. Defaults to :data:`DEFAULT_MODEL`.
    api_key:
        Gemini API key. If omitted, the ``google-genai`` client falls back to
        the ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` environment variable.
    temperature:
        Sampling temperature; 0.0 (default) for the most repeatable output.
    max_batch:
        Max names per request; larger inputs are chunked.
    client:
        Pre-built ``google.genai.Client`` (mainly for testing). If given,
        ``api_key`` is ignored.

    Notes
    -----
    ``google-genai`` is imported lazily so the rest of :mod:`firm_dedupe` works
    without it installed. A missing package raises a clear ImportError here.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_batch: int = _DEFAULT_MAX_BATCH,
        client: object = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_batch = max_batch

        if client is not None:
            self._client = client
            self._types = self._import_types()
        else:
            genai, types = self._import_genai()
            self._client = genai.Client(api_key=api_key)
            self._types = types

    # -- lazy imports (kept out of module import so the dep stays optional) --
    @staticmethod
    def _import_genai():
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "GeminiAliasResolver requires the 'google-genai' package. "
                "Install it with:  pip install google-genai"
            ) from exc
        return genai, types

    @staticmethod
    def _import_types():
        _, types = GeminiAliasResolver._import_genai()
        return types

    # -- public API ---------------------------------------------------------
    def resolve(self, names: Sequence[str]) -> Dict[str, str]:
        """Map each name to a canonical firm name (best-effort).

        Never raises on API failure: a failed request logs a warning and
        contributes nothing, so the caller degrades to its deterministic
        result rather than crashing.
        """
        unique = [n for n in dict.fromkeys(names) if n and n.strip()]
        if not unique:
            return {}

        result: Dict[str, str] = {}
        for start in range(0, len(unique), self.max_batch):
            chunk = unique[start : start + self.max_batch]
            result.update(self._resolve_chunk(chunk))
        return result

    def _resolve_chunk(self, names: List[str]) -> Dict[str, str]:
        prompt = _PROMPT.format(names_json=json.dumps(names, ensure_ascii=False))
        config = self._types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
        )
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
            payload = json.loads(resp.text)
        except Exception as exc:  # noqa: BLE001 - resilience is the point
            logger.warning(
                "Gemini alias resolution failed for a chunk of %d names: %s; "
                "falling back to deterministic grouping for these.",
                len(names),
                exc,
            )
            return {}

        allowed = set(names)
        out: Dict[str, str] = {}
        for item in payload if isinstance(payload, list) else []:
            name = item.get("name")
            canonical = item.get("canonical_name")
            # Guard against hallucinated names / blank canonicals.
            if name in allowed and canonical and str(canonical).strip():
                out[name] = str(canonical).strip()
        return out
