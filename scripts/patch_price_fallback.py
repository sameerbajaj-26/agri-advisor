#!/usr/bin/env python3
"""Adds the district->state price fallback. Run from the repo root:
    python scripts\\patch_price_fallback.py
Safe to re-run (skips edits already applied). Aborts a file untouched if an anchor is missing."""

import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
APP = ROOT / "backend" / "app"

FALLBACK_METHOD = '''    async def get_prices_with_fallback(
        self,
        *,
        state: str | None = None,
        district: str | None = None,
        commodity: str | None = None,
        max_records: int = 100,
    ) -> MandiPrices:
        """Exact scope first; if the district has no prices, show the state's markets,
        labelled as such. Never fabricates a district price."""
        result = await self.get_prices(
            state=state, district=district, commodity=commodity, max_records=max_records
        )
        if result.records or not district or not state:
            return result
        wider = await self.get_prices(state=state, commodity=commodity, max_records=max_records)
        if not wider.records:
            return result
        note = f"No price was reported for {district} district; these are other markets in {state}."
        return wider.model_copy(update={"scope": "state_fallback", "scope_note": note})

'''

PROMPT_RULES = """   do not guess a place.
7. PRICES: when calling get_mandi_prices, pass the state as well as the district (the state comes
   from geocode). If a result has scope 'state_fallback', tell the user its scope_note in your own
   words: no price was reported for their district, these are other markets in the state. Give
   today's price range from by_date, name a few markets with their district, and cite the price
   source. The price data is a snapshot of recent days with no long history: never say prices are
   rising, falling or good unless by_date itself shows it across several dates.
8. State the time window of every weather fact and alert (for example "in the next 48 hours"). If
   the question covers a longer period than the data, say so.
"""

# (file, marker proving it is already applied, [(old, new), ...])
EDITS = [
    (
        APP / "providers" / "mandi.py",
        "get_prices_with_fallback",
        [
            ("from typing import Any\n", "from typing import Any, Literal\n"),
            (
                "    records: list[MandiRecord]\n",
                "    records: list[MandiRecord]\n"
                '    scope: Literal["as_requested", "state_fallback"] = "as_requested"\n'
                "    scope_note: str | None = None\n",
            ),
            (
                "    async def _fetch(self, filters: dict[str, str], max_records: int)",
                FALLBACK_METHOD
                + "    async def _fetch(self, filters: dict[str, str], max_records: int)",
            ),
        ],
    ),
    (
        APP / "agent" / "tools.py",
        "get_prices_with_fallback",
        [
            (
                '        "price_unit": p.price_unit,\n',
                '        "price_unit": p.price_unit,\n'
                '        "scope": p.scope,\n'
                '        "scope_note": p.scope_note,\n',
            ),
            ("self._mandi.get_prices(", "self._mandi.get_prices_with_fallback("),
            (
                '"optionally by state/district. Call before any sell-or-hold or harvest-timing advice."',
                '"optionally by state/district. Call before any sell-or-hold or harvest-timing advice. "\n'
                '        "If the district has no price, other markets in the same state are returned and "\n'
                '        "scope_note says so: always tell the user."',
            ),
        ],
    ),
    (
        APP / "agent" / "prompts.py",
        "state_fallback",
        [
            ("   do not guess a place.\n", PROMPT_RULES),
            (
                "(cite it). Use null/[] when unsure.",
                '(cite it). "\n'
                '        "sell_or_hold.reasoning must never be empty: even for insufficient_data, explain why in "\n'
                '        "plain words (today\'s price range from the tool result with its source id in "\n'
                '        "sell_or_hold.sources, and that no price history is available to judge a trend). "\n'
                '        "Use null/[] when unsure.',
            ),
        ],
    ),
]


def main() -> int:
    plans: list[tuple[Path, str]] = []
    for path, marker, pairs in EDITS:
        if not path.exists():
            print(f"MISSING FILE: {path}")
            return 1
        text = path.read_text(encoding="utf-8")
        if marker in text:
            print(f"already applied: {path}")
            continue
        for old, new in pairs:
            if text.count(old) != 1:
                print(f"ABORT: {path}: expected exactly 1 match, found {text.count(old)} for:")
                print("   " + old.strip().splitlines()[0][:90])
                print("No files were changed.")
                return 1
            text = text.replace(old, new)
        plans.append((path, text))
    for path, text in plans:
        path.write_text(text, encoding="utf-8")
        print(f"patched: {path}")
    print("Done." if plans else "Nothing to do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
