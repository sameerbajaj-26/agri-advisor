"""Mandi (APMC) prices from Agmarknet via the data.gov.in OGD API.

Resource 9ef84268-d588-465a-a308-a864a43d0070 "Current Daily Price of Various Commodities from
Various Markets (Mandi)". Documented fields: state, district, market, commodity, variety, grade,
arrival_date (dd/mm/yyyy), min_price, max_price, modal_price. Filters: filters[state.keyword],
filters[district], filters[commodity]. Pagination: offset + limit; envelope has total/count/records.
Legacy capitalised names (e.g. Modal_x0020_Price) are normalised.
"""

import datetime as dt
from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel

from app.config import Settings
from app.errors import MalformedPayload, NotConfigured
from app.providers.base import (
    Provenance,
    ProviderContext,
    Sourced,
    cache_key,
    cached_fetch,
)
from app.providers.http import request_json

SOURCE = "Agmarknet via data.gov.in"
PRICE_UNIT = "INR per quintal (Agmarknet convention; confirm against live response)"
TTL = timedelta(hours=3)
STALE_MAX = timedelta(days=3)
MAX_PAGES = 10
IST = dt.timezone(timedelta(hours=5, minutes=30))
_REQUIRED = ("state", "district", "market", "commodity", "arrival_date")
_PRICES = ("min_price", "max_price", "modal_price")


class MandiRecord(BaseModel):
    state: str
    district: str
    market: str
    commodity: str
    variety: str | None = None
    grade: str | None = None
    arrival_date: dt.date
    min_price: float
    max_price: float
    modal_price: float


class MandiPrices(Sourced):
    filters: dict[str, str]
    price_unit: str = PRICE_UNIT
    total_available: int | None
    truncated: bool
    skipped_records: int  # source rows dropped because a price/date was unparseable
    dates_present: list[dt.date]
    latest_arrival_date: dt.date | None
    days_since_latest: int | None  # vs today in IST; >0 means no price reported today
    records: list[MandiRecord]
    scope: Literal["as_requested", "state_fallback"] = "as_requested"
    scope_note: str | None = None


def _norm_key(key: str) -> str:
    return key.strip().lower().replace("_x0020_", "_").replace(" ", "_")


def _norm_filter(value: str) -> str:
    v = " ".join(value.split())
    return v.title() if v == v.lower() or v == v.upper() else v


def _parse_date(raw: Any) -> dt.date | None:
    if not isinstance(raw, str):
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return dt.datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_price(raw: Any) -> float | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int | float):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw.replace(",", "").strip())
        except ValueError:
            return None
    return None


def _int_or_none(raw: Any) -> int | None:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def parse_record(raw: Any) -> MandiRecord | None:
    """Return a record, or None if this single row is unusable (counted as skipped)."""
    if not isinstance(raw, dict):
        return None
    row = {_norm_key(str(k)): v for k, v in raw.items()}
    if any(not isinstance(row.get(f), str) or not row[f].strip() for f in _REQUIRED[:4]):
        return None
    arrival = _parse_date(row.get("arrival_date"))
    prices = [_parse_price(row.get(p)) for p in _PRICES]
    if arrival is None or any(p is None for p in prices):
        return None
    lo, hi, modal = prices
    assert lo is not None and hi is not None and modal is not None
    return MandiRecord(
        state=row["state"].strip(),
        district=row["district"].strip(),
        market=row["market"].strip(),
        commodity=row["commodity"].strip(),
        variety=row["variety"].strip() if isinstance(row.get("variety"), str) else None,
        grade=row["grade"].strip() if isinstance(row.get("grade"), str) else None,
        arrival_date=arrival,
        min_price=lo,
        max_price=hi,
        modal_price=modal,
    )


class MandiProvider:
    def __init__(self, ctx: ProviderContext, settings: Settings, *, page_size: int = 500) -> None:
        self._ctx = ctx
        self._url = f"{settings.data_gov_base_url.rstrip('/')}/{settings.mandi_resource_id}"
        self._key = settings.data_gov_api_key
        self._timeout = settings.slow_http_timeout_s
        self._page_size = page_size
        self._user_agent = settings.data_gov_user_agent

    async def get_prices(
        self,
        *,
        state: str | None = None,
        district: str | None = None,
        commodity: str | None = None,
        max_records: int = 100,
    ) -> MandiPrices:
        filters = {
            name: _norm_filter(value)
            for name, value in (("state", state), ("district", district), ("commodity", commodity))
            if value and value.strip()
        }
        return await cached_fetch(
            self._ctx,
            key=cache_key("mandi", sorted(filters.items()), max_records),
            ttl=TTL,
            stale_max=STALE_MAX,
            fetch=lambda: self._fetch(filters, max_records),
            model=MandiPrices,
            cacheable=lambda r: len(r.records) > 0,  # never cache an empty answer
        )

    async def get_prices_with_fallback(
        self,
        *,
        state: str | None = None,
        district: str | None = None,
        commodity: str | None = None,
        max_records: int = 100,
    ) -> MandiPrices:
        """Exact scope first; if the district has no prices today, show the state's markets,
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

    async def _fetch(self, filters: dict[str, str], max_records: int) -> MandiPrices:
        if self._key is None:
            raise NotConfigured(SOURCE, "set DATA_GOV_API_KEY (free key from data.gov.in)")
        page = max(1, min(self._page_size, max_records))
        params: dict[str, Any] = {"api-key": self._key.get_secret_value(), "format": "json"}
        for name, value in filters.items():
            params["filters[state.keyword]" if name == "state" else f"filters[{name}]"] = value

        raw_rows: list[Any] = []
        total: int | None = None
        offset = 0
        for _ in range(MAX_PAGES):
            payload = await request_json(
                self._ctx.client,
                self._url,
                source=SOURCE,
                retry=self._ctx.retry,
                sleep=self._ctx.sleep,
                params={**params, "offset": offset, "limit": page},
                headers={"User-Agent": self._user_agent},
                timeout=self._timeout,
            )
            if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
                note = payload.get("message") if isinstance(payload, dict) else None
                raise MalformedPayload(SOURCE, f"no 'records' list in response ({note or 'n/a'})")
            rows = payload["records"]
            total = _int_or_none(payload.get("total")) if total is None else total
            raw_rows.extend(rows)
            offset += len(rows)
            if (
                not rows
                or len(rows) < page
                or len(raw_rows) >= max_records
                or (total is not None and offset >= total)
            ):
                break

        raw_rows = raw_rows[:max_records]
        parsed = [parse_record(r) for r in raw_rows]
        records = [r for r in parsed if r is not None]
        if raw_rows and not records:
            raise MalformedPayload(SOURCE, "none of the returned rows had usable fields/prices")
        records.sort(key=lambda r: (r.arrival_date, r.market), reverse=True)
        dates = sorted({r.arrival_date for r in records}, reverse=True)
        today = self._ctx.clock().astimezone(IST).date()
        return MandiPrices(
            provenance=Provenance(
                source=SOURCE,
                source_url=self._url,
                fetched_at=self._ctx.clock(),
                cache_status="live",
            ),
            filters=filters,
            total_available=total,
            truncated=total is not None and total > len(raw_rows),
            skipped_records=len(raw_rows) - len(records),
            dates_present=dates,
            latest_arrival_date=dates[0] if dates else None,
            days_since_latest=(today - dates[0]).days if dates else None,
            records=records,
        )
