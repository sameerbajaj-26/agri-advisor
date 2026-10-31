from app.agent.grounding import cited_ids, collect_numbers, find_ungrounded

KNOWN = collect_numbers({"a": [28.6, 16.0, {"p": 2450.0}], "s": "2026-10-31", "ok": True})


def test_collect_numbers_ignores_strings_and_bools() -> None:
    assert sorted(KNOWN) == [16.0, 28.6, 2450.0]


def test_numbers_with_units_must_match_tool_values() -> None:
    assert find_ungrounded("Max 28 °C and 16 mm rain [S1]", KNOWN) == []  # within tolerance
    assert find_ungrounded("Max 35 °C", KNOWN) == ["35"]
    assert find_ungrounded("Price ₹2,450 per quintal", KNOWN) == []
    assert find_ungrounded("Price about ₹2,600 per quintal", KNOWN) == ["2,600"]
    assert find_ungrounded("Price 2450", KNOWN) == [] and find_ungrounded("Price 3000", KNOWN) == [
        "3000"
    ]
    assert find_ungrounded("soil pH 6.8", KNOWN) == ["6.8"]


def test_dates_times_citations_and_plain_small_numbers_are_ignored() -> None:
    t = "Fetched 2026-10-31 05:09 UTC [S1, S2] for the next 3 days, 48 hours"
    assert find_ungrounded(t, KNOWN) == []


def test_indic_digits_and_units() -> None:
    assert find_ungrounded("तापमान ३५ डिग्री", KNOWN) == ["३५"]
    assert find_ungrounded("बारिश १६ मिमी", KNOWN) == []


def test_cited_ids_parses_lists() -> None:
    assert cited_ids("a [S1] b [S2, S10] c [x]") == {"S1", "S2", "S10"}
