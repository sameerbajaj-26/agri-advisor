import time

import httpx

key = input("data.gov.in API key: ").strip()
base = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
common = {"api-key": key, "format": "json", "limit": 5}
filt = {"filters[state.keyword]": "Karnataka", "filters[commodity]": "Tomato"}


def run(label, url, **kw):
    t = time.time()
    try:
        r = httpx.get(url, timeout=25, **kw)
        n = len(r.json().get("records", [])) if r.status_code == 200 else "-"
        print(f"{label}: status={r.status_code} records={n} time={time.time() - t:.1f}s")
    except Exception as e:
        print(f"{label}: FAILED {type(e).__name__} after {time.time() - t:.1f}s")


ua1 = "agri-advisor/0.1 (sameerbajaj9e@gmail.com)"
run("F our own UA", base, params={**common, **filt}, headers={"User-Agent": ua1})
run("G Mozilla", base, params={**common, **filt}, headers={"User-Agent": "Mozilla/5.0"})