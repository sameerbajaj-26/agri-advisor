#!/usr/bin/env python3
"""Ingest real agri documents (ICAR / SAU / KVK package-of-practices) into the ChromaDB KB.

    python scripts/ingest_docs.py                 # idempotent: unchanged documents are skipped
    python scripts/ingest_docs.py --rebuild       # drop the collection and re-ingest everything
    python scripts/ingest_docs.py --refresh       # re-download sources (else saved copies are used)
    python scripts/ingest_docs.py --prune         # also remove docs no longer listed in urls.txt

Sources come ONLY from urls.txt (see its header). Nothing is hardcoded or invented here.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))  # works even if the package is not pip-installed

from app.config import get_settings  # noqa: E402
from app.errors import SourceUnavailable  # noqa: E402
from app.rag.embed import get_embedder  # noqa: E402
from app.rag.ingest import UrlsFileError, ingest_all, parse_urls_file  # noqa: E402

EMPTY_MSG = """\
STOP: {path} has no sources, so there is nothing to ingest.

Add one real document per line (no placeholders):

    <URL or local PDF path> | <crop> | <title (optional)> | <region (optional)>

Supply package-of-practices / crop-production documents from ICAR institutes, state agricultural
universities (e.g. JNKVV, UAS Bengaluru) or KVKs. Use crop = general for multi-crop documents.
Example of the LINE SHAPE only (not a real link):
    https://<official-site>/<file>.pdf | wheat | <official document title> | Madhya Pradesh
For the milestone check ("wheat sowing time Madhya Pradesh"), include at least one wheat
document that covers Madhya Pradesh. If a site's certificate is broken or blocks scripts,
download the PDF in a browser and put its local path in the first column instead.
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--urls", type=Path, default=ROOT / "urls.txt")
    ap.add_argument("--rebuild", action="store_true", help="drop the collection first")
    ap.add_argument("--refresh", action="store_true", help="re-download, ignore saved copies")
    ap.add_argument("--prune", action="store_true", help="delete docs not listed in urls.txt")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")  # fmt: skip
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        entries = parse_urls_file(args.urls)
    except UrlsFileError as exc:
        print(f"ERROR in {args.urls}: {exc}", file=sys.stderr)
        return 2
    if not entries:
        print(EMPTY_MSG.format(path=args.urls), file=sys.stderr)
        return 1

    settings = get_settings()
    print(f"Embedding model: {settings.embedding_model}  |  store: {settings.chroma_path}")
    try:
        embedder = get_embedder(settings.embedding_model)
        results = ingest_all(
            entries, settings, embedder,
            rebuild=args.rebuild, refresh=args.refresh, prune=args.prune,
        )  # fmt: skip
    except SourceUnavailable as exc:
        print(f"ERROR: {exc.user_message()}", file=sys.stderr)
        return 2

    print("\nSummary")
    for r in results:
        extra = f" ({r.chunks} chunks)" if r.chunks else ""
        note = f" - {r.detail}" if r.detail else ""
        print(f"  [{r.status:9}] {r.source}{extra}{note}")
    failed = sum(r.status == "failed" for r in results)
    print(f"\n{sum(r.status == 'ingested' for r in results)} ingested, "
          f"{sum(r.status == 'unchanged' for r in results)} unchanged, "
          f"{sum(r.status == 'skipped' for r in results)} skipped, {failed} failed")  # fmt: skip
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
