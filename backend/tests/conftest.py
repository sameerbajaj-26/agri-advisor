"""Test-only environment. Dummy values live here and nowhere in app code."""

import os
import tempfile

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test")
os.environ.setdefault("CACHE_DIR", tempfile.mkdtemp(prefix="agri-test-cache-"))
