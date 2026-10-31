import re
import sys
from pathlib import Path

sys.path.insert(0, "backend")
from app.config import Settings

defined = set(Settings.model_fields) | {
    n for n in dir(Settings) if isinstance(getattr(Settings, n, None), property)
}
used = {}
for f in Path("backend/app").rglob("*.py"):
    for name in re.findall(r"\bsettings\.([a-z_][a-z0-9_]*)", f.read_text(encoding="utf-8")):
        used.setdefault(name, set()).add(f.name)

missing = {n: sorted(fs) for n, fs in used.items() if n not in defined}
print("MISSING:" if missing else "all settings used in the code are defined")
for n, fs in sorted(missing.items()):
    print(f"  {n}  (used in {', '.join(fs)})")