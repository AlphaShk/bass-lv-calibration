"""Copy the tagged figures of the executed replication notebook to docs/img/.

Cells whose metadata carries a tag ``fig:<name>`` are exported as ``docs/img/<name>.png``,
so the README always shows exactly what the notebook computed. Run it after re-executing
``notebooks/bass_lv_replication.ipynb``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "bass_lv_replication.ipynb"
OUT = ROOT / "docs" / "img"


def main() -> int:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for cell in cells:
        names = [t.split(":", 1)[1] for t in cell.get("metadata", {}).get("tags", []) if t.startswith("fig:")]
        images = [o["data"]["image/png"] for o in cell.get("outputs", []) if "image/png" in o.get("data", {})]
        for name in names:
            if not images:
                raise SystemExit(f"cell tagged fig:{name} has no image; execute the notebook first")
            (OUT / f"{name}.png").write_bytes(base64.b64decode(images[-1]))
            written.append(name)
    print("wrote", ", ".join(f"docs/img/{n}.png" for n in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
