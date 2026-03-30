from __future__ import annotations

import argparse
import json
from pathlib import Path


def strip_outputs(nb: dict) -> dict:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    # Drop some big metadata if present (keep kernelspec/language_info)
    md = nb.get("metadata", {})
    keep = {}
    for k in ("kernelspec", "language_info"):
        if k in md:
            keep[k] = md[k]
    nb["metadata"] = keep
    return nb


def main() -> int:
    p = argparse.ArgumentParser(description="Strip Jupyter notebook outputs for cleaner repos.")
    p.add_argument("paths", nargs="+", help="Notebook path(s) or directories.")
    args = p.parse_args()

    targets: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            targets.extend(sorted(path.glob("*.ipynb")))
        else:
            targets.append(path)

    changed = 0
    for nb_path in targets:
        if nb_path.suffix != ".ipynb" or not nb_path.exists():
            continue

        data = json.loads(nb_path.read_text(encoding="utf-8"))
        before = nb_path.stat().st_size
        data = strip_outputs(data)
        nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        after = nb_path.stat().st_size
        changed += 1
        print(f"{nb_path}: {before/1024/1024:.2f}MB -> {after/1024/1024:.2f}MB")

    print(f"Processed notebooks: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

