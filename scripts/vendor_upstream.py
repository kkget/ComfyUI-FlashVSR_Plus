"""Reproduce the pinned snapshot, isolating upstream's absolute package imports.

Usage: python scripts/vendor_upstream.py /path/to/FlashVSR_Plus
No download, pip install, or torch import is performed by this script.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

REVISION = "a08fb184a84cc2157c4f271234555fcda7902c87"


def main():
    source = Path(sys.argv[1]).resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"Expected upstream {REVISION}, found {revision}")
    root = Path(__file__).resolve().parents[1]
    target = root / "vendor" / "runtime"
    target.mkdir(parents=True, exist_ok=True)
    (root / "vendor" / "__init__.py").write_text("", encoding="utf-8")
    (target / "__init__.py").write_text("", encoding="utf-8")
    inputs = []
    for name in ("inference", "models", "model_io", "flashvsr_plus"):
        inputs.extend((p, p.relative_to(source)) for p in (source / name).rglob("*.py"))
    operator = source / "operators" / "flashvsr_attn"
    inputs.extend((p, p.relative_to(operator)) for p in (operator / "flashvsr_attn").rglob("*.py"))
    manifest = {}
    for path, relative in sorted(inputs):
        content = path.read_text(encoding="utf-8")
        dots = "." * len(relative.parts)
        # Only import paths change. Model, scheduler, masking and decoder math stay upstream.
        content = re.sub(r"(?m)^(\s*from )(inference|models|model_io|flashvsr_plus|flashvsr_attn)([. ])",
                         lambda m: m[1] + dots + m[2] + m[3], content)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8", newline="\n")
        manifest[relative.as_posix()] = hashlib.sha256(content.encode()).hexdigest()
    shutil.copyfile(source / "LICENSE", root / "LICENSE")
    prompt = root / "assets" / "posi_prompt.pth"
    prompt.parent.mkdir(exist_ok=True)
    shutil.copyfile(source / "prompt_tensor" / "posi_prompt.pth", prompt)
    # Preserve the optional operator's upstream provenance and installation metadata.
    for name in ("UPSTREAM.md", "README.md", "pyproject.toml"):
        shutil.copyfile(operator / name, target / "flashvsr_attn" / ("OPERATOR_" + name))
    (root / "vendor" / "manifest.json").write_text(json.dumps({
        "repository": "https://github.com/OpenImagingLab/FlashVSR_Plus",
        "revision": REVISION, "changes": "Relative import namespaces only",
        "sha256": manifest,
        "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Vendored {len(manifest)} Python files at {revision}")


if __name__ == "__main__":
    main()
