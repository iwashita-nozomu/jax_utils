#!/usr/bin/env python3
"""
safe_file_extractor.py

Produce a list of files reported by ruff that are "safe" to auto-fix,
i.e. files that are not modified in any other branch compared to origin/main.

Writes `/tmp/safe_files.txt` and `reports/static-analysis/safe_files.txt` (repo-relative paths).
"""
import json
import subprocess
from pathlib import Path
import sys


def load_ruff_files(ruff_json_path: Path):
    if not ruff_json_path.exists():
        print(f"ruff JSON not found: {ruff_json_path}")
        return []
    data = json.loads(ruff_json_path.read_text())
    files = sorted({item.get("filename") for item in data if item.get("filename")})
    # Normalize to repo relative
    rel = []
    for f in files:
        if f.startswith(str(Path.cwd())):
            rel.append(str(Path(f).relative_to(Path.cwd())))
        elif f.startswith("/workspace/"):
            rel.append(f[len("/workspace/"):])
        else:
            rel.append(f)
    return sorted(set(rel))


def gather_modified_files(base: str = "origin/main"):
    branches = subprocess.check_output([
        "git",
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/heads",
        "refs/remotes",
    ]).decode().splitlines()
    modified = set()
    for br in branches:
        if br == base:
            continue
        try:
            out = subprocess.check_output(["git", "diff", "--name-only", f"{base}..{br}"]).decode()
        except subprocess.CalledProcessError:
            out = ""
        for line in out.splitlines():
            if line:
                modified.add(line.strip())
    return modified


def main():
    ruff_json = Path("reports/static-analysis/ruff.json")
    ruff_files = load_ruff_files(ruff_json)
    modified = gather_modified_files()
    safe = [f for f in ruff_files if f not in modified]
    out_tmp = Path("/tmp/safe_files.txt")
    out_repo = Path("reports/static-analysis/safe_files.txt")
    out_tmp.write_text("\n".join(safe))
    out_repo.parent.mkdir(parents=True, exist_ok=True)
    out_repo.write_text("\n".join(safe))
    print(f"SAFE_COUNT {len(safe)}")
    for s in safe:
        print(s)


if __name__ == "__main__":
    main()
