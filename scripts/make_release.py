#!/usr/bin/env python3
"""Export this standalone project only, without parent Git history or runtime artifacts."""
import argparse
import hashlib
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"src", "scripts", "configs", "docs", "tests", "examples", ".github"}
TOP_FILES = {"README.md", "README.ja.md", "LICENSE", "NOTICE", "CITATION.cff", "PROVENANCE.json",
             "pyproject.toml", "requirements-training.txt", ".gitignore"}
SKIP = {"__pycache__", ".pytest_cache", ".git", "node_modules"}
SECRET = re.compile(r"(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{24,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)")
PNG_ALLOWLIST = {
    "docs/figures/figure_child_subtree_selection_supervision_innerlabels_transparent.png": "9f2a95a63fde4eddc472157b07990a08f721776c63543b99f982136250602997",
    "docs/figures/figure_candidate_granularity_compact_transparent.png": "6effb94ee161c3dc0441f3de72736808139c5e522a8decc217ebac8ec48d2aad",
}


def release_files(root=ROOT):
    root = Path(root)
    files = []
    candidates = []
    for entry in sorted(root.iterdir()):
        if entry.name in TOP_FILES:
            candidates.append(entry)
        elif entry.name in DIRECTORIES:
            if entry.is_symlink():
                raise ValueError(f"Release must not contain symlinks: {entry.name}")
            candidates.extend(sorted(entry.rglob("*")))
    for path in candidates:
        rel = path.relative_to(root)
        if str(rel) in {"configs/manuscript_claims.json", "scripts/audit_historical_runs.py", "scripts/verify_paper_results.py", "scripts/export_public_decisions.py"}:
            continue
        if any(part in SKIP or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Release must not contain symlinks: {rel}")
        if not path.is_file():
            continue
        allowed = str(rel) in TOP_FILES or rel.parts[0] in DIRECTORIES
        if not allowed or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.stat().st_size > 5_000_000:
            raise ValueError(f"Unexpected large file: {rel}")
        if path.suffix.lower() == ".png":
            raw = path.read_bytes()
            if str(rel) not in PNG_ALLOWLIST or not raw.startswith(b"\x89PNG\r\n\x1a\n") or hashlib.sha256(raw).hexdigest() != PNG_ALLOWLIST[str(rel)]:
                raise ValueError(f"Unapproved binary figure asset: {rel}")
            files.append(path)
            continue
        text = path.read_text(encoding="utf-8")
        if ("/home/" + "miluser/") in text or ("/data/" + "Hibitan/") in text:
            raise ValueError(f"Host-specific path in release: {rel}")
        if SECRET.search(text):
            raise ValueError(f"Potential credential in release: {rel}")
        files.append(path)
    missing = TOP_FILES - {p.name for p in files if p.parent == root}
    if missing:
        raise ValueError(f"Missing release files: {sorted(missing)}")
    return files


def package(output):
    output = Path(output).resolve()
    files = release_files()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        checksums = []
        for path in files:
            relative = str(path.relative_to(ROOT))
            data = path.read_bytes()
            archive.writestr("soict-code/" + relative, data)
            checksums.append(hashlib.sha256(data).hexdigest() + "  " + relative)
        archive.writestr("soict-code/SHA256SUMS", "\n".join(checksums) + "\n")
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="New ZIP path, preferably outside this source folder")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print(f"Release audit passed: {len(release_files())} source/artifact files")
    elif args.output:
        print(f"Exported {package(args.output)} files to {args.output}")
    else:
        parser.error("Specify --check or --output")
