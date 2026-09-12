"""Create and cold-build the minimal arXiv source package after science checks.

The output is a local submission candidate, not an arXiv upload. Nothing is
written outside the CBDC tree. All output is durable and no /tmp is used.
"""

import gzip
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "revision-2026-09-12"
SRC = REV / "manuscript"
EV = REV / "evidence"
RELEASE = ROOT.parent / "release-2026-09-12"


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stdout[-4000:] + p.stderr[-2000:])
    return p.stdout


def main():
    a = json.loads((EV / "adjudicated-results.json").read_text())
    assert a["confirmatory_reproduction"]["replicates_exactly_equal"]
    assert a["confirmatory_reproduction"]["H2_confirmatory"] is False
    assert (
        a["permutation_controls"]["n"] == 22
        and a["permutation_controls"]["failed"] == 0
    )
    r = json.loads((EV / "ladder-replay.json").read_text())
    assert all(r["fields_equal"].values())
    log = (REV / "build/main.log").read_text()
    assert "undefined" not in log.lower() and "Overfull" not in log
    pdf = REV / "build/main.pdf"
    shutil.copy2(pdf, SRC / "main.pdf")
    (SRC / "main.bbl").write_bytes((REV / "build/main.bbl").read_bytes())
    RELEASE.mkdir(exist_ok=True)
    files = (
        [SRC / "main.tex", SRC / "references.bib", SRC / "main.bbl"]
        + sorted((SRC / "tables").glob("*.tex"))
        + sorted((SRC / "figures").glob("*.pdf"))
    )
    archive = RELEASE / "cbdc-september-2026-arxiv-source.tar.gz"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for p in files:
            data = p.read_bytes()
            info = tarfile.TarInfo(str(p.relative_to(SRC)))
            info.size = len(data)
            info.mtime = 1789171200
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    with archive.open("wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as gz:
            gz.write(buf.getvalue())
    cold = RELEASE / "cold-build"
    if cold.exists():
        raise RuntimeError(
            "Cold-build directory already exists; preserve it and use a new release directory."
        )
    cold.mkdir()
    with tarfile.open(archive) as t:
        t.extractall(cold, filter="data")
    output = run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cold,
    )
    (EV / "cold-build.log").write_text(output)
    coldlog = (cold / "main.log").read_text()
    assert "undefined" not in coldlog.lower() and "Overfull" not in coldlog
    original_text = run(["pdftotext", str(pdf), "-"], ROOT)
    cold_text = run(["pdftotext", str(cold / "main.pdf"), "-"], ROOT)
    assert original_text == cold_text, (
        "Rendered-text mismatch between working and cold build"
    )
    assert "Pending" not in cold_text
    for word in ("7.90", "31.35", "2.40", "14.85", "22 worlds"):
        assert word in cold_text, word
    finalpdf = RELEASE / "cbdc-september-2026-draft.pdf"
    shutil.copy2(pdf, finalpdf)
    checks = {
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "pdf_sha256": hashlib.sha256(finalpdf.read_bytes()).hexdigest(),
        "source_files": len(files),
        "rendered_text_equal": True,
        "cold_build_exit": 0,
        "undefined_references": 0,
        "overfull_boxes": 0,
        "arxiv_uploaded": False,
    }
    (EV / "package-validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    (RELEASE / "PACKAGE_VALIDATION.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
