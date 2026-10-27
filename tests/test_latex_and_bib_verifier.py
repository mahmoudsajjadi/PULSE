"""
test_latex_and_bib_verifier.py
Comprehensive empirical verifier for LaTeX manuscript and refs.bib bibliography.
Executed by Challenger Final 2.
"""

import os
import re
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX_FILE = os.path.join(BASE_DIR, "manuscript.tex")
BIB_FILE = os.path.join(BASE_DIR, "refs.bib")
PDF_FILE = os.path.join(BASE_DIR, "manuscript.pdf")
AUX_FILE = os.path.join(BASE_DIR, "manuscript.aux")
LOG_FILE = os.path.join(BASE_DIR, "manuscript.log")
BLG_FILE = os.path.join(BASE_DIR, "manuscript.blg")

def test_refs_bib_syntax_and_entries():
    print("=== Test 1: refs.bib Syntax & Entries ===")
    assert os.path.exists(BIB_FILE), f"Missing {BIB_FILE}"
    with open(BIB_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Check balanced braces
    open_braces = content.count("{")
    close_braces = content.count("}")
    print(f"Brace counts: open={open_braces}, close={close_braces}")
    assert open_braces == close_braces, f"Mismatched braces in refs.bib: {open_braces} vs {close_braces}"

    # Extract all entries
    entry_pattern = re.compile(r'@(\w+)\s*\{\s*([^,]+),')
    entries = entry_pattern.findall(content)
    keys = [k.strip() for _, k in entries]
    print(f"Total entries found: {len(entries)}")
    assert len(keys) == len(set(keys)), f"Duplicate bib keys detected: {[k for k in keys if keys.count(k) > 1]}"
    print("All keys are unique.")
    return keys

def test_standards_presence(keys):
    print("\n=== Test 2: Standards Presence (IEEE C37.118 & FFR) ===")
    assert "ieeec37118" in keys, "Key 'ieeec37118' missing from refs.bib!"
    assert "erickson2020ffr" in keys, "Key 'erickson2020ffr' missing from refs.bib!"
    print("  [OK] ieeec37118 found in refs.bib")
    print("  [OK] erickson2020ffr found in refs.bib")

    with open(BIB_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    assert "C37.118" in content, "IEEE C37.118 not mentioned in refs.bib content!"
    assert "Fast frequency response" in content or "Fast Frequency Response" in content, "FFR standard text missing from refs.bib!"
    print("  [OK] Content checks for IEEE C37.118 and FFR passed.")

def test_doi_validity():
    print("\n=== Test 3: DOI Syntactic Validity ===")
    with open(BIB_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    doi_pattern = re.compile(r'doi\s*=\s*[\"{]([^\"}]+)[\"}]', re.IGNORECASE)
    dois = doi_pattern.findall(content)
    print(f"Found {len(dois)} DOI fields in refs.bib.")

    invalid_dois = []
    doi_regex = re.compile(r'^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$')
    for d in dois:
        d_clean = d.strip()
        if not doi_regex.match(d_clean):
            invalid_dois.append(d_clean)
        else:
            print(f"  Valid DOI: {d_clean}")

    assert len(invalid_dois) == 0, f"Found invalid DOIs: {invalid_dois}"
    print("All DOIs are syntactically valid!")

def test_citation_resolution():
    print("\n=== Test 4: Citation Resolution in manuscript.tex ===")
    with open(TEX_FILE, "r", encoding="utf-8") as f:
        tex = f.read()

    with open(BIB_FILE, "r", encoding="utf-8") as f:
        bib = f.read()

    entry_pattern = re.compile(r'@(\w+)\s*\{\s*([^,]+),')
    bib_keys = set(k.strip() for _, k in entry_pattern.findall(bib))

    cite_matches = re.finditer(r'\\cite\{([^}]+)\}', tex)
    cites = set()
    for m in cite_matches:
        for k in m.group(1).split(","):
            cites.add(k.strip())

    print(f"Total unique citations in manuscript.tex: {len(cites)}")
    print(f"Total available keys in refs.bib: {len(bib_keys)}")

    unresolved = cites - bib_keys
    print(f"Unresolved citations: {unresolved}")
    assert len(unresolved) == 0, f"Unresolved citations in manuscript.tex: {unresolved}"
    print("Zero unresolved citations in manuscript.tex!")

    # Check that IEEE C37.118 and FFR are cited in manuscript.tex
    assert "ieeec37118" in cites, "ieeec37118 is NOT cited in manuscript.tex!"
    assert "erickson2020ffr" in cites, "erickson2020ffr is NOT cited in manuscript.tex!"
    print("  [OK] ieeec37118 is actively cited in manuscript.tex")
    print("  [OK] erickson2020ffr is actively cited in manuscript.tex")

def test_cross_references():
    print("\n=== Test 5: Cross-Reference Consistency (Labels & Refs) ===")
    with open(TEX_FILE, "r", encoding="utf-8") as f:
        tex = f.read()

    labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
    refs = set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', tex))
    print(f"Total labels defined: {len(labels)}")
    print(f"Total cross-references called: {len(refs)}")

    missing_refs = refs - labels
    print(f"Missing cross-references: {missing_refs}")
    assert len(missing_refs) == 0, f"Dangling references detected: {missing_refs}"
    print("Zero dangling cross-references!")

def test_full_clean_rebuild():
    print("\n=== Test 6: Clean LaTeX & BibTeX Build Cycle ===")
    # Remove existing aux/bbl/blg to prove reproducibility from scratch
    for ext in [".aux", ".bbl", ".blg", ".log", ".spl"]:
        fpath = os.path.join(BASE_DIR, "manuscript" + ext)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception as e:
                print(f"Could not remove {fpath}: {e}")

    commands = [
        ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
        ["bibtex", "manuscript"],
        ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
        ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"]
    ]

    for step_idx, cmd in enumerate(commands, 1):
        print(f"Running step {step_idx}: {' '.join(cmd)}...")
        res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
        print(f"  Exit code: {res.returncode}")
        assert res.returncode == 0, f"Step {step_idx} failed with exit code {res.returncode}:\n{res.stdout[-1000:]}\n{res.stderr}"

    print("All 4 build passes completed with exit code 0!")

def test_build_artifacts():
    print("\n=== Test 7: Build Artifacts Integrity ===")
    # 1. Check aux file for citation{?}
    assert os.path.exists(AUX_FILE), f"Missing {AUX_FILE}"
    with open(AUX_FILE, "r", encoding="utf-8", errors="ignore") as f:
        aux = f.read()
    assert "citation{?}" not in aux, "Found undefined citation{?} in manuscript.aux!"
    print("  [OK] 0 undefined citations in manuscript.aux")

    # 2. Check log file for LaTeX errors
    assert os.path.exists(LOG_FILE), f"Missing {LOG_FILE}"
    with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
        log = f.read()

    errors = [line for line in log.splitlines() if line.startswith("!")]
    print(f"  LaTeX error lines count: {len(errors)}")
    assert len(errors) == 0, f"LaTeX errors detected in manuscript.log: {errors}"
    print("  [OK] 0 LaTeX errors in manuscript.log")

    # Check for undefined references in log
    undef_lines = [l for l in log.splitlines() if "undefined" in l.lower() and "reference" in l.lower()]
    assert len(undef_lines) == 0, f"Undefined references in log: {undef_lines}"
    print("  [OK] 0 undefined references in manuscript.log")

    # 3. Check blg file
    assert os.path.exists(BLG_FILE), f"Missing {BLG_FILE}"
    with open(BLG_FILE, "r", encoding="utf-8", errors="ignore") as f:
        blg = f.read()
    blg_errors = [l for l in blg.splitlines() if "error" in l.lower()]
    assert len(blg_errors) == 0, f"BibTeX errors detected: {blg_errors}"
    print("  [OK] 0 BibTeX errors in manuscript.blg")

    # 4. Check PDF exists and non-empty
    assert os.path.exists(PDF_FILE), f"Missing {PDF_FILE}"
    size = os.path.getsize(PDF_FILE)
    print(f"  PDF file size: {size} bytes ({size / (1024*1024):.2f} MB)")
    assert size > 500000, f"PDF file size {size} bytes is suspiciously small!"

    # Check page count
    with open(PDF_FILE, "rb") as f:
        pdf_bytes = f.read()
    pages = len(re.findall(rb'/Type\s*/Page[^s]', pdf_bytes))
    print(f"  PDF page count (approx/exact): {pages}")
    assert pages >= 20, f"PDF has too few pages ({pages})!"
    print(f"  [OK] PDF successfully generated with {pages} pages and {size} bytes")

def main():
    print("Starting Comprehensive Empirical Verification by Challenger Final 2...\n")
    keys = test_refs_bib_syntax_and_entries()
    test_standards_presence(keys)
    test_doi_validity()
    test_citation_resolution()
    test_cross_references()
    test_full_clean_rebuild()
    test_build_artifacts()
    print("\n==========================================")
    print("ALL EMPIRICAL VERIFICATION TESTS PASSED!")
    print("VERDICT: PASS")
    print("==========================================")

if __name__ == "__main__":
    main()
