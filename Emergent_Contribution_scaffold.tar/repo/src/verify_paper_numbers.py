#!/usr/bin/env python3
"""
verify_paper_numbers.py — Path A reproducibility check.

Asserts every number reported in the paper against the committed canonical
artifacts in results/. Prints one PASS/FAIL line per check and exits non-zero
if any check fails. Requires no GPU and runs in seconds.

Usage:
    python src/verify_paper_numbers.py --root results/ [--verbose]
"""
import argparse, sys
from verifier_core import run_checks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results",
                    help="folder containing the canonical artifacts")
    ap.add_argument("--verbose", action="store_true",
                    help="print every check, not just failures")
    args = ap.parse_args()

    npass, nfail, nerr, lines = run_checks(args.root, verbose=args.verbose)
    for ln in lines:
        print(ln)
    print(f"\n{npass} passed, {nfail} failed, {nerr} errored "
          f"(of {npass + nfail + nerr} checks).")
    if nfail == 0 and nerr == 0:
        print("ALL CHECKS PASSED — paper numbers match the committed artifacts.")
        sys.exit(0)
    else:
        print("SOME CHECKS DID NOT PASS — see lines above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
