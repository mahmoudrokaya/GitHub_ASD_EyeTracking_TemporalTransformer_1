# Reproducibility Guide

1. Create a Python 3.10+ environment.
2. Install `requirements.txt` or use `environment.yml`.
3. Obtain datasets according to `DATA_AVAILABILITY.md`.
4. Review and replace machine-specific absolute paths in the original scripts.
5. Run preprocessing before training where required.
6. Record all random seeds, hardware, CUDA, package versions, and deterministic settings.
7. Compare outputs using `MANIFEST.csv` and `SHA256SUMS.txt`.

Repository version: 1.0.0
Build date: 2026-07-26
