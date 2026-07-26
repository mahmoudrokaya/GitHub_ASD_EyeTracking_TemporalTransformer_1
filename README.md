# Improving Autism Diagnosis Across Ages Using Eye-Tracking and Temporal Transformer Models

Reproducible code, experiment definitions, metadata, selected derived files, and supporting materials for autism diagnosis using eye-tracking time-series data and Temporal Transformer models.

## Experiments

- `experiment_1`: 21 files
- `experiment_2`: 6 files
- `experiment_3`: 12 files
- `experiment_4`: 10 files
- `experiment_5`: 2 files

## Structure

- `experiments/`: experiment-specific code
- `metadata/`: permitted participant/project metadata
- `docs/`: documentation
- `results/`: optional derived outputs
- `data/`: optional redistributable data only
- `CITATION.cff`: GitHub citation metadata
- `.zenodo.json`: Zenodo metadata
- `codemeta.json`: software metadata
- `MANIFEST.csv`: included-file provenance
- `SHA256SUMS.txt`: integrity checksums

## Installation

```bash
python -m venv .venv
pip install -r requirements.txt
```

Review input/output path constants before running the original scripts. Some scripts may contain absolute Windows paths.

## DOI workflow

1. Review the generated repository and remove restricted or identifying files.
2. Upload the folder to GitHub.
3. Connect the GitHub repository to Zenodo.
4. Create a GitHub release such as `v1.0.0`.
5. Zenodo archives the release and issues a DOI.
6. Add the DOI to `CITATION.cff`, `.zenodo.json`, and this README in a later release.

## Authors

- Mohammed A. AlZain, Taif University, Saudi Arabia
- Mahmoud Rokaya, Taif University, Saudi Arabia
- Dalia I. Hemdan, Taif University, Saudi Arabia
- Ibrahim Gad, Tanta University, Egypt
- Malik Almaliki, Taibah University, Saudi Arabia
- El-Sayed Atlam, Taibah University, Saudi Arabia

## License and data

Software code is licensed under MIT unless a file states otherwise. Dataset rights are separate; see `DATA_LICENSE.md` and `DATA_AVAILABILITY.md`.

Included files: 61  
Excluded files: 133  
Build date: 2026-07-26
