# contrastive-reranking-for-sparse-archival-retrieval

This repository contains the complete implementation of the paper
**Contrastive Reranking for Sparse Archival Retrieval**, accepted at Journal of Archival Data Science 2026.

## Components

- `src/` — reference implementation used to produce the published results
- `requirements.txt` — pinned dependencies
- `scripts/` — command line entry points

## Reproducing the results

```
pip install -r requirements.txt
python scripts/run_experiment.py --config configs/default.json
```

The released release tag `v1.0.0` corresponds to the camera-ready artefact.
