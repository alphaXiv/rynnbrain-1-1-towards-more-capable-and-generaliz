# RynnBrain 1.1 contact-point reproduction

This repository contains a deterministic, eight-GPU reproduction of the
contact-point prediction examples released with RynnBrain 1.1. The benchmark
runs one official cookbook case per GPU and reports format validity plus
agreement with the authors' recorded generations.

The fixed experiment command is:

```bash
bash scripts/run_contact_point_eval.sh
```

Model selection is committed in `config.json`; experiment branches change the
configuration while keeping the command and Kubernetes environment fixed.

## Public reproduction package

The completed run-backed study is published as:

- [Detailed Markdown report](reports/rynnbrain-reproduction/report.md), including five evidence figures
- [Self-contained marimo notebook](notebooks/rynnbrain_reproduction.py)
- [Open the notebook in Molab](https://molab.marimo.io/github/alphaXiv/rynnbrain-1-1-towards-more-capable-and-generaliz/blob/main/notebooks/rynnbrain_reproduction.py)

The verdict is **partially reproduced**: RynnBrain 1.1 reliably follows the
released embodied interfaces and exhibits partial visual-spatial and temporal
conditioning, but exact homogeneous-camera controls show that its native-3D
focal response is not coherent projective geometry. All numeric claims come
from successful terminal Kubernetes logs; cancelled and failed runs are
excluded.
