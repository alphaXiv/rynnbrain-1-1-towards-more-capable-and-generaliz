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
