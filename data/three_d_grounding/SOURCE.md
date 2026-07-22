# Data provenance

The seven images and camera-intrinsics files used by this suite are copied
unchanged from the RynnBrain authors' public repository:

- Repository: https://github.com/alibaba-damo-academy/RynnBrain
- Commit: `eef38357e48bb6b8a1358aade3054f9a16b1cbb1`
- Original path: `cookbooks/assets/3d_grounding/images/`

The prompt template and the three recorded SUN RGB-D generations are replayed
from `cookbooks/8_3d_grounding.ipynb` at the same commit. The recorded
generations are consistency references, not ground-truth 3D boxes. Five
additional calibrated image/category pairs exercise output format, physical
constraints, and center projection on all eight GPUs.
