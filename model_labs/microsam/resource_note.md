# micro-sam laboratory — resource note (CL03.5)

Record measured values after the first real run on the pinned environment.
Planning estimates for a 3636×3636, 16-bit field until then.

| Item | Planned / measured |
|---|---|
| GPU | Blackwell cu128 — **separate** from cpenv and from the Omnipose env |
| Backbone | ViT-b/l SAM variant — record which |
| Tile size | model-native patch tiling; overlap ≥ 64 px |
| Automatic vs interactive | **official candidate = automatic, no prompts** (M05) |
| Runtime (auto infer, full field) | _measure_ |
| Peak GPU memory | _measure_ |
| Peak host memory | _measure_ |

**Prompt separation (CL03/M05):** interactive (prompted) performance may inform
annotation efficiency (CL01) but is reported separately; the bake-off candidate
uses no expert prompts. `ModelProvenance.used_prompts` records this per run.

**Tiling rule:** as in the Omnipose note — never silently cut a validation
object; log any object split across seams.
