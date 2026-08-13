# micro-sam laboratory

Microscopy-tuned SAM. Provides both the interactive annotation UI used by CL01
and an automatic (prompt-free) segmentation mode. Only the **automatic** mode
enters the bake-off (M05) — interactive usefulness does not prove automatic
accuracy.

- Environment: [`environment.yml`](environment.yml) — `pm-microsam` (isolated).
- Channel config: [`channel_config.json`](channel_config.json).
- Smoke test: [`smoke_test.py`](smoke_test.py) — `python model_labs/microsam/smoke_test.py`.
- Resource note: [`resource_note.md`](resource_note.md).

Official test predictions use **no expert prompts** (`ModelProvenance.used_prompts=False`).
Exports go through the shared exporter as unreviewed `InstanceSet` JSON with
provenance.
