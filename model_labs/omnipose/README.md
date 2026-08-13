# Omnipose laboratory

Highest-priority elongated-object architecture: its representation is designed to
reduce the center-seeking fragmentation that breaks long fibers (see the risk
register). It still requires myotube-specific ground truth (waits at G2).

- Environment: [`environment.yml`](environment.yml) — `pm-omnipose` (isolated).
- Channel config: [`channel_config.json`](channel_config.json) (`desmin_only` default; switch to `desmin_dapi` for the context experiment).
- Smoke test: [`smoke_test.py`](smoke_test.py) — `python model_labs/omnipose/smoke_test.py`.
- Resource note: [`resource_note.md`](resource_note.md).

Predictions export through `model_labs/_shared/predict_export.py` as unreviewed
`InstanceSet` JSON with provenance (M04 handoff). Replace the fallback predictor
in `smoke_test.py` with a real short train/infer once the pinned env exists.
