"""T02 candidate 2 harness (Omnipose).

Why this package is not called ``omnipose``
-------------------------------------------
It would shadow the installed Omnipose library. ``model_labs`` sits on
``PYTHONPATH``, so in the `pm-omnipose` environment a package at
``model_labs/omnipose/`` competes with site-packages for the name ``omnipose``.
That resolves inconsistently -- and if ours ever won, `cellpose_omni`'s own
``import omnipose`` would receive this package instead of the library, which
would fail in a confusing place rather than at import time.

``model_labs/omnipose/`` therefore keeps exactly what the development plan and
the T02 request already reference by path -- `environment.yml`, `verify_env.py`,
`README.md`, `resource_note.md`, `smoke_test.py`, `channel_config.json` -- and all
new T02 harness code lives here. Nothing plan-referenced moved.

`verify_env.py` is loaded from that directory by path; see :mod:`omnipose_lab.env`.
"""
