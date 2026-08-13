"""Shared utilities for the Claude-lane model laboratories (Omnipose, micro-sam).

These helpers are framework-agnostic: prediction export, channel configuration,
and synthetic fixtures. Each laboratory keeps its own isolated environment and
imports these read-only helpers. Nothing here imports Omnipose, micro-sam, or
Cellpose, so it is safe to run in any environment (including cpenv, for tests).
"""
