"""Deterministic tests for the read-only Tier-A conversion audit.

Two tiers:
- **pure** tests (always run) pin the transcribed algorithm and the hashing on
  synthetic inputs — reproduction, pooled-threshold, area gate, ring readout;
- **integration** tests (skipped when the Conversion_Efficiency caches are absent)
  re-assert the full six-well reproduction, the plate-wide operating point, the
  C08 nucleus reconciliation, and that the audit never writes into the read-only
  Conversion_Efficiency tree.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tier_a_audit import audit


ARTIFACTS = (audit.DBS_DIR.is_dir() and audit.NUC_DIR.is_dir()
             and audit.DECLARED_JSON.is_file())
needs_artifacts = pytest.mark.skipif(
    not ARTIFACTS, reason="Conversion_Efficiency caches not present")


# ------------------------------------------------------------------ pure


def test_ring_px_is_fifteen_at_declared_pixel_size():
    # round(10 um / 0.6493 um/px) = 15
    assert audit.ring_px() == 15


def test_pooled_log_otsu_splits_a_bimodal_distribution():
    rng = np.random.default_rng(0)
    low = rng.normal(100, 15, 4000).clip(1, None)     # negative cells
    high = rng.normal(1500, 200, 2000).clip(1, None)  # positive cells
    thr = audit.pooled_log_otsu(np.concatenate([low, high]))
    assert 100 < thr < 1500                            # sits in the valley
    # determinism
    assert thr == audit.pooled_log_otsu(np.concatenate([low, high]))


def test_pooled_threshold_is_from_the_pool_not_per_well():
    """The shared threshold must come from the concatenated data, so it need not
    equal either well's own Otsu — that is what 'no per-well tuning' means."""
    from skimage.filters import threshold_otsu
    a = np.full(500, 120.0)
    b = np.full(500, 1400.0)
    pooled = audit.pooled_log_otsu(np.concatenate([a, b]))
    per_a = 10 ** threshold_otsu(np.log10(np.maximum(a, 1.0)))
    assert pooled != per_a                             # not a per-well value


def test_valid_by_area_gate_is_50_to_500_um2():
    # area_um2 = px * UM2; UM2 ~ 0.4216, so [50,500] um2 ~ [119, 1186] px
    nuc = np.zeros((200, 200), dtype=np.int32)
    nuc[:5, :20] = 1      # 100 px -> ~42 um2  (too small)
    nuc[10:30, 10:30] = 2  # 400 px -> ~169 um2 (valid)
    nuc[50:90, 50:130] = 3  # 3200 px -> ~1349 um2 (too big)
    valid = audit.valid_by_area(nuc)
    assert not valid[1] and valid[2] and not valid[3]
    assert not valid[0]                                # background never valid


def test_ring_intensity_reads_the_cytoplasmic_ring():
    nuc = np.zeros((60, 60), dtype=np.int32)
    nuc[28:32, 28:32] = 1
    dbs = np.zeros((60, 60), dtype=np.float32)
    dbs[24:36, 24:36] = 500.0                          # bright halo around the nucleus
    mean, cnt = audit.ring_intensity(nuc, dbs, rp=3)
    assert cnt[1] > 0
    assert mean[1] == pytest.approx(500.0, abs=1.0)    # ring sits in the bright halo


def test_classify_requires_positive_valid_and_ring():
    nuc = np.zeros((60, 60), dtype=np.int32)
    nuc[26:34, 26:34] = 1                              # 64 px -> ~27 um2, below 50 -> invalid
    dbs = np.full((60, 60), 900.0, dtype=np.float32)
    pos, valid = audit.classify(nuc, dbs, rp=3, thr=440.0)
    assert not valid[1] and not pos[1]                 # invalid by area -> not positive


def test_sha256_is_deterministic(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"conversion-audit")
    assert audit.sha256_file(p) == audit.sha256_file(p)
    assert audit.sha256_bytes(b"conversion-audit") == audit.sha256_file(p)


def test_manifest_key_is_repo_relative_for_declared_inputs():
    assert audit.manifest_key(audit.DECLARED_JSON) == \
        "Conversion_Efficiency/New_Quantif_P23/visualize_final.json"
    assert "\\" not in audit.manifest_key(audit.NUC_DIR / "23_B02_ctrl_masks.npy")


def test_manifest_key_falls_back_to_absolute_out_of_tree():
    """`out_dir` is caller-chosen and may sit outside the repo (pytest's default
    `--basetemp` does exactly that). Keying it must not raise -- previously
    `relative_to(ROOT)` did, taking the whole audit down."""
    # out of tree by construction: the filesystem anchor is never under ROOT
    outside = Path(audit.ROOT.anchor) / "not_in_the_repo" / "reproduced.json"
    assert not outside.is_relative_to(audit.ROOT)
    assert audit.manifest_key(outside) == outside.as_posix()


def test_superseded_and_diagnostics_are_kept_separate():
    d = audit.method_distinction({"reproduced": {"per_well": {
        "32_C08_br223_igf1r": {"desmin_pos": 3341, "valid": 10114, "conversion_pct": 33.03}}}})
    assert d["declared_ring_method"]["conversion_pct"] == 33.03
    assert d["superseded_traced_fiber"]["conversion_pct"] == 6.6245
    assert d["superseded_traced_fiber"]["n_converted"] == 670
    assert any("k-sweep" in s for s in d["diagnostic_sweeps_not_operating_points"])


# ------------------------------------------------------------------ integration


@needs_artifacts
def test_reproduces_declared_visualize_final_for_all_six_wells():
    repro = audit.reproduce_all()
    assert repro["declared_present"] and repro["all_match"]
    assert repro["reproduced"]["threshold_raw"] == pytest.approx(440.77, abs=0.5)
    c08 = repro["reproduced"]["per_well"]["32_C08_br223_igf1r"]
    assert c08["desmin_pos"] == 3341 and c08["valid"] == 10114
    assert c08["conversion_pct"] == 33.03


@needs_artifacts
def test_one_plate_wide_operating_point():
    repro = audit.reproduce_all()
    checks = audit.verify_plate_wide(repro)
    assert checks["passed"]
    assert checks["declared_threshold_method_is_pooled_uniform"]
    assert checks["no_per_well_threshold_field"]


@needs_artifacts
def test_c08_nucleus_reconciliation_is_proven():
    rec = audit.reconcile_c08_nuclei()
    assert rec["proven"]
    assert rec["mask_arrays_identical"]                # same Cellpose run
    assert rec["counts"]["area_50_500um2_canonical"] == 10114
    assert rec["counts"]["total_labels"] == 10588
    # MyoFuse >=30px floor is within a couple nuclei of its reported 10,560
    assert 0 <= rec["counts"]["ge_30px_myofuse_floor"] - 10560 <= 20


@needs_artifacts
def test_audit_does_not_write_into_conversion_efficiency(tmp_path):
    """Read-only guard: running the audit must not touch Conversion_Efficiency."""
    watched = [audit.DECLARED_JSON,
               audit.NUC_DIR / "32_C08_br223_igf1r_masks.npy",
               audit.DBS_DIR / "23_B02_ctrl_dbs.npy"]
    before = {p: p.stat().st_mtime_ns for p in watched if p.is_file()}
    audit.run_audit(tmp_path / "audit_out")
    after = {p: p.stat().st_mtime_ns for p in watched if p.is_file()}
    assert before == after                             # nothing under CE was written
    assert (tmp_path / "audit_out" / "audit_manifest.json").is_file()


@needs_artifacts
def test_full_audit_overall_pass(tmp_path):
    result = audit.run_audit(tmp_path / "out")
    assert result["overall_pass"]
    assert result["reproduction"]["all_wells_match_declared"]
    assert result["c08_nucleus_reconciliation"]["proven"]
    # manifest carries source-image + mask + cache + script hashes
    m = result["manifest_sha256"]
    assert all(v for v in m["nucleus_masks"].values())
    assert all(v for v in m["source_images_nd2"].values())
