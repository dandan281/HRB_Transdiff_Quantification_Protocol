"""Oracle trace: run the tracer on PERFECT fields and measure the ceiling.

The direct analogue of the probe that cracked the Omnipose lane in one shot --
build the prediction *from the target* and see what the downstream code does
with it. No network, no training, CPU, seconds. If tracing from perfect
`centre` / `orient` / `crossing` fields cannot carry identity through a
crossing, no amount of training fixes it, and that is the branch that would
otherwise cost a month of GPU iterations.

The walk sees ONLY the three fields a network would predict. It never reads
``instance``, ``distance`` or the traces themselves -- those are reserved for
scoring. The walk is DeepBranchTracer's multi-feature loop (step on direction,
snap to the centreline ridge) with its one fatal rule replaced: DBT *stops* a
trace that touches an already-traced region, which on crossing fibres is a
false split at every contested junction. Here contact is split by angle:

* **transverse** contact (axial angle difference above ``--colinear-deg``) is a
  crossing -- pass through, both identities keep their pixels;
* **co-linear** contact sustained for ``--colinear-steps`` steps is the same
  fibre reached from another seed -- merge the two identities and stop.

Inside the ``crossing`` mask the orientation field is masked by construction
(two directions cannot share one angle-doubled vector), so the walk
dead-reckons: it keeps its incoming tangent until it exits the mask. That is
the executable form of `myotube-no-orthogonal-branching` -- an X is always two
objects passing through, so going straight is not a heuristic but the label.

Metrics, stated in polyline terms because both sides of the comparison are
polylines (the operator's traces ARE the ground truth here):

* identity through crossings -- of GT traces that touch a crossing, the
  fraction recovered as ONE object rather than several;
* ``length_mdape`` -- median |L_pred - L_gt| / L_gt over matched traces. The
  classical T03 floor is 0.3169 on `bootstrap_v1` (PLATE_23 masks); this
  number is the same quantity on PLATE_32 polylines, comparable in intent, not
  a substitute for the sealed benchmark run.
* false splits (one GT trace claimed by >=2 objects) and false merges (one
  object substantially covering >=2 GT traces), the polyline analogue of the
  T03 counts;
* trace recall -- fraction of GT traces with >=50% of their points covered.

    python model_labs/tracer_lab/oracle_trace.py --selftest
    python model_labs/tracer_lab/oracle_trace.py --well B02
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tracer_lab.centreline_targets import (  # noqa: E402
    TargetConfig, build_targets, resample_polyline, targets_from_roi_zip)


# ---------------------------------------------------------------------------
# the tracer: fields in, polylines out
# ---------------------------------------------------------------------------

class TraceParams:
    """Every knob of the walk, in one place so a run can record it.

    Values are starting points, not inherited truths -- the project rule is to
    sweep every knob on every new dataset. ``step_px`` is half the corpus fibre
    width (``width_px: 8.0``); myotube width barely varies, which is why there
    is no radius head and no radius-scaled step. ``seed_thresh`` 0.9 on a
    sigma=2 Gaussian centre is ~0.65 px off a centreline. ``support_thresh``
    0.5 is ~1.2 sigma -- past that the walk has lost the ridge.
    """

    def __init__(self, **kw):
        # Defaults are the plateau of the 2026-08-23 B02 oracle sweeps
        # (76 configs; scratchpad oracle_sweep*.log): identity 0.969,
        # mdape 0.084, splits 9, merges 34, with neighbouring configs within
        # a few counts. Swept on B02 GT only -- verify on other wells before
        # trusting further, and re-sweep for the trained-field case.
        self.step_px = 3.0
        self.seed_thresh = 0.90
        self.support_thresh = 0.30
        self.snap_lateral_px = 3.0
        self.snap_along_px = 1.5
        self.claim_radius_px = 1.5
        self.colinear_deg = 15.0
        self.colinear_steps = 2
        self.min_trace_px = 10.0
        self.max_steps = 200_000
        # Cap on seed CANDIDATES, kept brightest-first. On perfect fields the
        # threshold alone bounds this (~350k on B02); on predicted fields a
        # low seed threshold can admit millions of halo pixels and turn the
        # per-seed claim check into the whole runtime. Fibres own the
        # brightest pixels either way, so the cap costs nothing but time.
        self.max_seed_candidates = 500_000
        # Losing the ridge is not always the end of the fibre: a dead-reckoned
        # crossing exit lands slightly off-ridge, support fails, and the far
        # side gets re-traced by a fresh seed as a SECOND object -- a false
        # split manufactured by the walk. One rescue probe: search a short cone
        # along the incoming tangent for supported ridge whose orientation
        # agrees within `rescue_deg`; the angle gate keeps a genuine fibre end
        # from bridging onto a transverse neighbour.
        self.rescue_reach = 2.0     # in units of step_px
        self.rescue_lateral_px = 3.0
        self.rescue_deg = 30.0
        # Rescue only fires within this many steps of leaving a crossing.
        # Ungated, it bridges end-to-end fibre gaps just as readily as
        # crossing exits -- measured on B02: ~17 splits repaired at the cost
        # of ~17 fresh false merges, the old "fragments joined" error class.
        # Crossing exits are the failure rescue exists for; fibre ends are not.
        self.rescue_window_steps = 1
        # The follow gate: refuse to adopt a field direction that disagrees
        # with the incoming tangent by more than this (axial degrees).
        # Measured need on B02: a walk exiting a crossing can land on the
        # OTHER fibre's ridge -- support is good, no claim contact, and
        # field-following silently adopts the wrong identity (25 of 49 merged
        # pairs sat at 26-87 deg, reachable no other way). A genuine fibre
        # curves gently at the 15 px tangent scale; a ridge that turns >35 deg
        # in one 3-4 px step is someone else's fibre.
        self.follow_gate_deg = 25.0
        # Weak lateral snap INSIDE the crossing mask, clipped to this many px
        # per step. Tempting for the near-parallel overlap corridors that run
        # for hundreds of px (gt 46: 377 crossing px ON one trace), where
        # blind dead-reckoning drifts -- and it did buy 3 false splits on B02
        # at 1.0. But the selftest caught the price: a fibre that ENDS at a
        # junction gets dragged sideways onto the transverse fibre, tilting
        # the walk until the follow gate accepts the wrong ridge as its own.
        # Identity errors are worse than corridor splits, so this stays 0.
        self.crossing_snap_px = 0.0
        for k, v in kw.items():
            if not hasattr(self, k):
                raise TypeError(f"unknown param {k!r}")
            setattr(self, k, v)

    def to_dict(self):
        return {k: v for k, v in vars(self).items()}


def _axial_diff(a: float, b: float) -> float:
    """Smallest angle between two AXIAL directions (period pi), radians."""
    d = (a - b) % math.pi
    return min(d, math.pi - d)


def _field_dir(orient: np.ndarray, r: int, c: int) -> tuple[float, np.ndarray]:
    """Decode the angle-doubled field at a pixel -> (axial theta, unit vec)."""
    theta = 0.5 * math.atan2(float(orient[1, r, c]), float(orient[0, r, c]))
    return theta, np.array([math.sin(theta), math.cos(theta)])


def _walk_one_way(p0, v0, fields, claimed_id, claimed_theta, own_id, prm):
    """March from `p0` along `v0`. Returns (points, merge_with, reason).

    `points` excludes p0. `merge_with` is the id of a co-linear trace this walk
    merged into, else 0.
    """
    centre = fields["centre"]
    orient = fields["orient"]
    crossing = fields["crossing"]
    valid = fields["orient_valid"]
    H, W = centre.shape

    p = np.array(p0, dtype=np.float64)
    v = np.array(v0, dtype=np.float64)
    pts: list[np.ndarray] = []
    colinear_run = 0
    colinear_with = 0
    since_crossing = 10 ** 9

    for _ in range(prm.max_steps):
        ri, ci = int(round(p[0])), int(round(p[1]))
        since_crossing = 0 if crossing[ri, ci] else since_crossing + 1
        # direction: field where the field is trustworthy, dead-reckon where
        # it is masked (crossings) or unowned (off-fibre transients)
        if valid[ri, ci] and not crossing[ri, ci]:
            theta, u = _field_dir(orient, ri, ci)
            if float(u @ v) < 0.0:          # U-turn guard resolves the +/-
                u = -u
            # follow gate: a ridge that does not continue our direction is
            # another fibre's ridge -- dead-reckon rather than adopt it
            if _axial_diff(theta, math.atan2(float(v[0]), float(v[1]))) \
                    > math.radians(prm.follow_gate_deg):
                u = v
        else:
            u = v
        q = p + prm.step_px * u

        rq, cq = int(round(q[0])), int(round(q[1]))
        if not (0 <= rq < H and 0 <= cq < W):
            return pts, 0, "bounds"

        # lateral snap: pull q back onto the ridge, but only sideways --
        # sliding along-track would corrupt arc length. Inside the crossing
        # mask the ridge belongs to two fibres at once, so the correction is
        # clipped to `crossing_snap_px`: enough to survive a long overlap
        # corridor, too little to be dragged across a transverse fibre.
        if True:  # snap always runs; inside the mask it is clipped below
            n = np.array([-u[1], u[0]])
            r0, r1 = max(rq - 3, 0), min(rq + 4, H)
            c0, c1 = max(cq - 3, 0), min(cq + 4, W)
            win = centre[r0:r1, c0:c1]
            rr, cc = np.mgrid[r0:r1, c0:c1]
            dr, dc = rr - q[0], cc - q[1]
            along = dr * u[0] + dc * u[1]
            lat = dr * n[0] + dc * n[1]
            m = (np.abs(along) <= prm.snap_along_px) & \
                (np.abs(lat) <= prm.snap_lateral_px)
            wsum = float((win * m).sum())
            if wsum > 1e-9:
                off = float((win * m * lat).sum()) / wsum
                if crossing[rq, cq]:
                    off = float(np.clip(off, -prm.crossing_snap_px,
                                        prm.crossing_snap_px))
                q = q + off * n
                rq, cq = int(round(q[0])), int(round(q[1]))
                if not (0 <= rq < H and 0 <= cq < W):
                    return pts, 0, "bounds"

        if float(centre[rq, cq]) < prm.support_thresh:
            rescued = _rescue(p, u, fields, prm) \
                if since_crossing <= prm.rescue_window_steps else None
            if rescued is None:
                return pts, 0, "support"    # the point is NOT appended: prune
            q = rescued
            rq, cq = int(round(q[0])), int(round(q[1]))

        # contact with an already-traced identity: angle decides
        cid = int(claimed_id[rq, cq])
        if cid != 0 and cid != own_id:
            here = math.atan2(float(u[0]), float(u[1]))
            if _axial_diff(here, float(claimed_theta[rq, cq])) \
                    < math.radians(prm.colinear_deg):
                colinear_run += 1
                colinear_with = cid
                if colinear_run >= prm.colinear_steps:
                    # drop the contested tail: those points re-trace pixels the
                    # other identity already owns, and a merged object's length
                    # is the sum of its members -- leaving them in double-counts
                    # ~2 steps of arc per merge event
                    drop = colinear_run - 1
                    return (pts[:-drop] if drop else pts), colinear_with, "merge"
            else:
                colinear_run = 0            # transverse: a crossing, walk on
        else:
            colinear_run = 0

        pts.append(q.copy())
        v = (q - p) / max(float(np.linalg.norm(q - p)), 1e-9)
        p = q
    return pts, 0, "max_steps"


def _rescue(p, u, fields, prm):
    """Probe a cone along `u` for supported, orientation-consistent ridge.

    Returns the best landing point or None. Candidates must be outside the
    crossing mask with a defined orientation, so the probe re-acquires the
    SAME fibre on the far side of a junction rather than grabbing whatever is
    nearby.
    """
    centre = fields["centre"]
    orient = fields["orient"]
    crossing = fields["crossing"]
    valid = fields["orient_valid"]
    H, W = centre.shape
    n = np.array([-u[1], u[0]])
    best, best_c = None, prm.support_thresh
    for d in np.linspace(prm.step_px, prm.rescue_reach * prm.step_px, 4):
        for lat in np.linspace(-prm.rescue_lateral_px, prm.rescue_lateral_px, 7):
            q = p + d * u + lat * n
            r, c = int(round(q[0])), int(round(q[1]))
            if not (0 <= r < H and 0 <= c < W):
                continue
            if crossing[r, c] or not valid[r, c]:
                continue
            cv = float(centre[r, c])
            if cv <= best_c:
                continue
            theta, _ = _field_dir(orient, r, c)
            if _axial_diff(theta, math.atan2(float(u[0]), float(u[1]))) \
                    > math.radians(prm.rescue_deg):
                continue
            best, best_c = q, cv
    return best


def _claim(path, claimed_id, claimed_theta, own_id, prm):
    """Paint the walked path into the claim maps (free pixels only).

    First writer keeps a pixel: at a genuine crossing the second fibre passes
    through without repainting, so a later third contact still compares
    against a defined angle.

    The path is densified to ~1 px spacing first. Walk points are `step_px`
    apart, and discs of `claim_radius_px` around 4 px-spaced points leave
    unclaimed gaps along the fibre -- through which a second seed on the same
    fibre escapes suppression and re-traces it as a duplicate object. Measured
    on the selftest before this line existed: the bystander fibre came back
    twice.
    """
    H, W = claimed_id.shape
    path = resample_polyline(np.asarray(path), 1.0)
    r = int(math.ceil(prm.claim_radius_px))
    for i in range(len(path)):
        a = path[max(i - 1, 0)]
        b = path[min(i + 1, len(path) - 1)]
        theta = math.atan2(float(b[0] - a[0]), float(b[1] - a[1]))
        pr, pc = path[i]
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if dr * dr + dc * dc > prm.claim_radius_px ** 2:
                    continue
                rr, cc = int(round(pr)) + dr, int(round(pc)) + dc
                if 0 <= rr < H and 0 <= cc < W and claimed_id[rr, cc] == 0:
                    claimed_id[rr, cc] = own_id
                    claimed_theta[rr, cc] = theta


def trace_field(fields: dict, prm: TraceParams | None = None) -> dict:
    """Fields -> traced polylines with merged identities.

    Returns ``paths`` (list of (n,2) float arrays), ``object_of`` mapping each
    path index to a final object id after merge resolution, and diagnostics.
    """
    prm = prm or TraceParams()
    centre = fields["centre"]
    crossing = fields["crossing"]
    valid = fields["orient_valid"]
    orient = fields["orient"]
    H, W = centre.shape

    # seed candidates: on-ridge, orientation defined, outside crossings --
    # consumed brightest first. Every skipped seed costs one claim lookup.
    cand = np.argwhere((centre >= prm.seed_thresh) & valid & ~crossing)
    order = np.argsort(-centre[cand[:, 0], cand[:, 1]], kind="stable")
    cand = cand[order[:prm.max_seed_candidates]]

    claimed_id = np.zeros((H, W), dtype=np.int32)
    claimed_theta = np.zeros((H, W), dtype=np.float32)

    paths: list[np.ndarray] = []
    merges: list[tuple[int, int]] = []      # (path_id, absorbed into path_id)
    reasons: dict[str, int] = {}
    next_id = 1

    for r, c in cand:
        if claimed_id[r, c] != 0:
            continue
        _, u0 = _field_dir(orient, int(r), int(c))
        p0 = np.array([float(r), float(c)])

        fwd, m_f, reason_f = _walk_one_way(
            p0, u0, fields, claimed_id, claimed_theta, next_id, prm)
        bwd, m_b, reason_b = _walk_one_way(
            p0, -u0, fields, claimed_id, claimed_theta, next_id, prm)
        pts = [q for q in reversed(bwd)] + [p0] + fwd
        path = np.asarray(pts)
        seg = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) \
            if len(path) > 1 else 0.0
        if seg < prm.min_trace_px:
            continue

        paths.append(path)
        for m in (m_f, m_b):
            if m:
                merges.append((next_id, m))
        for rr in (reason_f, reason_b):
            reasons[rr] = reasons.get(rr, 0) + 1
        _claim(path, claimed_id, claimed_theta, next_id, prm)
        next_id += 1

    # union-find over merges: a fibre traced from three seeds is one object
    parent = list(range(next_id))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in merges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    object_of = {i + 1: find(i + 1) for i in range(len(paths))}
    return {"paths": paths, "object_of": object_of, "merge_events": merges,
            "stop_reasons": reasons, "params": prm.to_dict()}


def weld_objects(result: dict, fields: dict, *, weld_dist_px: float,
                 weld_deg: float = 15.0, crossing_gate_px: float = 12.0,
                 touch_px: float = 2.0) -> dict:
    """Post-walk merge of co-linear pieces that meet at a crossing.

    Motivated by the 2026-08-27 break-point attribution: breaks between the
    pieces of one cut fibre sit at predicted crossings 2-3x above base rate,
    and ~1/3 of them ABUT (the pieces touch or overlap; only the identity
    decision failed). The in-walk contact merge misses these because contact
    requires a LIVE walk stepping onto claimed pixels co-linearly for
    `colinear_steps` consecutive steps -- a walk that died in the junction, or
    a duplicate running just outside the claim band, never triggers it.

    A weld joins object A to object B when an ENDPOINT of one of A's paths
    lies within `weld_dist_px` of any point of B's paths and:

    * **tangents are co-linear** (axial difference <= `weld_deg`) -- same
      test the walk applies at contact;
    * **the connector runs along-track** (endpoint->landing direction within
      `weld_deg` of A's end tangent, tested only when the pieces do not
      already touch, i.e. distance > `touch_px`). This is the guard the
      2026-07 linker lacked: two PARALLEL fibres a few px apart are co-linear
      in tangent, but their connector is lateral -- rejected here;
    * **the site is at a crossing** (predicted crossing within
      `crossing_gate_px` of the endpoint). Ungated end-to-end joining across
      open field is the old fragments-joined error class and stays banned.

    The weld merges IDENTITIES ONLY -- no path is added, so a wrong weld
    costs a false merge but a correct one adds no fabricated arc length
    (unlike the refuted orientation bridging, whose probe paths inflated
    mdape as fast as reunification reduced it).

    Returns a new result dict with updated ``object_of`` and a
    ``weld_events`` list; the input is not mutated. ``weld_dist_px <= 0``
    returns the input unchanged (the frozen configuration).
    """
    if weld_dist_px <= 0:
        return result
    from scipy import ndimage
    from scipy.spatial import cKDTree

    paths = result["paths"]
    object_of = result["object_of"]
    crossing = fields["crossing"]
    H, W = crossing.shape
    r = max(int(round(crossing_gate_px)), 1)
    near_x = ndimage.binary_dilation(
        crossing, np.ones((2 * r + 1, 2 * r + 1), dtype=bool))

    # dense points of every path, with path id and local tangent
    all_pts, all_pid, all_theta = [], [], []
    for pid, path in enumerate(paths, start=1):
        d = resample_polyline(np.asarray(path), 1.0)
        if len(d) < 2:
            continue
        diffs = np.diff(d, axis=0)
        theta = np.arctan2(diffs[:, 0], diffs[:, 1])
        theta = np.append(theta, theta[-1])
        all_pts.append(d)
        all_pid.append(np.full(len(d), pid))
        all_theta.append(theta)
    if not all_pts:
        return result
    P = np.concatenate(all_pts)
    PID = np.concatenate(all_pid)
    TH = np.concatenate(all_theta)
    tree = cKDTree(P)

    welds: list[tuple[int, int, dict]] = []
    for pid, path in enumerate(paths, start=1):
        path = np.asarray(path)
        if len(path) < 2:
            continue
        for sign in (0, -1):
            e = path[sign]
            # end tangent from the last few points, pointing OUT of the path
            k = min(4, len(path) - 1)
            t = (path[-1] - path[-1 - k]) if sign == -1 else \
                (path[0] - path[k])
            tn = float(np.linalg.norm(t))
            if tn < 1e-6:
                continue
            t = t / tn
            er, ec = int(round(e[0])), int(round(e[1]))
            if not (0 <= er < H and 0 <= ec < W) or not near_x[er, ec]:
                continue
            own_obj = object_of[pid]
            best = None
            for i in tree.query_ball_point(e, r=weld_dist_px):
                if object_of.get(int(PID[i]), 0) == own_obj:
                    continue
                # co-linear tangents (axial)
                a_th = math.atan2(float(t[0]), float(t[1]))
                if _axial_diff(a_th, float(TH[i])) > math.radians(weld_deg):
                    continue
                d = P[i] - e
                dn = float(np.linalg.norm(d))
                if dn > touch_px:
                    # connector must run along-track, and forward
                    if float(d @ t) <= 0.0:
                        continue
                    c_th = math.atan2(float(d[0]), float(d[1]))
                    if _axial_diff(a_th, c_th) > math.radians(weld_deg):
                        continue
                if best is None or dn < best[0]:
                    best = (dn, int(PID[i]))
            if best is not None:
                welds.append((pid, best[1],
                              {"dist_px": round(best[0], 2),
                               "at": [er, ec]}))

    if not welds:
        return {**result, "weld_events": []}
    ids = sorted(set(object_of.values()))
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _info in welds:
        ra, rb = find(object_of[a]), find(object_of[b])
        if ra != rb:
            parent[ra] = rb
    new_of = {pid: find(oid) for pid, oid in object_of.items()}
    return {**result, "object_of": new_of,
            "weld_events": [(a, b, info) for a, b, info in welds]}


# ---------------------------------------------------------------------------
# scoring: traced objects vs the operator's polylines
# ---------------------------------------------------------------------------

def score_against_gt(result: dict, fields: dict, *,
                     cover_frac: float = 0.5, piece_frac: float = 0.10,
                     piece_px: float = 10.0, one_object_frac: float = 0.80,
                     length_mode: str = "sum"):
    """Polyline-level metrics. Uses `instance` as the point->GT lookup.

    A GT trace is *recovered* when >= `cover_frac` of its resampled points lie
    under some traced object. It is recovered *as one object* when the largest
    single object holds >= `one_object_frac` of its covered points. A *false
    split* is a GT trace substantially covered by >= 2 objects; a *false
    merge* is one object substantially covering >= 2 GT traces --
    "substantially" meaning >= `piece_frac` of the points and >= `piece_px`
    pixels, so a 3-pixel graze at a junction does not count as either.
    """
    instance = fields["instance"]
    crossing = fields["crossing"]
    gt = fields["traces"]
    H, W = instance.shape

    # length and crossing-contact per GT trace
    gt_len = np.array([float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum())
                       for t in gt])
    gt_touches = np.zeros(len(gt), dtype=bool)
    for i, t in enumerate(gt):
        r = np.clip(np.rint(t[:, 0]).astype(int), 0, H - 1)
        c = np.clip(np.rint(t[:, 1]).astype(int), 0, W - 1)
        gt_touches[i] = bool(crossing[r, c].any())

    # Object length. Two definitions, because they differ by a factor of two
    # on predicted fields and the difference is not the tracer's geometry:
    #
    # "sum"   -- add up the member paths' arc lengths. Correct when each part
    #            of a fibre is walked once, but a fibre walked twice (two
    #            seeds inside a fuzzy band, merged into one object) reports
    #            double its true length even when both walks lie on it.
    # "union" -- stamp the member paths at 1 px and count distinct pixels. A
    #            1 px-wide curve of length L covers ~L pixels, so duplicate
    #            coverage collapses and complementary halves still add.
    #
    # `sum` stays the default. `union` was added on the hypothesis that
    # duplicate walks inside a fuzzy band were inflating length by double
    # counting -- MEASURED and REFUTED the same hour: on D04, net_v2/raw goes
    # 22.55 -> 21.91 and net_v6/raw 2.69 -> 2.59. The over-length is real
    # geometry (walks wandering off-axis and re-entering), not an artefact of
    # how member paths are added up. Kept as an option, and as the record of
    # a wrong guess; changing a metric definition for a 3% effect would have
    # broken comparability with every number reported before it for nothing.
    obj_len: dict[int, float] = {}
    if length_mode == "sum":
        for pid, path in enumerate(result["paths"], start=1):
            oid = result["object_of"][pid]
            seg = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            obj_len[oid] = obj_len.get(oid, 0.0) + seg
    else:
        px: dict[int, set] = {}
        for pid, path in enumerate(result["paths"], start=1):
            oid = result["object_of"][pid]
            d = resample_polyline(np.asarray(path), 1.0)
            r = np.clip(np.rint(d[:, 0]).astype(int), 0, H - 1)
            c = np.clip(np.rint(d[:, 1]).astype(int), 0, W - 1)
            px.setdefault(oid, set()).update(zip(r.tolist(), c.tolist()))
        obj_len = {oid: float(len(s)) for oid, s in px.items()}

    # votes: object -> GT trace, one vote per ~1 px of traced path. Walk
    # points are `step_px` apart while GT points are resampled at 1 px, so
    # votes must be counted at the same density or coverage under-reads 4x.
    votes: dict[int, dict[int, int]] = {}
    pts_of_obj: dict[int, int] = {}
    for pid, path in enumerate(result["paths"], start=1):
        oid = result["object_of"][pid]
        dense = resample_polyline(np.asarray(path), 1.0)
        r = np.clip(np.rint(dense[:, 0]).astype(int), 0, H - 1)
        c = np.clip(np.rint(dense[:, 1]).astype(int), 0, W - 1)
        ids = instance[r, c]
        pts_of_obj[oid] = pts_of_obj.get(oid, 0) + len(dense)
        v = votes.setdefault(oid, {})
        for g in ids[ids > 0]:
            v[int(g)] = v.get(int(g), 0) + 1

    # per-GT coverage: which objects hold its points
    covered_by: dict[int, dict[int, int]] = {g + 1: {} for g in range(len(gt))}
    for oid, v in votes.items():
        for g, n in v.items():
            covered_by[g][oid] = covered_by[g].get(oid, 0) + n

    n_gt = len(gt)
    recovered = np.zeros(n_gt, dtype=bool)
    as_one = np.zeros(n_gt, dtype=bool)
    split = np.zeros(n_gt, dtype=bool)
    ape: list[float] = []

    for g in range(n_gt):
        n_pts = len(gt[g])
        by = covered_by[g + 1]
        total_cov = sum(by.values())
        if total_cov == 0:
            continue
        big = max(by.values())
        recovered[g] = total_cov >= cover_frac * n_pts
        as_one[g] = recovered[g] and big >= one_object_frac * total_cov
        pieces = [o for o, n in by.items()
                  if n >= piece_frac * n_pts and n >= piece_px]
        split[g] = len(pieces) >= 2
        if recovered[g]:
            oid = max(by, key=by.get)
            ape.append(abs(obj_len[oid] - gt_len[g]) / max(gt_len[g], 1e-9))

    n_merge = 0
    for oid, v in votes.items():
        substantial = [g for g, n in v.items()
                       if n >= piece_frac * len(gt[g - 1]) and n >= piece_px]
        if len(substantial) >= 2:
            n_merge += 1

    xing = gt_touches.sum()
    out = {
        "n_gt": int(n_gt),
        "n_objects": int(len(set(result["object_of"].values()))),
        "n_paths": int(len(result["paths"])),
        "recall_traces": float(recovered.mean()) if n_gt else 0.0,
        "length_mdape": float(np.median(ape)) if ape else float("nan"),
        "n_length_matched": int(len(ape)),
        "false_split_count": int(split.sum()),
        "false_merge_count": int(n_merge),
        "n_gt_touching_crossing": int(xing),
        "identity_through_crossing":
            float(as_one[gt_touches].mean()) if xing else float("nan"),
        "identity_all": float(as_one.mean()) if n_gt else 0.0,
        "stop_reasons": result["stop_reasons"],
        "merge_events": int(len(result["merge_events"])),
    }
    return out


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def _selftest() -> int:
    """An X of two straight fibres plus one bystander. The oracle in miniature:
    both arms of the X must come back as ONE object each, sharing the junction.
    """
    n = 200
    a = np.column_stack([np.linspace(20, 180, n), np.linspace(20, 180, n)])
    b = np.column_stack([np.linspace(180, 20, n), np.linspace(20, 180, n)])
    c = np.column_stack([np.full(n, 30.0), np.linspace(30, 150, n)])
    fields = build_targets((200, 200), [a, b, c])
    res = trace_field(fields)
    sc = score_against_gt(res, fields)
    print(json.dumps(sc, indent=2))
    ok = (sc["false_split_count"] == 0 and sc["false_merge_count"] == 0
          and sc["identity_through_crossing"] == 1.0
          and sc["recall_traces"] == 1.0 and sc["length_mdape"] < 0.05)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true",
                    help="run on a synthetic X + bystander instead of a well")
    ap.add_argument("--plate-dir", default="Q_PLATES/Q_Plates/PLATE_32")
    ap.add_argument("--well", default="B02")
    ap.add_argument("--shape", type=int, default=3636)
    ap.add_argument("--out", default="model_labs/tracer_lab/_runs/oracle")
    # None = use the swept TraceParams default. A concrete CLI default here
    # would silently override the plateau config on every plain run -- which
    # it did, once: three wells got scored on stale knobs.
    ap.add_argument("--step", type=float, default=None)
    ap.add_argument("--colinear-deg", type=float, default=None)
    ap.add_argument("--support", type=float, default=None)
    a = ap.parse_args(argv)

    if a.selftest:
        return _selftest()

    zips = sorted((ROOT / a.plate_dir).glob(f"*{a.well}*ROIs.zip")) or \
        sorted((ROOT / a.plate_dir).glob(f"*{a.well}*.zip"))
    if not zips:
        print(f"no ROI zip for {a.well} under {a.plate_dir}", file=sys.stderr)
        return 1

    t0 = time.time()
    fields = targets_from_roi_zip(zips[0], (a.shape, a.shape))
    t1 = time.time()
    overrides = {k: v for k, v in
                 (("step_px", a.step), ("colinear_deg", a.colinear_deg),
                  ("support_thresh", a.support)) if v is not None}
    prm = TraceParams(**overrides)
    res = trace_field(fields, prm)
    t2 = time.time()
    sc = score_against_gt(res, fields)
    t3 = time.time()

    sc["timing_s"] = {"targets": round(t1 - t0, 2), "trace": round(t2 - t1, 2),
                      "score": round(t3 - t2, 2)}
    sc["well"] = a.well
    sc["roi_zip"] = zips[0].name
    sc["params"] = prm.to_dict()

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rec = out / f"oracle_{a.well}.json"
    rec.write_text(json.dumps(sc, indent=2))
    print(json.dumps(sc, indent=2))
    print(f"\nwritten: {rec}")
    print(f"classical floor for context: length_mdape 0.3169 "
          f"(bootstrap_v1, different GT -- comparable in intent only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
