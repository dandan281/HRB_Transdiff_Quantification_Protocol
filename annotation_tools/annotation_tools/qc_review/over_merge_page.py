"""Blinded review page for candidate over-merges: "is this one myotube or two?"

Why a separate page
-------------------
`link_page.py` asks the operator to *propose* links and `junction_page.py` asks
which branches continue. This page asks a different, narrower question about an
already-applied decision: the linker merged these fragments at the locked
threshold, and two or more reviewed reference masks each claim >=20% of the
result. Only a human can say whether that is the linker over-merging or the
reference set having split one real myotube in two.

The decision vocabulary is therefore three-way and deliberately not the
linker's accept/reject:

``same_myotube``
    one physical myotube; the merge is correct and the reference masks
    disagree with each other, not with the linker.
``different_myotubes``
    two or more distinct myotubes were joined; a confirmed over-merge.
``ambiguous_2d``
    cannot be resolved in a single 2-D plane -- crossing or touching fibres
    where the answer needs z or a different stain. **Recorded as unresolved,
    never as evidence that the merge was safe.**

Blinding contract (enforced, not just documented)
------------------------------------------------
`build_over_merge_page` refuses to embed any field that would tell the reviewer
which cases are the real over-merges or what the model thought:
`_FORBIDDEN_KEYS`. The well id, the merged label, the link probability and the
case/control flag live only in the separate key file written by the caller. Cases
and controls are rendered by the identical code path with the identical panel
set, so an empty reference panel cannot be used to spot a control. Order is
shuffled by the caller.

The five panels the reviewer gets, per case:

1. **Desmin** -- raw fluorescence, no overlay at all. The ground of appeal.
2. **Fragments** -- the pre-merge pieces, one colour each.
3. **Proposed link** -- the bridge(s) the linker drew, endpoint to endpoint.
4. **Linked mask** -- the merged object as a single instance.
5. **Reference masks** -- every reviewed reference mask intersecting the crop,
   one colour each. These are the masks whose disagreement raised the flag.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

# distinct, colour-blind-safer hues; index 0..n reused across fragments/references
PALETTE = [
    (34, 211, 238),     # cyan
    (251, 191, 36),     # amber
    (167, 139, 250),    # violet
    (74, 222, 128),     # green
    (248, 113, 113),    # red
    (244, 114, 182),    # pink
    (125, 211, 252),    # sky
    (253, 224, 71),     # yellow
]
LINK_RGB = (255, 255, 255)
MERGED_RGB = (236, 72, 153)

DECISIONS = [
    ("same_myotube", "S", "one myotube — merge correct"),
    ("different_myotubes", "D", "distinct myotubes — confirmed over-merge"),
    ("ambiguous_2d", "A", "cannot resolve in 2D — unresolved"),
]
# A `different_myotubes` call is the finding that costs the linker something, so it has
# to carry a reason. Round 1 produced seven of them with zero notes, which left nothing
# to audit. Enforced as a soft block: the decision registers, but the case is not
# COMPLETE until a reason exists, and the export records the gap either way.
NOTE_REQUIRED_FOR = ["different_myotubes"]
# Bumped whenever the instrument changes what it records, so a scored report can never
# silently mix a round that carried telemetry with one that did not.
INSTRUMENT_VERSION = "over_merge_page/v2-telemetry"
PANELS = ["desmin", "fragments", "link", "linked", "references"]
PANEL_LABELS = {
    "desmin": "Desmin (no overlay)",
    "fragments": "original fragments",
    "link": "proposed link",
    "linked": "linked mask (one instance)",
    "references": "reference masks in view",
}

# Anything that would unblind the reviewer or leak the model's opinion.
_FORBIDDEN_KEYS = frozenset({
    "well", "merged_label", "probability", "accepted_pairs", "is_control",
    "control", "overlapping_references", "reference_id", "reference_ids",
    "fragment_ids", "prediction_area_px", "bbox", "case_kind",
})


def crop_window(shape, bbox, pad_px: int):
    """The crop the reviewer sees. Shared by the renderer and the blinding check so
    they can never disagree about what is in view."""
    return (max(0, bbox[0] - pad_px), max(0, bbox[1] - pad_px),
            min(shape[0], bbox[2] + pad_px), min(shape[1], bbox[3] + pad_px))


def count_references_in_view(references, shape, bbox, pad_px: int) -> int:
    """How many reference masks the reference panel will draw, without rendering.

    Needed for **control matching**: a real over-merge sits by construction where
    two reference masks meet, which tends to be a densely annotated neighbourhood.
    If controls are drawn from sparse neighbourhoods the reference panel alone
    separates the groups -- the reviewer can count outlines and never look at the
    biology. So controls have to be matched on this, not just on fragment count.
    """
    r0, c0, r1, c1 = crop_window(shape, bbox, pad_px)
    n = 0
    for ref in references:
        br0, bc0, br1, bc1 = ref["bbox"]
        if br1 <= r0 or br0 >= r1 or bc1 <= c0 or bc0 >= c1:
            continue
        rows = slice(max(br0, r0) - br0, min(br1, r1) - br0)
        cols = slice(max(bc0, c0) - bc0, min(bc1, c1) - bc0)
        if ref["mask"][rows, cols].any():
            n += 1
    return n


def _outline(mask: np.ndarray, width: int = 2) -> np.ndarray:
    if not mask.any():
        return mask
    return mask & ~ndi.binary_erosion(mask, iterations=max(1, width))


def _line(shape, start, end, thickness: int = 2) -> np.ndarray:
    """Rasterise a straight segment (the bridge the linker asserts)."""
    out = np.zeros(shape, dtype=bool)
    (r0, c0), (r1, c1) = start, end
    n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
    rows = np.round(np.linspace(r0, r1, n)).astype(int)
    cols = np.round(np.linspace(c0, c1, n)).astype(int)
    ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    out[rows[ok], cols[ok]] = True
    return ndi.binary_dilation(out, iterations=thickness) if out.any() else out


def _encode(rgb: np.ndarray, size: int) -> str:
    from PIL import Image

    img = Image.fromarray(rgb, mode="RGB")
    scale = size / max(img.size) if max(img.size) else 1.0
    if scale != 1:
        target = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(target, Image.LANCZOS if scale < 1 else Image.BICUBIC)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def render_case_panels(fiber, dapi, kept_only, merged, references, *,
                       fragment_ids, merged_label, links, bbox,
                       pad_px: int = 90, size: int = 900):
    """Render the five panels for one merged object.

    ``references`` is the full per-well list of reviewed reference masks; every
    one intersecting the crop is drawn, so the panel is built the same way for a
    real over-merge and for a control. ``links`` is a list of
    ``(endpoint_a, endpoint_b)`` full-field row/col pairs.
    """
    from .pipeline import _composite_native

    r0, c0, r1, c1 = crop_window(fiber.shape, bbox, pad_px)

    fiber_crop = fiber[r0:r1, c0:c1]
    dapi_crop = None if dapi is None else dapi[r0:r1, c0:c1]
    lo, hi = float(np.percentile(fiber, 1.0)), float(np.percentile(fiber, 99.5))
    dlo, dhi = ((float(np.percentile(dapi, 1.0)), float(np.percentile(dapi, 99.5)))
                if dapi is not None else (0.0, 1.0))
    clean = _composite_native(fiber_crop, dapi_crop, lo, hi, dlo, dhi)
    shape = clean.shape[:2]

    panels = {"desmin": _encode(clean, size)}

    frag_window = kept_only[r0:r1, c0:c1]
    arr = clean.copy()
    for n, fid in enumerate(fragment_ids):
        edge = _outline(frag_window == fid)
        arr[edge] = PALETTE[n % len(PALETTE)]
    panels["fragments"] = _encode(arr, size)

    arr = clean.copy()
    faint = _outline(frag_window > 0)
    arr[faint] = (arr[faint] * 0.45 + np.array([90, 100, 96]) * 0.55).astype(np.uint8)
    for (ra, ca), (rb, cb) in links:
        bridge = _line(shape, (ra - r0, ca - c0), (rb - r0, cb - c0))
        arr[bridge] = LINK_RGB
    panels["link"] = _encode(arr, size)

    arr = clean.copy()
    merged_window = merged[r0:r1, c0:c1] == merged_label
    arr[_outline(merged_window, width=3)] = MERGED_RGB
    panels["linked"] = _encode(arr, size)

    arr = clean.copy()
    n_refs = 0
    for ref in references:
        br0, bc0, br1, bc1 = ref["bbox"]
        if br1 <= r0 or br0 >= r1 or bc1 <= c0 or bc0 >= c1:
            continue
        full = np.zeros(fiber.shape, dtype=bool)
        full[br0:br1, bc0:bc1] = ref["mask"]
        window = full[r0:r1, c0:c1]
        if not window.any():
            continue
        arr[_outline(window, width=3)] = PALETTE[n_refs % len(PALETTE)]
        n_refs += 1
    panels["references"] = _encode(arr, size)

    return panels, (r0, c0, r1, c1), n_refs


def _check_blinded(cases: list[dict]) -> None:
    for case in cases:
        leaked = _FORBIDDEN_KEYS & set(case)
        if leaked:
            raise ValueError(
                f"case {case.get('uid', '?')} would unblind the reviewer: "
                f"{sorted(leaked)}. These belong in the key file only.")
        if "uid" not in case or "panels" not in case:
            raise ValueError("each case needs a uid and rendered panels")


def assert_no_separating_field(cases: list[dict], kinds: list[str]) -> dict:
    """Fail if any displayed scalar perfectly separates the flagged cases from the
    controls.

    Withholding the *label* is not enough. The first build of this packet passed
    `_check_blinded` and was still trivially unblinded, twice:

    * **by value** -- the number of reference masks in view was 7/7/4 for the three
      real cases and 0-3 for every control, so ranking on one number identified them
      all without looking at a myotube;
    * **by presence** -- the controls were extracted without `gap_um`, so the page
      showed a gap chip for the real cases and nothing for the controls. Every value
      was innocuous; having the field at all was the tell.

    Both rules below exist because of those. Absence of a forbidden key does not imply
    absence of a giveaway, so the packet builder (which legitimately holds the key)
    must run this before shipping.

    Returns the observed per-field ranges so the caller can report them.
    """
    if len(cases) != len(kinds):
        raise ValueError("one kind per case is required")
    flagged = {i for i, k in enumerate(kinds) if k != "control"}
    if not flagged or len(flagged) == len(cases):
        return {}
    keys = {k for case in cases for k in case if k not in ("uid", "panels")}
    report = {}
    for key in sorted(keys):
        # Presence is a signal in its own right. The second leak in this packet was a
        # `gaps_um` that only the real cases carried: the page renders it as a chip, so
        # "has a gap chip" separated the groups while every *value* looked innocuous.
        # An earlier version of this check skipped fields with missing values, which is
        # exactly how that got through.
        present = [bool(case.get(key) not in (None, [], {}, "")) for case in cases]
        a_present = {present[i] for i in range(len(cases)) if i in flagged}
        b_present = {present[i] for i in range(len(cases)) if i not in flagged}
        if len(a_present) == 1 and len(b_present) == 1 and a_present != b_present:
            raise ValueError(
                f"displayed field {key!r} is present for one group and absent for the "
                f"other (flagged present={a_present.pop()}): the packet is not blinded. "
                f"Give controls the same fields or stop displaying it.")

        values = []
        for case in cases:
            value = case.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float, list)):
                values.append(None)
            elif isinstance(value, list):
                values.append(max(value) if value else None)
            else:
                values.append(value)
        a = sorted(v for i, v in enumerate(values) if i in flagged and v is not None)
        b = sorted(v for i, v in enumerate(values) if i not in flagged and v is not None)
        if not a or not b:
            continue
        report[key] = {"flagged_range": [a[0], a[-1]], "control_range": [b[0], b[-1]]}
        if a[0] > b[-1] or a[-1] < b[0]:
            raise ValueError(
                f"displayed field {key!r} perfectly separates the flagged cases "
                f"{a} from the controls {b}: the packet is not blinded. Match the "
                f"controls on this field or stop displaying it.")
    return report


_SHORTCUTS = [
    ("1 - 5", "switch view: Desmin / fragments / link / linked / references"),
    ("L", "hide or show the outlines on the current view"),
    ("Left / Right", "previous / next case"),
    ("S", "same myotube — the merge is correct"),
    ("D", "different myotubes — confirmed over-merge"),
    ("A", "ambiguous in 2D — recorded as unresolved"),
    ("N", "write the reason (required for a 'different myotubes' call)"),
    ("0", "reset brightness and contrast"),
    ("Tab", "cycle panels forward"),
    ("J", "jump to the next case you have not decided"),
    ("?", "show or hide this list"),
]

_CSS = """
:root{
  --bg:#f4f7f5; --panel:#ffffff; --panel-2:#eef3f0; --border:#dae4de;
  --ink:#16221c; --muted:#5c6b63; --accent:#0d9488; --accent-ink:#ffffff;
  --accept:#1f9d4d; --reject:#d63a44; --ambig:#c08718;
  --shadow:0 1px 2px rgba(16,40,30,.06),0 4px 14px rgba(16,40,30,.05);
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0c120f; --panel:#121b16; --panel-2:#0f1712; --border:#22302a;
  --ink:#dbe6df; --muted:#8ba296; --accent:#2dd4bf; --accent-ink:#04241f;
  --accept:#39c46a; --reject:#f0616b; --ambig:#e0a83a;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 6px 20px rgba(0,0,0,.3);
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);height:100vh;overflow:hidden;
  font:14px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  display:flex;flex-direction:column}
header{background:var(--panel);border-bottom:1px solid var(--border);padding:10px 18px;
  display:flex;gap:16px;align-items:center;flex-wrap:wrap;box-shadow:var(--shadow)}
h1{font-size:15px;margin:0;font-weight:650;font-family:ui-monospace,Consolas,monospace}
.sub{color:var(--muted);font-size:12.5px}
.eyebrow{text-transform:uppercase;letter-spacing:.09em;font-size:10.5px;color:var(--muted);font-weight:600}
.spacer{margin-left:auto}
.pos{font-variant-numeric:tabular-nums;font-weight:700;font-size:15px}
.bar{height:6px;background:var(--panel-2);border-radius:999px;overflow:hidden;width:200px;
  border:1px solid var(--border)}
.bar>i{display:block;height:100%;background:var(--accent);width:0;transition:width .15s}
.chip{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;
  border:1px solid var(--border);font-size:12px;font-weight:600;background:var(--panel-2)}
kbd{font:600 11px ui-monospace,Consolas,monospace;background:var(--panel-2);
  border:1px solid var(--border);border-bottom-width:2px;border-radius:5px;padding:1px 6px;color:var(--ink)}

/* header / panelbar / footer never shrink; the image takes whatever is left. A fixed
   `calc(100vh - Npx)` on the image pushed the panel bar off-screen once the footer
   grew, which hid the only visible hint that other views exist. */
header,.panelbar,footer{flex:0 0 auto}
/* Two load-bearing details, both learned the hard way -- the image previously rendered
   at its natural size and painted over the view bar and the decision buttons:
   1. `align-items:stretch` (NOT center) so #stage inherits a *definite* height from
      main. Against an auto-height parent, `max-height:100%` on the image is ignored.
   2. `overflow:hidden` on main, so even if a future change breaks (1) the image is
      clipped instead of covering the controls. */
main{flex:1 1 auto;min-height:0;overflow:hidden;display:flex;align-items:stretch;
  justify-content:center;padding:8px 18px;background:var(--bg);position:relative}
#stage{position:relative;flex:1 1 auto;min-width:0;min-height:0;display:flex;
  align-items:center;justify-content:center;line-height:0}
#img{max-width:100%;max-height:100%;object-fit:contain;border-radius:12px;display:block;
  border:1px solid var(--border);box-shadow:var(--shadow);background:#060a09}
.overlaybadge{position:absolute;left:10px;top:10px;padding:3px 9px;border-radius:999px;
  font-size:11.5px;font-weight:700;letter-spacing:.03em;background:rgba(0,0,0,.62);color:#fff;
  pointer-events:none}
.navbtn{position:absolute;top:50%;transform:translateY(-50%);width:44px;height:64px;
  border-radius:10px;border:1px solid var(--border);background:var(--panel);color:var(--ink);
  font-size:20px;cursor:pointer;opacity:.75}
.navbtn:hover{opacity:1;border-color:var(--accent)}
#prev{left:16px} #next{right:16px}
.panelbar{display:flex;gap:6px;justify-content:center;padding:8px 18px 2px;background:var(--bg);
  flex-wrap:wrap;align-items:center}
.panelbar .hint{color:var(--muted);font-size:12px;margin-right:4px}
.panelbar button{font:inherit;font-size:12.5px;font-weight:600;cursor:pointer;border-radius:8px;
  padding:5px 11px;border:1px solid var(--border);background:var(--panel);color:var(--muted)}
.panelbar button.on{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}

footer{background:var(--panel);border-top:1px solid var(--border);padding:10px 18px 12px;
  box-shadow:var(--shadow)}
.statusline{font-size:13px;color:var(--muted);margin-bottom:6px}
.statusline b{font-size:14.5px;letter-spacing:.02em}
.decide{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px}
.decide button{font:inherit;font-weight:650;cursor:pointer;border-radius:9px;padding:9px 14px;
  border:1.5px solid var(--border);background:var(--panel-2);color:var(--ink);
  display:inline-flex;align-items:center;gap:8px}
.decide button:hover{border-color:var(--accent)}
.decide button.on{color:#04170d}
.decide button.on kbd{background:rgba(255,255,255,.75);border-color:rgba(0,0,0,.2);color:#04170d}
.decide .sep{width:1px;height:28px;background:var(--border);margin:0 4px}
.rowtools{display:flex;gap:10px;align-items:center;flex-wrap:wrap;color:var(--muted);font-size:12.5px}
.rowtools button{font:inherit;cursor:pointer;border-radius:8px;padding:5px 11px;
  border:1px solid var(--border);background:var(--panel);color:var(--ink)}
.rowtools button.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);font-weight:650}
input[type=text]{font:inherit;background:var(--panel-2);color:var(--ink);border:1px solid var(--border);
  border-radius:8px;padding:6px 10px;min-width:280px}

.kbmap{position:fixed;inset:0;z-index:60;display:none;align-items:center;justify-content:center;
  background:rgba(0,0,0,.55)}
.kbmap.open{display:flex}
.kbmap-inner{background:var(--panel);border:1px solid var(--border);border-radius:14px;
  padding:22px 26px;max-width:600px;width:92%;box-shadow:var(--shadow)}
.kbmap h3{margin:0 0 12px}
.kbmap table{border-collapse:collapse;width:100%;font-size:13.5px}
.kbmap td{padding:5px 8px;border-bottom:1px solid var(--border)}
.kbmap td:first-child{width:130px}
dialog{border:none;background:var(--panel);color:var(--ink);border-radius:14px;padding:0;max-width:94vw}
dialog::backdrop{background:rgba(0,0,0,.6)}
.dlg{padding:16px;display:flex;flex-direction:column;gap:10px}
textarea{width:min(90vw,760px);height:44vh;background:var(--panel-2);color:var(--ink);
  border:1px solid var(--border);border-radius:8px;padding:10px;
  font-family:ui-monospace,Consolas,monospace;font-size:12px}
"""

_JS = r"""
const state = {};
const telemetry = {};
DATA.cases.forEach(c => {
  state[c.uid] = {decision: null, note: "", t: null, at_decision: null};
  telemetry[c.uid] = {dwell_ms: {}, views: {}, total_ms: 0};
});
// Open on an overlay panel, not raw Desmin: the reviewer must see immediately that
// there ARE overlays. Starting on the clean image made the page look un-annotated.
const FIRST_OVERLAY = Math.max(0, DATA.PANELS.indexOf('fragments'));
let idx = 0, panel = FIRST_OVERLAY, bright = 1, contrast = 1;
let overlayVisible = true, lastOverlayPanel = FIRST_OVERLAY;

const cur = () => DATA.cases[idx];
const nowISO = () => new Date().toISOString().replace(/\.\d+Z$/, 'Z');

// ---------------------------------------------------------------- telemetry
// Round 1's verdicts arrived at a median of 6 s with no notes, and nothing in the
// export showed it -- the pace had to be reconstructed from decided_at gaps
// afterwards. So the instrument now records what was actually looked at, per case and
// per view, and ships it. Away-time is excluded (flush on blur, restart on focus).
let activeUid = null, activeShowing = null, activeSince = 0;
const shownPanel = () => overlayVisible ? DATA.PANELS[panel] : 'desmin';

function flush(){
  if(activeUid === null) return;
  const now = performance.now();
  const ms = Math.round(now - activeSince);
  const t = telemetry[activeUid];
  t.dwell_ms[activeShowing] = (t.dwell_ms[activeShowing] || 0) + ms;
  t.total_ms += ms;
  activeSince = now;
}

function retarget(){
  const uid = cur().uid, showing = shownPanel();
  if(activeUid === uid && activeShowing === showing) return;
  flush();
  activeUid = uid; activeShowing = showing; activeSince = performance.now();
  const t = telemetry[uid];
  t.views[showing] = (t.views[showing] || 0) + 1;
}

window.addEventListener('blur', flush);
window.addEventListener('focus', () => { activeSince = performance.now(); });

// ---------------------------------------------------------------- completeness
// A decision that still owes a reason is NOT complete. Round 1 exported seven
// `different_myotubes` calls with empty notes; there was nothing to audit.
const needsNote = s => !!s.decision && DATA.NOTE_REQUIRED_FOR.includes(s.decision)
                       && !(s.note || '').trim();
const isComplete = c => !!state[c.uid].t && !needsNote(state[c.uid]);
const decidedCount = () => DATA.cases.filter(isComplete).length;
const owingNotes = () => DATA.cases.filter(c => needsNote(state[c.uid]));

function renderHeader(){
  const n = DATA.cases.length, d = decidedCount();
  const tally = {};
  DATA.DECISIONS.forEach(x => tally[x[0]] = 0);
  DATA.cases.forEach(c => { const s = state[c.uid]; if(s.t && s.decision) tally[s.decision]++; });
  document.getElementById('caseId').textContent = cur().uid;
  document.getElementById('pos').textContent = `${idx+1} / ${n}`;
  document.getElementById('progress').style.width = (100*d/n) + '%';
  let h = '';
  DATA.DECISIONS.forEach(x => { h += `<span class="chip">${x[0].replace(/_/g,' ')} <b>${tally[x[0]]}</b></span>`; });
  h += `<span class="chip" ${d<n?'style="border-color:var(--ambig)"':''}>decided <b>${d}</b>/${n}</span>`;
  const owing = owingNotes().length;
  if(owing) h += `<span class="chip" style="border-color:var(--reject);color:var(--reject)">`+
    `<b>${owing}</b> awaiting a reason</span>`;
  document.getElementById('tally').innerHTML = h;
  const c = cur();
  // Deliberately NOT shown: the number of reference masks in view. It separated the
  // flagged cases from the controls in the first build of this packet.
  document.getElementById('caseInfo').innerHTML =
    `<span class="chip">${c.n_fragments} fragments joined</span>` +
    (c.gaps_um && c.gaps_um.length
      ? `<span class="chip">gap ${c.gaps_um.map(g=>g.toFixed(1)).join(', ')} &micro;m</span>` : '');
  document.getElementById('exportBtn').textContent = d < n ? `Export (${n-d} left)` : 'Export all';
}

function renderPanels(){
  const c = cur();
  let h = '<span class="hint">view:</span>';
  DATA.PANELS.forEach((p, i) => {
    const on = i === panel && (overlayVisible || p === 'desmin');
    h += `<button class="${on?'on':''}" onclick="setPanel(${i})">
      <kbd>${i+1}</kbd> ${DATA.PANEL_LABELS[p]}</button>`;
  });
  h += `<button class="${overlayVisible?'':'on'}" onclick="toggleOverlay()"
    title="hide or show the outlines"><kbd>L</kbd> ${
    overlayVisible ? 'hide outlines' : 'outlines hidden'}</button>`;
  document.getElementById('panelbar').innerHTML = h;
  const selected = DATA.PANELS[panel];
  const showing = overlayVisible ? selected : 'desmin';
  const img = document.getElementById('img');
  img.src = c.panels[showing];
  applyBC();
  document.getElementById('ovlBadge').textContent = overlayVisible
    ? DATA.PANEL_LABELS[selected]
    : `outlines hidden — L to show ${DATA.PANEL_LABELS[selected]}`;
}

// L toggles the outlines off and back, the same gesture as every other page in this
// tool. It was missing entirely in the first build, so the page looked as though it
// had no overlays at all.
function toggleOverlay(){
  if(DATA.PANELS[panel] === 'desmin'){
    panel = lastOverlayPanel;          // already on the clean image -> bring one back
    overlayVisible = true;
  } else {
    overlayVisible = !overlayVisible;
  }
  render();
}

// Desmin in these crops averages ~10/255 -- judging whether stain continues across a
// gap is not possible at native brightness, so this is a working control, not a frill.
function applyBC(){
  document.getElementById('img').style.filter = `brightness(${bright}) contrast(${contrast})`;
}
function setBC(){
  bright = parseFloat(document.getElementById('rBright').value);
  contrast = parseFloat(document.getElementById('rContrast').value);
  applyBC();
}
function resetBC(){
  bright = 1; contrast = 1;
  document.getElementById('rBright').value = 1;
  document.getElementById('rContrast').value = 1;
  applyBC();
}

function renderDecide(){
  const c = cur(), s = state[c.uid];
  let h = '';
  DATA.DECISIONS.forEach(x => {
    const on = s.decision === x[0];
    const colour = x[0]==='same_myotube' ? 'var(--accept)'
                 : x[0]==='different_myotubes' ? 'var(--reject)' : 'var(--ambig)';
    h += `<button class="${on?'on':''}" style="${on?`background:${colour};border-color:${colour}`:''}"
      onclick="decide('${x[0]}')"><kbd>${x[1]}</kbd> ${on?'&#10003; ':''}${x[2]}</button>`;
  });
  h += `<span class="sep"></span><button onclick="clearCase()">clear</button>`;
  document.getElementById('decide').innerHTML = h;
  const el = document.getElementById('status');
  if(s.decision){
    el.textContent = s.decision.replace(/_/g,' ').toUpperCase() +
      (s.decision==='ambiguous_2d' ? ' — recorded as UNRESOLVED, not as a safe merge' : '') +
      (needsNote(s) ? ' — NEEDS A REASON before it counts' : '');
    el.style.color = needsNote(s) ? 'var(--reject)'
                   : s.decision==='same_myotube' ? 'var(--accept)'
                   : s.decision==='different_myotubes' ? 'var(--reject)' : 'var(--ambig)';
  } else { el.textContent = 'not yet decided'; el.style.color = 'var(--muted)'; }
  const note = document.getElementById('note');
  note.value = s.note || '';
  note.placeholder = needsNote(s)
    ? 'required: what tells you these are separate myotubes?'
    : 'optional: what makes this hard?';
  note.style.borderColor = needsNote(s) ? 'var(--reject)' : 'var(--border)';
  const seen = !!telemetry[c.uid].views['references'];
  document.getElementById('refSeen').innerHTML = seen ? ''
    : `<span class="chip" style="border-color:var(--ambig)">reference masks not yet viewed `+
      `&mdash; <kbd>5</kbd></span>`;
}

function render(){ retarget(); renderHeader(); renderPanels(); renderDecide(); }
function setPanel(i){
  panel = i;
  if(DATA.PANELS[i] !== 'desmin'){ lastOverlayPanel = i; overlayVisible = true; }
  render();
}
function step(d){
  idx = (idx + d + DATA.cases.length) % DATA.cases.length;
  panel = lastOverlayPanel; overlayVisible = true;
  render();
}

function decide(kind){
  flush();                                  // the dwell up to this moment is the evidence
  const c = cur(), s = state[c.uid], t = telemetry[c.uid];
  s.decision = kind;
  s.t = nowISO();
  s.at_decision = {
    ms_on_case: t.total_ms,
    views: Object.assign({}, t.views),
    reference_panel_seen: !!t.views['references'],
    panels_seen: Object.keys(t.views),
  };
  render();
  // Asking for the reason at the moment of the call, not at export time.
  if(needsNote(s)) document.getElementById('note').focus();
}
function clearCase(){
  const s = state[cur().uid];
  s.decision = null; s.t = null; s.at_decision = null;
  render();
}
function saveNote(){ state[cur().uid].note = document.getElementById('note').value; renderHeader(); renderDecide(); }
function nextUndecided(){
  for(let k = 1; k <= DATA.cases.length; k++){
    const j = (idx + k) % DATA.cases.length;
    if(!isComplete(DATA.cases[j])){ idx = j; panel = lastOverlayPanel; overlayVisible = true; render(); return; }
  }
}

function exportAll(){
  flush();
  const n = DATA.cases.length, d = decidedCount(), owing = owingNotes();
  if(owing.length && !confirm(
      `${owing.length} "different myotubes" call(s) have no reason written. They export `+
      `flagged as note_missing and should not be counted as findings. Export anyway?`)) {
    idx = DATA.cases.indexOf(owing[0]); render();
    document.getElementById('note').focus();
    return;
  }
  if(d < n && !confirm(
      `${n - d} case(s) are not complete. They export with decision = null or `+
      `note_missing = true. Export anyway?`)) return;
  const payload = {
    batch_id: DATA.batch_id, reviewer: DATA.reviewer,
    session_started_at: DATA.session_started_at,
    exported_at: nowISO(),
    threshold: DATA.threshold,
    threshold_status: DATA.threshold_status,
    decision_vocabulary: DATA.DECISIONS.map(x => x[0]),
    note_required_for: DATA.NOTE_REQUIRED_FOR,
    instrument: DATA.instrument,
    blinded: true,
    completeness: {n_cases: n, n_complete: d, n_note_missing: owing.length},
    decisions: {},
  };
  DATA.cases.forEach(c => {
    const s = state[c.uid], t = telemetry[c.uid];
    payload.decisions[c.uid] = {
      decision: s.decision, decided_at: s.t, note: s.note || "",
      note_required: !!s.decision && DATA.NOTE_REQUIRED_FOR.includes(s.decision),
      note_missing: needsNote(s),
      ms_on_case: t.total_ms,
      panel_dwell_ms: t.dwell_ms,
      panel_views: t.views,
      reference_panel_seen_before_decision:
        s.at_decision ? s.at_decision.reference_panel_seen : null,
      ms_on_case_at_decision: s.at_decision ? s.at_decision.ms_on_case : null,
      panels_seen_at_decision: s.at_decision ? s.at_decision.panels_seen : null,
    };
  });
  document.getElementById('outText').value = JSON.stringify(payload, null, 2);
  document.getElementById('out').showModal();
}
function copyOut(){
  const t = document.getElementById('outText');
  t.select(); document.execCommand('copy');
  document.getElementById('copyBtn').textContent = 'Copied';
}
function downloadOut(){
  const blob = new Blob([document.getElementById('outText').value], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${DATA.batch_id}.over_merge_review.json`;
  a.click();
}

document.addEventListener('keydown', e => {
  if(e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA'){
    if(e.key === 'Escape') e.target.blur();
    return;
  }
  const k = e.key.toLowerCase();
  if(k === 'arrowright'){ step(1); }
  else if(k === 'arrowleft'){ step(-1); }
  else if(k === 'tab'){ e.preventDefault(); setPanel((panel+1) % DATA.PANELS.length); }
  else if(k >= '1' && k <= String(DATA.PANELS.length)){ setPanel(parseInt(k,10)-1); }
  else if(k === 'l'){ toggleOverlay(); }
  else if(k === 's'){ decide('same_myotube'); }
  else if(k === 'd'){ decide('different_myotubes'); }
  else if(k === 'a'){ decide('ambiguous_2d'); }
  else if(k === 'n'){ e.preventDefault(); document.getElementById('note').focus(); }
  else if(k === '0'){ resetBC(); }
  else if(k === 'j'){ nextUndecided(); }
  else if(k === '?'){ document.getElementById('kbmap').classList.toggle('open'); }
});

document.getElementById('stamp').textContent =
  `reviewer ${DATA.reviewer} · session ${DATA.session_started_at} · blinded, random order`;
document.getElementById('sub2').textContent =
  `${DATA.cases.length} merged objects · threshold ${DATA.threshold} (locked)`;
render();
"""


def build_over_merge_page(cases: list[dict], out_path: str | Path, *, batch_id: str,
                          reviewer: str, session_started_at: str, threshold: float,
                          note: str = "") -> str:
    """Write the blinded review page. Raises if a case would unblind the reviewer."""
    if not reviewer:
        raise ValueError("reviewer is required: an export without an identified "
                         "reviewer cannot be used as evidence")
    if not cases:
        raise ValueError("no cases to review")
    _check_blinded(cases)
    payload = {
        "cases": cases, "batch_id": batch_id, "reviewer": reviewer,
        "session_started_at": session_started_at, "threshold": threshold,
        "threshold_status": "LOCKED -- this review must not be used to tune it",
        "DECISIONS": DECISIONS, "PANELS": PANELS, "PANEL_LABELS": PANEL_LABELS,
        "NOTE_REQUIRED_FOR": NOTE_REQUIRED_FOR,
        "instrument": INSTRUMENT_VERSION,
    }
    rows = "".join(f"<tr><td><kbd>{k}</kbd></td><td>{v}</td></tr>" for k, v in _SHORTCUTS)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Over-merge review {batch_id}</title><style>{_CSS}</style></head><body>
<header>
  <div>
    <div class="eyebrow">over-merge review &middot; blinded</div>
    <h1 id="caseId"></h1>
    <div class="sub" id="stamp"></div>
  </div>
  <div class="spacer"></div>
  <div id="caseInfo" class="sub"></div>
  <div class="pos" id="pos"></div>
  <div class="bar"><i id="progress"></i></div>
  <div id="tally" class="sub"></div>
  <span class="chip" style="cursor:pointer"
    onclick="document.getElementById('kbmap').classList.add('open')"><kbd>?</kbd> shortcuts</span>
</header>

<div class="panelbar" id="panelbar"></div>
<main>
  <button class="navbtn" id="prev" onclick="step(-1)" title="previous (Left arrow)">&#8249;</button>
  <div id="stage">
    <img id="img" alt="">
    <span class="overlaybadge" id="ovlBadge"></span>
  </div>
  <button class="navbtn" id="next" onclick="step(1)" title="next (Right arrow)">&#8250;</button>
</main>

<footer>
  <div class="statusline">this merged object: <b id="status">not yet decided</b>
    <span id="refSeen"></span></div>
  <div class="decide" id="decide"></div>
  <div class="rowtools">
    <span>These fragments were joined into <b>one</b> instance. Is that one real myotube,
      or were separate myotubes joined? <b>Ambiguous is a real answer</b> &mdash; it is
      recorded as unresolved, not as agreement.</span>
  </div>
  <div class="rowtools" style="margin-top:8px">
    <span><kbd>N</kbd> reason</span>
    <input type="text" id="note" oninput="saveNote()" placeholder="optional: what makes this hard?">
    <span class="spacer"></span>
    <span>bright <input type="range" id="rBright" min="0.3" max="6" step="0.05" value="1"
      oninput="setBC()"></span>
    <span>contrast <input type="range" id="rContrast" min="0.3" max="3" step="0.05" value="1"
      oninput="setBC()"></span>
    <button onclick="resetBC()"><kbd>0</kbd> reset</button>
    <button onclick="nextUndecided()"><kbd>J</kbd> next undecided</button>
    <button class="primary" id="exportBtn" onclick="exportAll()">Export all</button>
  </div>
  {f'<div class="sub" style="margin-top:6px">{note}</div>' if note else ''}
</footer>

<div class="kbmap" id="kbmap" onclick="if(event.target.id==='kbmap')this.classList.remove('open')">
  <div class="kbmap-inner">
    <h3>Keyboard shortcuts</h3>
    <table>{rows}</table>
    <p class="sub" style="margin:12px 0 0" id="sub2"></p>
    <p class="sub" style="margin:8px 0 0">
      You are blinded: the well, the model's confidence and whether a case is a real
      flag or a control are not shown, and the order is random. Some objects here are
      ordinary merges included so your judgement can be calibrated.</p>
  </div>
</div>

<dialog id="out"><div class="dlg">
  <b>Over-merge review export</b>
  <span class="sub">Undecided cases export with <code>decision: null</code> rather than a
    guess. Ambiguous verdicts stay ambiguous &mdash; they are never counted as safe.</span>
  <textarea id="outText" readonly></textarea>
  <div><button class="primary" id="copyBtn" onclick="copyOut()">Copy all</button>
    <button onclick="downloadOut()">Download .json</button>
    <button onclick="document.getElementById('out').close()">Close</button></div>
</div></dialog>

<script>const DATA = {json.dumps(payload)};</script>
<script>{_JS}</script>
</body></html>"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)
