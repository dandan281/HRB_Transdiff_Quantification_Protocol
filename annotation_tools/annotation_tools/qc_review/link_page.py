"""Fragment-linking page: "does this fragment join that one?".

Interaction model deliberately mirrors `page.py`, the tool the operator already
used for 1,800 decisions: a **focused one-at-a-time viewer**, decide-and-advance,
arrow-key navigation, an ``i / N`` position counter, brightness/contrast on the
up/down arrows, and a ``?`` shortcut overlay. Same theme variables, so it looks
and feels like the same instrument.

Every control is reachable from the keyboard, and every control is a **toggle** --
pressing a candidate's letter labels it, pressing it again unlabels it. Nothing
is decided implicitly.

Multi-select is required, not a convenience: a fragment can be the *middle* of a
fibre and link at both ends, so a single-choice control would silently lose half
of those joins.

Every card also records the candidates that were **offered and declined**. Those
are the linker's negatives -- a training set of only positives would teach a model
to join everything.

Provenance matches the R04 contract: reviewer identity, session start, export
time, per-decision UTC timestamp. A card never explicitly decided exports
``decided_at: null`` and is dropped downstream.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

FRAGMENT_RGB = (255, 74, 170)                       # magenta, as elsewhere in the lane
CANDIDATE_RGB = [
    ("A", (34, 211, 238)),                          # cyan
    ("B", (251, 191, 36)),                          # amber
    ("C", (167, 139, 250)),                         # violet
    ("D", (52, 211, 153)),                          # green
    ("E", (248, 113, 113)),                         # red
]

_CSS = """
:root{
  --bg:#f4f7f5; --panel:#ffffff; --panel-2:#eef3f0; --border:#dae4de;
  --ink:#16221c; --muted:#5c6b63; --accent:#0d9488; --accent-ink:#ffffff;
  --accept:#1f9d4d; --reject:#d63a44; --ambig:#c08718;
  --accept-bg:#e6f6ec; --reject-bg:#fceaeb; --ambig-bg:#f9f0dc;
  --shadow:0 1px 2px rgba(16,40,30,.06),0 4px 14px rgba(16,40,30,.05);
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0c120f; --panel:#121b16; --panel-2:#0f1712; --border:#22302a;
  --ink:#dbe6df; --muted:#8ba296; --accent:#2dd4bf; --accent-ink:#04241f;
  --accept:#39c46a; --reject:#f0616b; --ambig:#e0a83a;
  --accept-bg:#12291b; --reject-bg:#2a1417; --ambig-bg:#291f0f;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 6px 20px rgba(0,0,0,.3);
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);height:100vh;overflow:hidden;
  font:14px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  display:flex;flex-direction:column}
header{background:var(--panel);border-bottom:1px solid var(--border);padding:10px 18px;
  display:flex;gap:16px;align-items:center;flex-wrap:wrap;box-shadow:var(--shadow)}
h1{font-size:15px;margin:0;font-weight:650}
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

main{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;
  padding:14px 18px;background:var(--bg);position:relative}
#stage{position:relative;display:inline-block;line-height:0}
#img,#ovl{max-width:100%;max-height:calc(100vh - 260px);border-radius:12px;display:block}
#img{border:1px solid var(--border);box-shadow:var(--shadow);background:#060a09}
/* the outline layer sits exactly on the picture and is purely decorative */
#ovl{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;
  image-rendering:pixelated}
#ovl.hidden{display:none}
.overlaybadge{position:absolute;left:10px;top:10px;padding:3px 9px;border-radius:999px;
  font-size:11.5px;font-weight:700;letter-spacing:.03em;background:rgba(0,0,0,.62);color:#fff;
  pointer-events:none}
.navbtn{position:absolute;top:50%;transform:translateY(-50%);width:44px;height:64px;
  border-radius:10px;border:1px solid var(--border);background:var(--panel);color:var(--ink);
  font-size:20px;cursor:pointer;opacity:.75}
.navbtn:hover{opacity:1;border-color:var(--accent)}
#prev{left:16px} #next{right:16px}

footer{background:var(--panel);border-top:1px solid var(--border);padding:10px 18px 12px;
  box-shadow:var(--shadow)}
.statusline{font-size:13px;color:var(--muted);margin-bottom:6px}
.statusline b{font-size:14.5px;letter-spacing:.02em}
.decide{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px}
.decide button{font:inherit;font-weight:650;cursor:pointer;border-radius:9px;padding:9px 14px;
  border:1.5px solid var(--border);background:var(--panel-2);color:var(--ink);
  display:inline-flex;align-items:center;gap:8px}
.decide button:hover{border-color:var(--accent)}
.decide button .swatch{width:12px;height:12px;border-radius:3px;border:1px solid rgba(0,0,0,.25)}
.decide button.on{color:#04170d}
.decide button.on kbd{background:rgba(255,255,255,.75);border-color:rgba(0,0,0,.2);color:#04170d}
.decide .sep{width:1px;height:28px;background:var(--border);margin:0 4px}
.rowtools{display:flex;gap:10px;align-items:center;flex-wrap:wrap;color:var(--muted);font-size:12.5px}
.rowtools button{font:inherit;cursor:pointer;border-radius:8px;padding:5px 11px;
  border:1px solid var(--border);background:var(--panel);color:var(--ink)}
.rowtools button.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);font-weight:650}
input[type=range]{width:110px;vertical-align:middle}

.kbmap{position:fixed;inset:0;z-index:60;display:none;align-items:center;justify-content:center;
  background:rgba(0,0,0,.55)}
.kbmap.open{display:flex}
.kbmap-inner{background:var(--panel);border:1px solid var(--border);border-radius:14px;
  padding:22px 26px;max-width:560px;width:92%;box-shadow:var(--shadow)}
.kbmap h3{margin:0 0 12px}
.kbmap table{border-collapse:collapse;width:100%;font-size:13.5px}
.kbmap td{padding:5px 8px;border-bottom:1px solid var(--border)}
.kbmap td:first-child{width:120px}
dialog{border:none;background:var(--panel);color:var(--ink);border-radius:14px;padding:0;max-width:94vw}
dialog::backdrop{background:rgba(0,0,0,.6)}
.dlg{padding:16px;display:flex;flex-direction:column;gap:10px}
textarea{width:min(90vw,760px);height:44vh;background:var(--panel-2);color:var(--ink);
  border:1px solid var(--border);border-radius:8px;padding:10px;
  font-family:ui-monospace,Consolas,monospace;font-size:12px}
"""

_JS = r"""
const state = {};
DATA.cases.forEach(c => state[c.uid] = {picked: [], none: false, unsure: false, t: null});
let idx = 0;                       // index into DATA.cases
let bright = 1, contrast = 1;
let overlayVisible = true;         // L hides the outlines to inspect the raw stain

const cur = () => DATA.cases[idx];
const nowISO = () => new Date().toISOString().replace(/\.\d+Z$/, 'Z');
const decidedCount = () => DATA.cases.filter(c => state[c.uid].t).length;

function renderHeader(){
  const n = DATA.cases.length, d = decidedCount();
  let links = 0, none = 0, unsure = 0;
  DATA.cases.forEach(c => { const s = state[c.uid]; if(!s.t) return;
    links += s.picked.length; if(s.none) none++; if(s.unsure) unsure++; });
  document.getElementById('pos').textContent = `${idx+1} / ${n}`;
  document.getElementById('progress').style.width = (100*d/n) + '%';
  document.getElementById('tally').innerHTML =
    `<span class="chip">links <b>${links}</b></span>`+
    `<span class="chip">no join <b>${none}</b></span>`+
    `<span class="chip">unsure <b>${unsure}</b></span>`+
    `<span class="chip" ${d<n?'style="border-color:var(--ambig)"':''}>decided <b>${d}</b>/${n}</span>`;
  document.getElementById('exportBtn').textContent =
    d < n ? `Export (${n-d} left)` : 'Export all';
}

function renderDecide(){
  const c = cur(), s = state[c.uid];
  let h = '';
  c.candidates.forEach(k => {
    const on = s.picked.includes(k.letter);
    h += `<button data-k="${k.letter}" class="${on?'on':''}"
      style="${on?`background:rgb(${k.rgb});border-color:rgb(${k.rgb})`:''}"
      onclick="toggle('${k.letter}')">
      <span class="swatch" style="background:rgb(${k.rgb})"></span>
      <kbd>${k.letter}</kbd> ${on ? '&#10003; LINKED' : 'link'} &middot; ${k.gap_um} µm</button>`;
  });
  h += `<span class="sep"></span>`;
  h += `<button data-k="__none" class="${s.none?'on':''}"
    style="${s.none?'background:var(--reject);border-color:var(--reject)':''}"
    onclick="toggle('__none')"><kbd>N</kbd> ${s.none?'&#10003; no join':'no join'}</button>`;
  h += `<button data-k="__unsure" class="${s.unsure?'on':''}"
    style="${s.unsure?'background:var(--ambig);border-color:var(--ambig)':''}"
    onclick="toggle('__unsure')"><kbd>U</kbd> ${s.unsure?'&#10003; unsure':'unsure'}</button>`;
  h += `<span class="sep"></span>`;
  h += `<button onclick="clearCase()"><kbd>C</kbd> clear</button>`;
  h += `<button onclick="confirmNext()"><kbd>Enter</kbd> confirm &amp; next</button>`;
  document.getElementById('decide').innerHTML = h;

  // Plain-language echo of the current state, so a toggle is never ambiguous.
  let status, colour;
  if(s.none){ status = 'NO JOIN'; colour = 'var(--reject)'; }
  else if(s.unsure){ status = 'UNSURE'; colour = 'var(--ambig)'; }
  else if(s.picked.length){
    status = 'LINKED TO ' + s.picked.slice().sort().map(L =>
      (c.candidates.find(x => x.letter === L) || {}).candidate_id || L).join(' + ');
    colour = 'var(--accept)';
  } else { status = s.t ? 'nothing selected' : 'not yet decided'; colour = 'var(--muted)'; }
  const el = document.getElementById('status');
  el.textContent = status;
  el.style.color = colour;
}

function renderOverlay(){
  const ovl = document.getElementById('ovl');
  ovl.classList.toggle('hidden', !overlayVisible);
  const badge = document.getElementById('ovlBadge');
  badge.textContent = overlayVisible ? '' : 'outlines hidden — L to show';
  badge.style.display = overlayVisible ? 'none' : '';
}
function toggleOverlay(){ overlayVisible = !overlayVisible; renderOverlay(); }

function render(){
  const c = cur(), s = state[c.uid];
  const img = document.getElementById('img'), ovl = document.getElementById('ovl');
  // Only reload the images when the case actually changes: re-assigning a 140 KB
  // data URL on every toggle makes the picture flash, which reads as "my click
  // did nothing".
  if(img.dataset.uid !== c.uid){
    img.src = c.img; ovl.src = c.overlay; img.dataset.uid = c.uid;
  }
  // brightness/contrast is a view aid on the fluorescence only, never the outlines
  img.style.filter = `brightness(${bright}) contrast(${contrast})`;
  renderOverlay();
  document.getElementById('caseId').textContent = `${c.well} · fragment ${c.id}`;
  document.getElementById('caseInfo').innerHTML =
    c.candidates.map(k => `<span class="chip"><span class="swatch" style="display:inline-block;
      width:10px;height:10px;border-radius:2px;background:rgb(${k.rgb})"></span>
      ${k.letter} = ${k.candidate_id} · ${k.gap_um} µm</span>`).join(' ');
  document.getElementById('stamp').textContent = s.t ? `decided ${s.t}` : 'not yet decided';
  renderDecide(); renderHeader();
}

function toggle(key){
  const s = state[cur().uid];
  if(key === '__none'){ s.none = !s.none; if(s.none){ s.picked = []; s.unsure = false; } }
  else if(key === '__unsure'){ s.unsure = !s.unsure; if(s.unsure){ s.picked = []; s.none = false; } }
  else {
    const i = s.picked.indexOf(key);
    if(i >= 0) s.picked.splice(i, 1);              // pressing again UNLABELS
    else { s.picked.push(key); s.none = false; s.unsure = false; }
  }
  s.t = nowISO();
  render(); save();
}

function clearCase(){
  // Full unlabel: back to untouched, including the decision stamp.
  state[cur().uid] = {picked: [], none: false, unsure: false, t: null};
  render(); save();
}

function step(d){ idx = (idx + d + DATA.cases.length) % DATA.cases.length; render(); }
function confirmNext(){
  const s = state[cur().uid];
  if(!s.t) s.t = nowISO();                          // Enter alone confirms as-is
  save(); step(1);
}
function nextUndecided(){
  for(let k = 1; k <= DATA.cases.length; k++){
    const j = (idx + k) % DATA.cases.length;
    if(!state[DATA.cases[j].uid].t){ idx = j; render(); return; }
  }
  alert('Every fragment has been decided.');
}
function nudge(which, d){
  if(which === 'b') bright = Math.min(3.5, Math.max(0.3, bright + d));
  else contrast = Math.min(3.0, Math.max(0.3, contrast + d));
  document.getElementById('rBright').value = bright;
  document.getElementById('rContrast').value = contrast;
  document.getElementById('img').style.filter = `brightness(${bright}) contrast(${contrast})`;
}
function setBC(){
  bright = +document.getElementById('rBright').value;
  contrast = +document.getElementById('rContrast').value;
  document.getElementById('img').style.filter = `brightness(${bright}) contrast(${contrast})`;
}
function resetBC(){ bright = 1; contrast = 1; nudge('b', 0); }

const KEY = 'links_' + DATA.batch_id;
function save(){ try{ localStorage.setItem(KEY, JSON.stringify(state)); }catch(_){ } }
function restore(){
  try{ const raw = localStorage.getItem(KEY); if(!raw) return;
    const saved = JSON.parse(raw);
    Object.keys(saved).forEach(u => { if(state[u]) state[u] = saved[u]; });
  }catch(_){ }
}

document.addEventListener('keydown', e => {
  // Only a text field may swallow shortcuts. The brightness/contrast sliders are
  // INPUTs too, and bailing on those meant one drag of a slider silently killed
  // every shortcut for the rest of the session.
  const t = e.target;
  if(t && (t.tagName === 'TEXTAREA' ||
           (t.tagName === 'INPUT' && t.type !== 'range'))) return;
  const km = document.getElementById('kbmap');
  if(e.key === '?'){ e.preventDefault(); km.classList.toggle('open'); return; }
  if(km.classList.contains('open')){ if(e.key === 'Escape') km.classList.remove('open'); return; }
  if(document.getElementById('out').open) return;
  const k = e.key.toUpperCase();
  if(e.key === 'ArrowRight'){ e.preventDefault(); step(1); }
  else if(e.key === 'ArrowLeft'){ e.preventDefault(); step(-1); }
  else if(e.key === 'ArrowUp'){ e.preventDefault(); nudge('b', 0.15); }
  else if(e.key === 'ArrowDown'){ e.preventDefault(); nudge('b', -0.15); }
  else if(e.key === 'Enter'){ e.preventDefault(); confirmNext(); }
  else if(k === 'N'){ toggle('__none'); step(1); }
  else if(k === 'U'){ toggle('__unsure'); step(1); }
  else if(k === 'J'){ nextUndecided(); }
  else if(k === 'C'){ clearCase(); }
  else if(k === 'L'){ toggleOverlay(); }
  else if(k === '0'){ resetBC(); }
  else if(cur().candidates.some(x => x.letter === k)) toggle(k);
  // Sliders keep focus after a drag; blur so Space/Enter never nudge them.
  if(document.activeElement && document.activeElement.tagName === 'INPUT')
    document.activeElement.blur();
});

function exportAll(){
  const n = DATA.cases.length, d = decidedCount();
  if(d < n && !confirm(`${n-d} of ${n} fragments were never decided.\n\n`+
     `They export with decided_at = null and are dropped. Export anyway?`)) return;
  const payload = {schema:'fragment_links.v1', batch_id:DATA.batch_id,
    reviewer:DATA.reviewer, session_started_at:DATA.session_started_at,
    exported_at:nowISO(), n_cases:n, n_explicitly_decided:d,
    gap_um:DATA.gap_um, cos_min:DATA.cos_min, decisions:{}};
  DATA.cases.forEach(c => {
    const s = state[c.uid];
    payload.decisions[c.uid] = {
      well:c.well, fragment_id:c.id, decided_at:s.t,
      // accepted AND declined are recorded: declined candidates are the
      // linker's negatives.
      linked_to: c.candidates.filter(k => s.picked.includes(k.letter)).map(k => k.candidate_id),
      declined:  c.candidates.filter(k => !s.picked.includes(k.letter)).map(k => k.candidate_id),
      offered:   c.candidates.map(k => ({id:k.candidate_id, gap_um:k.gap_um,
                                         cos_fragment:k.cos_fragment,
                                         cos_candidate:k.cos_candidate})),
      no_join: !!s.none, unsure: !!s.unsure};
  });
  document.getElementById('outText').value = JSON.stringify(payload, null, 1);
  document.getElementById('out').showModal();
}
function copyOut(){ const t=document.getElementById('outText'); t.select();
  document.execCommand('copy'); document.getElementById('copyBtn').textContent='Copied'; }
function downloadOut(){
  const b=new Blob([document.getElementById('outText').value],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(b);
  a.download=DATA.batch_id+'.links.json'; a.click();
}

restore(); render();
"""


def _outline(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, iterations=2)


def render_case(fiber: np.ndarray, dapi, labels: np.ndarray, fragment_label: int,
                candidate_labels: list[int], size: int = 900,
                margin: int = 46) -> tuple[str, str, tuple[int, int, int, int]]:
    """Render the crop as two separate layers.

    Returns ``(clean_jpeg, outline_png, bbox)``:

    * **clean** -- the raw two-channel fluorescence composite only: green Desmin,
      purple/blue DAPI, nothing drawn on it;
    * **outline** -- a transparent RGBA layer holding *only* the mask outlines.

    They are kept apart on purpose. Burning the outlines into the picture makes
    it impossible to hide them, and judging whether a broken fibre truly
    continues across a gap requires seeing the unobscured stain. The page stacks
    the two and the ``L`` key hides the outline layer, matching the overlay
    toggle in `page.py`.
    """
    from PIL import Image

    from .pipeline import _composite_native

    masks = [labels == fragment_label] + [labels == c for c in candidate_labels]
    union = np.zeros(labels.shape, dtype=bool)
    for m in masks:
        union |= m
    rows, cols = np.nonzero(union)
    r0 = max(0, int(rows.min()) - margin); r1 = min(labels.shape[0], int(rows.max()) + margin)
    c0 = max(0, int(cols.min()) - margin); c1 = min(labels.shape[1], int(cols.max()) + margin)

    fiber_crop = fiber[r0:r1, c0:c1]
    dapi_crop = None if dapi is None else dapi[r0:r1, c0:c1]
    lo, hi = float(np.percentile(fiber, 1.0)), float(np.percentile(fiber, 99.5))
    dlo, dhi = ((float(np.percentile(dapi, 1.0)), float(np.percentile(dapi, 99.5)))
                if dapi is not None else (0.0, 1.0))
    clean = _composite_native(fiber_crop, dapi_crop, lo, hi, dlo, dhi)

    # outlines on a fully transparent canvas
    label_crop = labels[r0:r1, c0:c1]
    overlay = np.zeros((clean.shape[0], clean.shape[1], 4), dtype=np.uint8)
    fragment_edge = _outline(label_crop == fragment_label)
    overlay[fragment_edge, :3] = FRAGMENT_RGB
    overlay[fragment_edge, 3] = 255
    for i, label_id in enumerate(candidate_labels):
        edge = _outline(label_crop == label_id)
        overlay[edge, :3] = CANDIDATE_RGB[i % len(CANDIDATE_RGB)][1]
        overlay[edge, 3] = 255

    base = Image.fromarray(clean, mode="RGB")
    layer = Image.fromarray(overlay, mode="RGBA")
    scale = size / max(base.size)
    if scale != 1:
        target = (max(1, round(base.width * scale)), max(1, round(base.height * scale)))
        base = base.resize(target, Image.LANCZOS if scale < 1 else Image.BICUBIC)
        # NEAREST keeps the outline crisp and fully opaque rather than fading it
        layer = layer.resize(target, Image.NEAREST)

    clean_buffer = io.BytesIO()
    base.save(clean_buffer, format="JPEG", quality=88)
    layer_buffer = io.BytesIO()
    layer.save(layer_buffer, format="PNG", optimize=True)
    return ("data:image/jpeg;base64," + base64.b64encode(clean_buffer.getvalue()).decode("ascii"),
            "data:image/png;base64," + base64.b64encode(layer_buffer.getvalue()).decode("ascii"),
            (r0, c0, r1, c1))


_SHORTCUTS = [
    ("L", "<b>hide / show the outlines</b> &mdash; see the bare green Desmin and "
          "purple nuclei with nothing drawn on top"),
    ("A / B / C ...", "toggle that candidate on or off (press again to unlabel it)"),
    ("C", "clear this fragment completely &mdash; back to undecided"),
    ("N", "no join &mdash; then advance"),
    ("U", "unsure &mdash; then advance"),
    ("Enter", "confirm this fragment as shown, then advance"),
    ("&rarr; / &larr;", "next / previous fragment (no change)"),
    ("&uarr; / &darr;", "brighten / darken the image"),
    ("0", "reset brightness and contrast"),
    ("J", "jump to the next fragment you have not decided"),
    ("?", "show or hide this list"),
]


def build_link_page(cases: list[dict], out_path: str | Path, *, batch_id: str,
                    reviewer: str, session_started_at: str, gap_um: float,
                    cos_min: float, note: str = "") -> str:
    if not reviewer:
        raise ValueError("reviewer is required: an export without an identified "
                         "reviewer cannot be used as evidence")
    payload = {"cases": cases, "batch_id": batch_id, "reviewer": reviewer,
               "session_started_at": session_started_at,
               "fragment_rgb": list(FRAGMENT_RGB), "gap_um": gap_um, "cos_min": cos_min}
    total_candidates = sum(len(c["candidates"]) for c in cases)
    rows = "".join(f"<tr><td><kbd>{k}</kbd></td><td>{v}</td></tr>" for k, v in _SHORTCUTS)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fragment linking {batch_id}</title><style>{_CSS}</style></head><body>
<header>
  <div>
    <div class="eyebrow">fragment linking</div>
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

<main>
  <button class="navbtn" id="prev" onclick="step(-1)" title="previous (Left arrow)">&#8249;</button>
  <div id="stage">
    <img id="img" alt="">
    <img id="ovl" alt="">
    <span class="overlaybadge" id="ovlBadge"></span>
  </div>
  <button class="navbtn" id="next" onclick="step(1)" title="next (Right arrow)">&#8250;</button>
</main>

<footer>
  <div class="statusline">this fragment: <b id="status">not yet decided</b></div>
  <div class="decide" id="decide"></div>
  <div class="rowtools">
    <span><span style="display:inline-block;width:11px;height:11px;border-radius:3px;
      background:rgb{FRAGMENT_RGB};vertical-align:middle;margin-right:5px"></span>
      <b>magenta = the fragment</b>. Toggle every letter that continues the
      <i>same</i> myotube &mdash; it can join at <b>both</b> ends.</span>
    <span class="spacer"></span>
    <span>bright <input type="range" id="rBright" min="0.3" max="3.5" step="0.05" value="1"
      oninput="setBC()"></span>
    <span>contrast <input type="range" id="rContrast" min="0.3" max="3" step="0.05" value="1"
      oninput="setBC()"></span>
    <button onclick="toggleOverlay()"><kbd>L</kbd> hide/show outlines</button>
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
    <p class="sub" style="margin:12px 0 0">
      {len(cases)} fragments &middot; {total_candidates} candidate links &middot;
      gap &le; {gap_um:.0f} &micro;m, collinearity &ge; {cos_min} &middot; reviewer <b>{reviewer}</b>.
      Declining a candidate is useful data: it becomes a negative example for the linker.</p>
  </div>
</div>

<dialog id="out"><div class="dlg">
  <b>Fragment-link export</b>
  <span class="sub">Declined candidates are recorded too &mdash; they are the linker's negatives.</span>
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
