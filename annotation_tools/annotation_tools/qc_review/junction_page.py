"""Junction-splitting page: "which two branches are the same myotube passing through?"

Interaction model and visual theme are the fragment-linker page's
(`link_page.py`), which is itself `page.py`'s: a focused one-at-a-time viewer,
decide-and-advance, arrow-key navigation, an ``i / N`` counter,
brightness/contrast, a ``?`` shortcut overlay. Reusing the shell keeps this
feeling like the same instrument the operator already has muscle memory for.

The **decision model differs from the linker's on purpose**. A fragment can
join a partner at each of its two (separate, distant) ends, so the linker is
multi-select. A junction is one physical location where exactly three
branch-ends meet (this tool is scoped to degree-3 junctions -- see
`junction_pairs`), and a real fibre passes straight through **at most one**
partner there -- it cannot simultaneously continue into two different
branches from the same point. So the control here is a **single choice**
among the three possible pairs, plus "none continue through" (a genuine
branch point) and "unsure". Selecting a new pair replaces the previous
choice; selecting the current choice again clears it.

Whichever pair is NOT chosen becomes a labelled **negative** -- same principle
as the linker's declined candidates, extended to the fact that choosing one
pair here structurally declines the other two.

The **dot product that produced the classical floor's current pairing is
never shown**, for the same reason the linker never shows the model's
prediction: it would anchor the operator's independent judgement. Branch
length in microns (a plain fact, not the algorithm's opinion) is shown, the
same way the linker shows `gap_um`.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

BRANCH_RGB = [
    ("A", (34, 211, 238)),      # cyan
    ("B", (251, 191, 36)),      # amber
    ("C", (167, 139, 250)),     # violet
]
PAIR_KEYS = [("A", "B"), ("A", "C"), ("B", "C")]

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
#ovl{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;image-rendering:pixelated}
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
.decide button .swatch2{display:inline-flex;gap:2px}
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
DATA.cases.forEach(c => state[c.uid] = {choice: null, unsure: false, t: null});
let idx = 0;
let bright = 1, contrast = 1;
let overlayVisible = true;

const cur = () => DATA.cases[idx];
const nowISO = () => new Date().toISOString().replace(/\.\d+Z$/, 'Z');
const decidedCount = () => DATA.cases.filter(c => state[c.uid].t).length;

function renderHeader(){
  const n = DATA.cases.length, d = decidedCount();
  let pairs = 0, none = 0, unsure = 0;
  DATA.cases.forEach(c => { const s = state[c.uid]; if(!s.t) return;
    if(s.choice && s.choice !== 'NONE') pairs++;
    if(s.choice === 'NONE') none++;
    if(s.unsure) unsure++; });
  document.getElementById('pos').textContent = `${idx+1} / ${n}`;
  document.getElementById('progress').style.width = (100*d/n) + '%';
  document.getElementById('tally').innerHTML =
    `<span class="chip">through-pair <b>${pairs}</b></span>`+
    `<span class="chip">branch point <b>${none}</b></span>`+
    `<span class="chip">unsure <b>${unsure}</b></span>`+
    `<span class="chip" ${d<n?'style="border-color:var(--ambig)"':''}>decided <b>${d}</b>/${n}</span>`;
  document.getElementById('exportBtn').textContent =
    d < n ? `Export (${n-d} left)` : 'Export all';
}

function swatch2(rgbA, rgbB){
  return `<span class="swatch2"><span class="swatch" style="background:rgb(${rgbA})"></span>`+
         `<span class="swatch" style="background:rgb(${rgbB})"></span></span>`;
}

function renderDecide(){
  const c = cur(), s = state[c.uid];
  let h = '';
  c.pairs.forEach((p, i) => {
    const on = s.choice === p.key;
    const A = c.branches.find(b => b.letter === p.letters[0]);
    const B = c.branches.find(b => b.letter === p.letters[1]);
    h += `<button data-k="${p.key}" class="${on?'on':''}"
      style="${on?`background:linear-gradient(90deg,rgb(${A.rgb}),rgb(${B.rgb}));border-color:rgb(${A.rgb})`:''}"
      onclick="toggle('${p.key}')">
      ${swatch2(A.rgb, B.rgb)}
      <kbd>${i+1}</kbd> ${on ? '&#10003; ' : ''}${p.letters[0]}+${p.letters[1]} continue</button>`;
  });
  h += `<span class="sep"></span>`;
  h += `<button data-k="__none" class="${s.choice==='NONE'?'on':''}"
    style="${s.choice==='NONE'?'background:var(--reject);border-color:var(--reject)':''}"
    onclick="toggle('__none')"><kbd>N</kbd> ${s.choice==='NONE'?'&#10003; branch point':'branch point (none continue)'}</button>`;
  h += `<button data-k="__unsure" class="${s.unsure?'on':''}"
    style="${s.unsure?'background:var(--ambig);border-color:var(--ambig)':''}"
    onclick="toggle('__unsure')"><kbd>U</kbd> ${s.unsure?'&#10003; unsure':'unsure'}</button>`;
  h += `<span class="sep"></span>`;
  h += `<button onclick="clearCase()"><kbd>C</kbd> clear</button>`;
  h += `<button onclick="confirmNext()"><kbd>Enter</kbd> confirm &amp; next</button>`;
  document.getElementById('decide').innerHTML = h;

  let status, colour;
  if(s.unsure){ status = 'UNSURE'; colour = 'var(--ambig)'; }
  else if(s.choice === 'NONE'){ status = 'BRANCH POINT — none continue through'; colour = 'var(--reject)'; }
  else if(s.choice){
    const p = c.pairs.find(x => x.key === s.choice);
    status = `${p.letters[0]} CONTINUES INTO ${p.letters[1]}`;
    colour = 'var(--accept)';
  } else { status = s.t ? 'confirmed with nothing selected' : 'not yet decided'; colour = 'var(--muted)'; }
  const el = document.getElementById('status');
  el.textContent = status;
  el.style.color = colour;
}

function renderOverlay(){
  const ovl = document.getElementById('ovl');
  ovl.classList.toggle('hidden', !overlayVisible);
  const badge = document.getElementById('ovlBadge');
  badge.textContent = overlayVisible ? '' : 'branch outlines hidden — L to show';
  badge.style.display = overlayVisible ? 'none' : '';
}
function toggleOverlay(){ overlayVisible = !overlayVisible; renderOverlay(); }

function render(){
  const c = cur(), s = state[c.uid];
  const img = document.getElementById('img'), ovl = document.getElementById('ovl');
  if(img.dataset.uid !== c.uid){
    img.src = c.img; ovl.src = c.overlay; img.dataset.uid = c.uid;
  }
  img.style.filter = `brightness(${bright}) contrast(${contrast})`;
  renderOverlay();
  document.getElementById('caseId').textContent = `${c.well} · junction ${c.node}`;
  document.getElementById('caseInfo').innerHTML =
    c.branches.map(b => `<span class="chip"><span class="swatch" style="display:inline-block;
      width:10px;height:10px;border-radius:2px;background:rgb(${b.rgb})"></span>
      ${b.letter} · ${b.length_um} µm</span>`).join(' ');
  document.getElementById('stamp').textContent = s.t ? `decided ${s.t}` : 'not yet decided';
  renderDecide(); renderHeader();
}

function toggle(key){
  const s = state[cur().uid];
  if(key === '__none'){ s.unsure = false; s.choice = (s.choice === 'NONE') ? null : 'NONE'; }
  else if(key === '__unsure'){ s.choice = null; s.unsure = !s.unsure; }
  else { s.unsure = false; s.choice = (s.choice === key) ? null : key; }
  s.t = nowISO();
  render(); save();
}

function clearCase(){
  state[cur().uid] = {choice: null, unsure: false, t: null};
  render(); save();
}

function step(d){ idx = (idx + d + DATA.cases.length) % DATA.cases.length; render(); }
function confirmNext(){
  const s = state[cur().uid];
  if(!s.t) s.t = nowISO();
  save(); step(1);
}
function nextUndecided(){
  for(let k = 1; k <= DATA.cases.length; k++){
    const j = (idx + k) % DATA.cases.length;
    if(!state[DATA.cases[j].uid].t){ idx = j; render(); return; }
  }
  alert('Every junction has been decided.');
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

const KEY = 'junctions_' + DATA.batch_id;
function save(){ try{ localStorage.setItem(KEY, JSON.stringify(state)); }catch(_){ } }
function restore(){
  try{ const raw = localStorage.getItem(KEY); if(!raw) return;
    const saved = JSON.parse(raw);
    Object.keys(saved).forEach(u => { if(state[u]) state[u] = saved[u]; });
  }catch(_){ }
}

document.addEventListener('keydown', e => {
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
  else if(['1','2','3'].includes(e.key)){ const p = cur().pairs[+e.key - 1]; if(p) toggle(p.key); }
  if(document.activeElement && document.activeElement.tagName === 'INPUT')
    document.activeElement.blur();
});

function pairLabel(p, s){
  if(s.unsure) return null;
  if(s.choice === p.key) return 1;
  return 0;   // NONE, a different pair chosen, or nothing chosen: this pair did not continue
}

function exportAll(){
  const n = DATA.cases.length, d = decidedCount();
  if(d < n && !confirm(`${n-d} of ${n} junctions were never decided.\n\n`+
     `They export with decided_at = null and are dropped. Export anyway?`)) return;
  const payload = {schema:'junction_pairs.v1', batch_id:DATA.batch_id,
    reviewer:DATA.reviewer, session_started_at:DATA.session_started_at,
    exported_at:nowISO(), n_cases:n, n_explicitly_decided:d, decisions:{}};
  DATA.cases.forEach(c => {
    const s = state[c.uid];
    payload.decisions[c.uid] = {
      well:c.well, node:c.node, decided_at:s.t,
      chosen_pair: (s.choice && s.choice !== 'NONE') ? s.choice : null,
      branch_point: s.choice === 'NONE',
      unsure: !!s.unsure,
      pairs: c.pairs.map(p => ({key:p.key, branches:p.branches, label: s.t ? pairLabel(p, s) : null}))};
  });
  document.getElementById('outText').value = JSON.stringify(payload, null, 1);
  document.getElementById('out').showModal();
}
function copyOut(){ const t=document.getElementById('outText'); t.select();
  document.execCommand('copy'); document.getElementById('copyBtn').textContent='Copied'; }
function downloadOut(){
  const b=new Blob([document.getElementById('outText').value],{type:'application/json'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(b);
  a.download=DATA.batch_id+'.junctions.json'; a.click();
}

restore(); render();
"""

_SHORTCUTS = [
    ("1 / 2 / 3", "toggle that pair as the through-pair (press again to clear it)"),
    ("N", "<b>branch point</b> &mdash; none of the three continue through, then advance"),
    ("U", "unsure &mdash; then advance"),
    ("L", "<b>hide / show branch outlines</b> &mdash; see the bare Desmin"),
    ("C", "clear this junction completely &mdash; back to undecided"),
    ("Enter", "confirm this junction as shown, then advance"),
    ("&rarr; / &larr;", "next / previous junction (no change)"),
    ("&uarr; / &darr;", "brighten / darken the image"),
    ("0", "reset brightness and contrast"),
    ("J", "jump to the next junction you have not decided"),
    ("?", "show or hide this list"),
]


def _dilated_mask(shape: tuple[int, int], rows: np.ndarray, cols: np.ndarray,
                  iterations: int = 2) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    valid = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    mask[rows[valid], cols[valid]] = True
    return ndi.binary_dilation(mask, iterations=iterations) if mask.any() else mask


def render_case(fiber: np.ndarray, dapi, coordinates: list, branch_ids: tuple,
                centroid_rc: tuple, size: int = 460, radius_um: float = 60.0,
                margin_px: int = 20, pixel_um: float = 0.6493,
                ) -> tuple[str, str, tuple[int, int, int, int]]:
    """Render a local crop around one junction: clean fluorescence + branch-path overlay.

    Unlike the linker's `render_case` (which outlines filled instance masks),
    branches are 1px skeleton paths, so each is rasterised then dilated for
    visibility. Only the part of each branch within ``radius_um`` of the
    junction is drawn -- a branch can run the length of a whole fibre, and
    drawing all of it would either blow out the crop or shrink the junction to
    an unreadable speck.
    """
    from PIL import Image

    from .pipeline import _composite_native

    radius_px = int(round(radius_um / pixel_um))
    cr, cc = centroid_rc
    r0 = max(0, int(cr) - radius_px - margin_px)
    r1 = min(fiber.shape[0], int(cr) + radius_px + margin_px)
    c0 = max(0, int(cc) - radius_px - margin_px)
    c1 = min(fiber.shape[1], int(cc) + radius_px + margin_px)

    fiber_crop = fiber[r0:r1, c0:c1]
    dapi_crop = None if dapi is None else dapi[r0:r1, c0:c1]
    lo, hi = float(np.percentile(fiber, 1.0)), float(np.percentile(fiber, 99.5))
    dlo, dhi = ((float(np.percentile(dapi, 1.0)), float(np.percentile(dapi, 99.5)))
                if dapi is not None else (0.0, 1.0))
    clean = _composite_native(fiber_crop, dapi_crop, lo, hi, dlo, dhi)

    crop_shape = (r1 - r0, c1 - c0)
    overlay = np.zeros((crop_shape[0], crop_shape[1], 4), dtype=np.uint8)
    for branch_id, (_, rgb) in zip(branch_ids, BRANCH_RGB):
        coords = coordinates[branch_id]
        near = np.sum((coords - np.array([cr, cc])) ** 2, axis=1) <= (radius_px + margin_px) ** 2
        local = coords[near]
        rows = np.round(local[:, 0]).astype(int) - r0
        cols = np.round(local[:, 1]).astype(int) - c0
        edge = _dilated_mask(crop_shape, rows, cols)
        overlay[edge, :3] = rgb
        overlay[edge, 3] = 255

    base = Image.fromarray(clean, mode="RGB")
    layer = Image.fromarray(overlay, mode="RGBA")
    scale = size / max(base.size) if max(base.size) else 1.0
    if scale != 1:
        target = (max(1, round(base.width * scale)), max(1, round(base.height * scale)))
        base = base.resize(target, Image.LANCZOS if scale < 1 else Image.BICUBIC)
        layer = layer.resize(target, Image.NEAREST)

    clean_buffer = io.BytesIO()
    base.save(clean_buffer, format="JPEG", quality=88)
    layer_buffer = io.BytesIO()
    layer.save(layer_buffer, format="PNG", optimize=True)
    return ("data:image/jpeg;base64," + base64.b64encode(clean_buffer.getvalue()).decode("ascii"),
            "data:image/png;base64," + base64.b64encode(layer_buffer.getvalue()).decode("ascii"),
            (r0, c0, r1, c1))


def build_junction_page(cases: list[dict], out_path: str | Path, *, batch_id: str,
                        reviewer: str, session_started_at: str, note: str = "") -> str:
    if not reviewer:
        raise ValueError("reviewer is required: an export without an identified "
                         "reviewer cannot be used as evidence")
    payload = {"cases": cases, "batch_id": batch_id, "reviewer": reviewer,
               "session_started_at": session_started_at}
    rows = "".join(f"<tr><td><kbd>{k}</kbd></td><td>{v}</td></tr>" for k, v in _SHORTCUTS)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Junction splitting {batch_id}</title><style>{_CSS}</style></head><body>
<header>
  <div>
    <div class="eyebrow">junction splitting</div>
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
  <div class="statusline">this junction: <b id="status">not yet decided</b></div>
  <div class="decide" id="decide"></div>
  <div class="rowtools">
    <span>Three branches meet here. Pick the <b>one</b> pair that is the same myotube
      passing straight through, or <b>branch point</b> if none do.</span>
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
      {len(cases)} junctions &middot; reviewer <b>{reviewer}</b>.
      A pair you do not choose is useful data: it becomes a negative example for the
      junction classifier.</p>
  </div>
</div>

<dialog id="out"><div class="dlg">
  <b>Junction-pair export</b>
  <span class="sub">Declined pairs are recorded too &mdash; they are the classifier's negatives.</span>
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
