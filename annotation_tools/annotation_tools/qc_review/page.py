"""Build the serverless QC review page from assembled cases + model state."""
from __future__ import annotations

import json
from pathlib import Path

from . import model as M


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Myotube QC review — __STEM__</title>
<style>
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
:root[data-theme="light"]{
  --bg:#f4f7f5; --panel:#ffffff; --panel-2:#eef3f0; --border:#dae4de;
  --ink:#16221c; --muted:#5c6b63; --accent:#0d9488; --accent-ink:#ffffff;
  --accept:#1f9d4d; --reject:#d63a44; --ambig:#c08718;
  --accept-bg:#e6f6ec; --reject-bg:#fceaeb; --ambig-bg:#f9f0dc;
}
:root[data-theme="dark"]{
  --bg:#0c120f; --panel:#121b16; --panel-2:#0f1712; --border:#22302a;
  --ink:#dbe6df; --muted:#8ba296; --accent:#2dd4bf; --accent-ink:#04241f;
  --accept:#39c46a; --reject:#f0616b; --ambig:#e0a83a;
  --accept-bg:#12291b; --reject-bg:#2a1417; --ambig-bg:#291f0f;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.mono{font-family:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
header{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--panel) 92%,transparent);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--border);padding:14px 20px}
.hrow{display:flex;flex-wrap:wrap;gap:14px 20px;align-items:baseline}
h1{font-size:17px;margin:0;letter-spacing:-.01em;font-weight:650}
h1 .stem{color:var(--accent)}
.sub{color:var(--muted);font-size:12.5px}
.eyebrow{text-transform:uppercase;letter-spacing:.09em;font-size:10.5px;color:var(--muted);font-weight:600}
.counts{display:flex;gap:8px;flex-wrap:wrap;margin-left:auto}
.chip{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;
  border:1px solid var(--border);font-size:12px;font-weight:600;background:var(--panel-2)}
.chip .dot{width:8px;height:8px;border-radius:50%}
.chip.accept{color:var(--accept)} .chip.accept .dot{background:var(--accept)}
.chip.reject{color:var(--reject)} .chip.reject .dot{background:var(--reject)}
.chip.ambig{color:var(--ambig)} .chip.ambig .dot{background:var(--ambig)}
.chip.left{color:var(--muted)} .chip.left .dot{background:var(--muted)}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:12px}
button{font:inherit;cursor:pointer;border-radius:8px;border:1px solid var(--border);
  background:var(--panel);color:var(--ink);padding:6px 12px;font-weight:550}
button:hover{border-color:var(--accent)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.primary{background:var(--accent);color:var(--accent-ink);border-color:transparent}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.seg button{border:0;border-radius:0;padding:5px 10px;background:transparent;font-size:12.5px}
.seg button+button{border-left:1px solid var(--border)}
.filterbar{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}
.filterbar button.on{background:var(--panel-2);border-color:var(--accent);color:var(--accent)}
.intfilter{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:10px;
  padding:8px 12px;border:1px solid var(--border);border-radius:8px;background:var(--panel-2);font-size:12.5px}
.intfilter input[type=range]{width:160px;accent-color:var(--accent);vertical-align:middle}
.feat.toodark{color:var(--reject);border:1px solid var(--reject);background:transparent;font-weight:600}
.card.toodark{opacity:.5}
.kbmap{position:fixed;inset:0;z-index:60;display:none;align-items:center;justify-content:center;
  background:rgba(4,8,7,.7);backdrop-filter:blur(3px)}
.kbmap.open{display:flex}
.kbmap-inner{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:22px 26px;
  max-width:520px;box-shadow:0 20px 50px rgba(0,0,0,.4)}
.kbmap h3{margin:0 0 12px}
.kbmap table{border-collapse:collapse;width:100%;font-size:13.5px}
.kbmap td{padding:4px 8px;border-bottom:1px solid var(--border)}
.kbmap td:first-child{width:96px}
.model{margin-top:12px;padding:10px 14px;border:1px solid var(--border);border-radius:10px;
  background:var(--panel-2);font-size:12.5px;color:var(--muted);display:flex;gap:10px;align-items:flex-start}
.model b{color:var(--ink)}
.model .badge{flex:none;font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;
  padding:3px 8px;border-radius:6px;font-weight:700}
.model .badge.learn{background:var(--accept-bg);color:var(--accept)}
.model .badge.cold{background:var(--ambig-bg);color:var(--ambig)}
.savewarn{margin-top:10px;padding:8px 12px;border:1px solid var(--ambig);border-radius:8px;
  background:var(--ambig-bg);color:var(--ambig);font-size:12.5px}
main{padding:18px 20px 120px;max-width:1400px;margin:0 auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;
  box-shadow:var(--shadow);overflow:hidden;transition:border-color .12s}
.card[data-act="accept"]{border-color:color-mix(in srgb,var(--accept) 55%,var(--border))}
.card[data-act="reject"]{border-color:color-mix(in srgb,var(--reject) 55%,var(--border));opacity:.72}
.card[data-act="ambiguous"]{border-color:color-mix(in srgb,var(--ambig) 55%,var(--border))}
.card-top{display:flex;gap:14px;padding:14px}
.thumbwrap{position:relative;flex:none}
.thumb{width:150px;height:150px;display:block;border-radius:8px;border:1px solid var(--border);
  object-fit:contain;background:#080c0b;cursor:zoom-in}
.thumbwrap .zoom{position:absolute;right:6px;bottom:6px;background:color-mix(in srgb,var(--panel) 80%,transparent);
  border:1px solid var(--border);border-radius:6px;font-size:11px;padding:1px 6px;color:var(--muted);
  pointer-events:none;opacity:0;transition:opacity .12s}
.thumbwrap:hover .zoom{opacity:1}
.meta{min-width:0;flex:1}
.idrow{display:flex;align-items:baseline;gap:8px;justify-content:space-between}
.pid{font-weight:650;font-size:13px}
.suggest{font-size:11px;font-weight:600;padding:2px 7px;border-radius:6px;white-space:nowrap}
.suggest.accept{background:var(--accept-bg);color:var(--accept)}
.suggest.reject{background:var(--reject-bg);color:var(--reject)}
.feats{display:flex;flex-wrap:wrap;gap:4px 6px;margin-top:8px}
.feat{font-size:11px;color:var(--muted);background:var(--panel-2);border-radius:5px;padding:2px 6px}
.feat b{color:var(--ink);font-weight:600}
.feat.warn{color:var(--reject)}
.feat.splitbadge{color:var(--accent);border:1px solid var(--accent);background:transparent;font-weight:600}
.decide{display:flex;border-top:1px solid var(--border)}
.decide button{flex:1;border:0;border-radius:0;padding:8px 4px;background:transparent;
  font-size:12.5px;font-weight:600;color:var(--muted)}
.decide button+button{border-left:1px solid var(--border)}
.decide button.sel-accept{background:var(--accept-bg);color:var(--accept)}
.decide button.sel-reject{background:var(--reject-bg);color:var(--reject)}
.decide button.sel-ambig{background:var(--ambig-bg);color:var(--ambig)}
footer{position:fixed;left:0;right:0;bottom:0;z-index:20;background:color-mix(in srgb,var(--panel) 94%,transparent);
  backdrop-filter:blur(8px);border-top:1px solid var(--border);padding:12px 20px;
  display:flex;gap:14px;align-items:center;flex-wrap:wrap}
footer .grow{flex:1;min-width:120px;color:var(--muted);font-size:12.5px}
.saved{font-size:11.5px;color:var(--muted)}
.iopanel{position:fixed;left:0;right:0;bottom:56px;z-index:25;display:flex;gap:16px;
  padding:14px 20px;background:var(--panel);border-top:1px solid var(--border);
  box-shadow:0 -8px 24px rgba(0,0,0,.18)}
.iopanel[hidden]{display:none}
.iocol{flex:1;display:flex;flex-direction:column;gap:8px;min-width:0}
.iocol label{font-size:12px;color:var(--muted);font-weight:600}
.iocol textarea{width:100%;height:150px;resize:vertical;border:1px solid var(--border);
  border-radius:8px;background:var(--panel-2);color:var(--ink);padding:8px;
  font-family:ui-monospace,Consolas,monospace;font-size:11.5px}
details.help{margin-top:12px}
details.help summary{cursor:pointer;font-size:12.5px;color:var(--accent);font-weight:600}
details.help p{color:var(--muted);font-size:12.5px;max-width:70ch}
kbd{font:inherit;font-size:11px;background:var(--panel-2);border:1px solid var(--border);
  border-radius:4px;padding:1px 5px}
/* lightbox */
.lb{position:fixed;inset:0;z-index:50;display:none;background:rgba(4,8,7,.82);
  backdrop-filter:blur(4px);padding:24px}
.lb.open{display:flex;align-items:center;justify-content:center}
.lb-inner{display:flex;gap:22px;max-width:1140px;width:100%;max-height:92vh;
  background:var(--panel);border:1px solid var(--border);border-radius:16px;
  box-shadow:0 24px 60px rgba(0,0,0,.5);padding:20px;align-items:flex-start}
.lb-editor{flex:none;width:min(58vw,560px);display:flex;flex-direction:column;gap:10px}
.lb-canvas-wrap{position:relative;width:100%;height:min(58vw,560px);max-height:74vh;
  background:#080c0b;border-radius:10px;border:1px solid var(--border);overflow:hidden;
  display:flex;align-items:center;justify-content:center;touch-action:none}
.lb-canvas-wrap canvas{position:absolute;image-rendering:auto}
.lb-canvas-wrap #lbBg{z-index:1}
.lb-canvas-wrap canvas.mask{cursor:crosshair;z-index:2}
.lb-tools{display:flex;gap:8px 10px;align-items:center;flex-wrap:wrap;width:100%;
  font-size:12.5px;color:var(--muted)}
.lb-tools .seg button.on,.lb-tools button.on{background:var(--accent);color:var(--accent-ink);border-color:transparent}
.lb-tools input[type=range]{width:120px;accent-color:var(--accent)}
.edited-flag{color:var(--accent);font-weight:600}
.lb-labels{display:flex;gap:6px;flex-wrap:wrap}
.lbchip{font-size:12px;padding:4px 9px;border:1px solid var(--border);border-radius:999px;
  background:var(--panel-2);color:var(--ink)}
.lbchip .dot{color:var(--c)}
.lbchip.on{border-color:var(--c,var(--accent));box-shadow:inset 0 0 0 1px var(--c,var(--accent))}
.lbchip.add{border-style:dashed;color:var(--muted)}
.lbchip.splitcards{background:var(--accent);color:var(--accent-ink);border-color:transparent;font-weight:600}
.lb-side{flex:1 1 260px;min-width:0;display:flex;flex-direction:column;gap:12px}
.lb-side h2{margin:0;font-size:20px;letter-spacing:-.01em}
.lb-feats{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.lb-feat{background:var(--panel-2);border:1px solid var(--border);border-radius:8px;padding:8px 10px}
.lb-feat .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.lb-feat .v{font-size:17px;font-weight:600}
.lb-decide{display:flex;gap:8px;margin-top:auto}
.lb-decide button{flex:1;padding:11px;font-size:14px;font-weight:650}
.lb-decide .b-accept.on{background:var(--accept-bg);color:var(--accept);border-color:var(--accept)}
.lb-decide .b-ambig.on{background:var(--ambig-bg);color:var(--ambig);border-color:var(--ambig)}
.lb-decide .b-reject.on{background:var(--reject-bg);color:var(--reject);border-color:var(--reject)}
.lb-nav{display:flex;gap:8px;align-items:center;color:var(--muted);font-size:12.5px}
.lb-close{position:absolute;top:16px;right:20px;font-size:20px;line-height:1;
  background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:6px 12px}
@media (max-width:720px){.lb-inner{flex-direction:column;overflow:auto}.lb-img{width:100%;height:auto}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head>
<body>
<header>
  <div class="hrow">
    <div>
      <div class="eyebrow">PrecisionMyotube · proposal QC · <span style="color:var(--accent)">auto-split build</span></div>
      <h1>Confirm real myotubes — <span class="stem" id="stem"></span></h1>
      <div class="sub" id="sub"></div>
    </div>
    <div class="counts" id="counts"></div>
  </div>
  <div class="model" id="model"></div>
  <div class="toolbar">
    <button class="primary" onclick="acceptSuggested()">✓ Confirm all suggested</button>
    <button onclick="resetSuggested()">Reset to suggested</button>
    <div class="filterbar" id="filters">
      <button data-f="all" class="on">All</button>
      <button data-f="todo">Needs a look</button>
      <button data-f="accept">Accepted</button>
      <button data-f="reject">Rejected</button>
      <button data-f="ambiguous">Ambiguous</button>
    </div>
  </div>
  <div class="intfilter">
    <label>🔅 min fibre intensity
      <input type="range" id="intFilter" min="0" value="0" step="10" oninput="setIntensity()">
      <span class="mono" id="intVal">0</span></label>
    <span class="muted" id="intCount"></span>
    <button onclick="rejectTooDark()">Reject all too-dark</button>
    <span class="muted" style="margin-left:auto;cursor:pointer"
      onclick="document.getElementById('kbmap').classList.add('open')"><kbd>?</kbd> shortcuts</span>
  </div>
  <div class="savewarn">⚠ This shared view can't save files. To keep your work: click <b>⇅ Save / Restore</b> (bottom),
    <b>Copy all</b>, and paste the text somewhere safe. Restore by pasting it back. For real saving, open the
    local <b>review.html</b> file instead — it saves automatically.</div>
  <details class="help">
    <summary>How this works — you confirm, it learns</summary>
    <p>Each card is one automated <b>proposal</b>. The question is: <b>is this one real,
    complete myotube?</b></p>
    <ul style="color:var(--muted);font-size:12.5px;max-width:70ch;line-height:1.6">
      <li><b>Accept</b> — yes, one real myotube (edit the mask first if the shape is off).</li>
      <li><b>Reject</b> — not a real myotube at all (debris, a stray fragment, background).</li>
      <li><b>Ambiguous</b> — you genuinely can't resolve the identity or boundary. Kept, but
        excluded from length/width stats. <i>Not</i> for "there are two" — that's a split.</li>
      <li><b>Merged / branched — actually several myotubes?</b> Enlarge it and press <b>⟳ Hypothesis</b>
        (key <kbd>H</kbd>) to auto-split by tracing fibres through crossings. If the colours are wrong,
        use <b>◫ Assign</b> (key <kbd>G</kbd>) and click a segment to move it to the selected myotube;
        <b>+ separate myotube</b> adds another. Then <b>Accept</b> — each colour becomes its own instance.</li>
      <li><b>Label too short?</b> Enlarge it and use <b>✏ Add</b> to extend the mask along the
        fibre (there's extra image around each proposal for exactly this).</li>
    </ul>
    <p>Once you've made ≥12 accept/reject decisions, a small logistic-regression model learns
    your pattern from the shown features and pre-decides the next batch — so you
    <b>confirm rather than trace</b>. Every suggestion shows its probability.</p>
  </details>
</header>
<main><div class="grid" id="grid"></div></main>
<div class="iopanel" id="ioPanel" hidden>
  <div class="iocol">
    <label>① SAVE — copy this text and keep it (a note, an email to yourself, anywhere)</label>
    <textarea id="ioOut" readonly spellcheck="false"></textarea>
    <button class="primary" onclick="copyJson()">⧉ Copy all</button>
  </div>
  <div class="iocol">
    <label>② RESTORE — paste a saved text here, then Restore</label>
    <textarea id="ioIn" spellcheck="false" placeholder="paste your saved decisions text here…"></textarea>
    <button onclick="restoreFromBox()">↩ Restore from text</button>
    <input type="file" id="importFile" accept="application/json,.json" style="display:none"
           onchange="importFromFile(event)">
    <button onclick="document.getElementById('importFile').click()">or load a .json file</button>
  </div>
</div>
<footer>
  <span class="grow" id="footsummary"></span>
  <span class="saved" id="saved"></span>
  <button class="primary" onclick="toggleIO()">⇅ Save / Restore</button>
  <button onclick="downloadJson()">⬇ Download (may be blocked here)</button>
</footer>
<div class="kbmap" id="kbmap" onclick="if(event.target.id==='kbmap')this.classList.remove('open')">
  <div class="kbmap-inner">
    <h3>Keyboard shortcuts</h3>
    <table>
      <tr><td><kbd>A</kbd>/<kbd>Enter</kbd></td><td>Accept &amp; next fibre</td></tr>
      <tr><td><kbd>R</kbd></td><td>Reject &amp; next</td></tr>
      <tr><td><kbd>X</kbd></td><td>Ambiguous &amp; next</td></tr>
      <tr><td><kbd>←</kbd>/<kbd>→</kbd></td><td>Previous / next fibre</td></tr>
      <tr><td><kbd>↑</kbd>/<kbd>↓</kbd></td><td>Brightness up / down</td></tr>
      <tr><td><kbd>B</kbd>/<kbd>E</kbd></td><td>Brush add / erase</td></tr>
      <tr><td><kbd>H</kbd></td><td>Cycle split hypothesis</td></tr>
      <tr><td><kbd>G</kbd></td><td>Assign-segment tool</td></tr>
      <tr><td><kbd>1</kbd>–<kbd>5</kbd></td><td>Select myotube colour</td></tr>
      <tr><td><kbd>S</kbd></td><td>Split into separate cards</td></tr>
      <tr><td><kbd>L</kbd></td><td>Toggle label overlay</td></tr>
      <tr><td><kbd>Backspace</kbd></td><td>Clear active label</td></tr>
      <tr><td><kbd>U</kbd></td><td>Reset to original proposal</td></tr>
      <tr><td><kbd>Esc</kbd></td><td>Close</td></tr>
      <tr><td><kbd>?</kbd></td><td>This help</td></tr>
    </table>
  </div>
</div>
<div class="lb" id="lb" aria-hidden="true">
  <button class="lb-close" onclick="closeLb()" aria-label="Close">✕ Esc</button>
  <div class="lb-inner">
    <div class="lb-editor">
      <div class="lb-canvas-wrap" id="lbWrap">
        <canvas id="lbBg"></canvas>
        <canvas id="lbMask" class="mask"></canvas>
      </div>
      <div class="lb-tools">
        <span class="seg">
          <button id="tBrush" class="on" onclick="setTool('brush')">✏ Add</button>
          <button id="tErase" onclick="setTool('erase')">⌫ Remove</button>
          <button id="tAssign" onclick="setTool('assign')">◫ Assign</button>
        </span>
        <button id="lbHyp" onclick="cycleHypothesis()">⟳ Hypothesis</button>
        <label>brush <input type="range" id="tSize" min="1" max="24" value="3"
          oninput="document.getElementById('tSizeVal').textContent=this.value"><span id="tSizeVal">3</span> px</label>
        <label title="brightness">☀<input type="range" id="tBright" min="0.3" max="3.5" step="0.05" value="1"
          oninput="applyBC()" style="width:66px"></label>
        <label title="contrast">◐<input type="range" id="tContrast" min="0.5" max="3.5" step="0.05" value="1"
          oninput="applyBC()" style="width:66px"></label>
        <button onclick="resetBC()">reset B/C</button>
        <button id="tOverlay" class="on" onclick="toggleOverlay()">👁 Label</button>
        <button onclick="clearMask()">✕ Clear</button>
        <button onclick="resetMask()">↺ Reset</button>
        <label title="why the machine proposal was wrong (auto-detected; override if needed)">why:
          <select id="lbReason" onchange="setReason()">
            <option value="">auto</option><option value="too_short">too short</option>
            <option value="spillover">spillover</option><option value="split">split</option>
            <option value="reshape">reshape</option><option value="other">other</option>
          </select></label>
        <span id="lbEdited"></span>
      </div>
      <div class="lb-labels" id="lbLabels"></div>
    </div>
    <div class="lb-side">
      <div class="lb-nav">
        <button onclick="stepLb(-1)">← Prev</button>
        <button onclick="stepLb(1)">Next →</button>
        <span id="lbPos"></span>
      </div>
      <h2 class="mono" id="lbId"></h2>
      <div class="mono" id="lbSuggest" style="color:var(--muted);font-size:12.5px"></div>
      <div class="lb-feats mono" id="lbFeats"></div>
      <div class="lb-decide">
        <button class="b-accept" onclick="setLb('accept')">Accept</button>
        <button class="b-ambig" onclick="setLb('ambiguous')">Ambiguous</button>
        <button class="b-reject" onclick="setLb('reject')">Reject</button>
      </div>
    </div>
  </div>
</div>
<script>
const DATA = __DATA__;
const PIXEL_UM = DATA.pixel_um || 0.6493;
const LSKEY = "myoqc_"+DATA.stem;
const state = {};
// intensity filter: whole fibres dimmer than the threshold are "too dark to count"
let intensityMin = 0;
const MAX_FIBER = Math.max(1, ...DATA.cases.map(c => (c.features && c.features.fiber_mean) || 0));
function fiberMean(c){ return (c.features && c.features.fiber_mean) || 0; }
function tooDark(c){ return intensityMin > 0 && fiberMean(c) < intensityMin; }
function setIntensity(){
  intensityMin = +document.getElementById("intFilter").value;
  document.getElementById("intVal").textContent = intensityMin;
  const n = DATA.cases.filter(tooDark).length;
  document.getElementById("intCount").textContent =
    intensityMin > 0 ? `${n} of ${DATA.cases.length} below threshold (too dark)` : "off — show all";
  try { localStorage.setItem(LSKEY+"_int", intensityMin); } catch(e){}
  build();
}
function rejectTooDark(){
  if(intensityMin<=0){flash("set the intensity threshold first");return;}
  let n=0; DATA.cases.forEach(c=>{ if(tooDark(c)){ state[c.id].action="reject"; stampDecision(c.id); n++; } });
  build(); save(); flash("rejected "+n+" too-dark fibres");
}

function suggested(c){ if(DATA.blind) return "ambiguous";   // blind: conservative default, no pre-accept
  if(tooDark(c)) return "reject";
  return c.learned_action || (c.prior>=0.5?"accept":"reject"); }
DATA.cases.forEach(c=>state[c.id]={action:suggested(c),note:"",labels:null,active:0,edited:false,hyp:0,reason:""});
function setReason(){const c=curCase();if(c){state[c.id].reason=document.getElementById("lbReason").value;save();}}

// One proposal can resolve to several myotubes; each is a separate mask ("label").
const LABEL_COLORS=[[255,74,170],[64,209,255],[255,196,64],[120,255,140],[180,140,255]];

/* ---- run-length codec for editable masks (row-major, bg-first) ---- */
function decodeRle(rle){const h=rle.h,w=rle.w,arr=new Uint8Array(h*w);
  let pos=0,val=0;for(const c of rle.counts){if(val)arr.fill(1,pos,pos+c);pos+=c;val^=1;}return arr;}
function encodeRle(arr,h,w){const counts=[];let prev=0,run=0;
  for(let i=0;i<arr.length;i++){const v=arr[i]?1:0;
    if(i===0){if(v===1)counts.push(0);run=1;prev=v;}
    else if(v===prev)run++;else{counts.push(run);run=1;prev=v;}}
  counts.push(run);return{h,w,counts};}
function nonEmpty(arr){for(let i=0;i<arr.length;i++)if(arr[i])return true;return false;}

/* ---- auto-split hypotheses (crossing-tracer) ---- */
function hasSplit(c){return c.segments&&c.segments.length>=2&&c.hypotheses&&c.hypotheses.length>1;}
function segMap(c){                 // Int16Array: 0=bg, s+1 = segment index; + per-segment pixel lists
  if(c._segmap)return c._segmap;
  const map=new Int16Array(c.geom.edit_w*c.geom.edit_h),pix=[];
  (c.segments||[]).forEach((rle,si)=>{const m=decodeRle(rle),arr=[];
    for(let i=0;i<m.length;i++)if(m[i]){map[i]=si+1;arr.push(i);} pix.push(arr);});
  c._segmap=map;c._segpix=pix;return map;
}
function loadHypothesis(c){          // build one group-mask per fibre from segments + grouping
  const st=state[c.id];
  if(st.hyp===0){st.labels=[decodeRle(c.mask_rle)];st.active=0;return;} // exact whole
  const grouping=c.hypotheses[st.hyp],nG=Math.max.apply(null,grouping)+1;
  const labels=Array.from({length:nG},()=>new Uint8Array(c.geom.edit_w*c.geom.edit_h));
  segMap(c);c._segpix.forEach((arr,si)=>{const g=grouping[si];arr.forEach(i=>labels[g][i]=1);});
  st.labels=labels;if(st.active>=nG)st.active=0;
}
function ensureLabels(c){const st=state[c.id];
  if(!st.labels){ if(hasSplit(c))loadHypothesis(c); else st.labels=[decodeRle(c.mask_rle)]; }
  if(st.active>=st.labels.length)st.active=0;return st.labels;}
const HYP_NAMES=["whole (one myotube)","traced fibres","every crossing"];
function hypLabel(c){const st=state[c.id];
  return `⟳ ${HYP_NAMES[st.hyp]||("hyp "+st.hyp)} (${st.hyp+1}/${c.hypotheses.length})`;}
function cycleHypothesis(){const c=curCase();if(!c||!hasSplit(c))return;
  const st=state[c.id];st.hyp=(st.hyp+1)%c.hypotheses.length;loadHypothesis(c);
  st.edited=(st.hyp!==0);renderLabelChips(c);renderOverlay(c);markEdited(c);save();
  document.getElementById("lbHyp").textContent=hypLabel(c);}
/* ---- split one proposal into two independent review cards ---- */
function geomFeats(mask,ew,eh){
  let n=0,sx=0,sy=0,minx=1e9,miny=1e9,maxx=-1,maxy=-1;
  for(let i=0;i<mask.length;i++)if(mask[i]){const y=(i/ew)|0,x=i%ew;
    sx+=x;sy+=y;n++;if(x<minx)minx=x;if(x>maxx)maxx=x;if(y<miny)miny=y;if(y>maxy)maxy=y;}
  if(n<2)return null;
  const mx=sx/n,my=sy/n;let cxx=0,cyy=0,cxy=0;
  for(let i=0;i<mask.length;i++)if(mask[i]){const y=(i/ew)|0,x=i%ew,dx=x-mx,dy=y-my;cxx+=dx*dx;cyy+=dy*dy;cxy+=dx*dy;}
  cxx/=n;cyy/=n;cxy/=n;
  const tr=cxx+cyy,disc=Math.sqrt(Math.max(0,tr*tr/4-(cxx*cyy-cxy*cxy)));
  const major=4*Math.sqrt(Math.max(0,tr/2+disc)),minor=4*Math.sqrt(Math.max(0,tr/2-disc));
  return {n,major,minor,minx,miny,maxx,maxy};
}
function childFeatures(mask,ew,eh,geom,parent){
  const g=geomFeats(mask,ew,eh);if(!g)return {...parent.features};
  const scale=geom.src_h/geom.edit_h;
  return {length_um:+(g.major*scale*PIXEL_UM).toFixed(2),
    width_um:+(g.minor*scale*PIXEL_UM).toFixed(2),
    area_um2:+(g.n*scale*scale*PIXEL_UM*PIXEL_UM).toFixed(1),
    aspect:+(g.major/Math.max(g.minor,1e-6)).toFixed(2),
    solidity:parent.features.solidity, fiber_mean:parent.features.fiber_mean,
    territory_overlap:parent.features.territory_overlap,
    touches_border:parent.features.touches_border};
}
function priorJS(f){
  const lt=1-Math.exp(-f.length_um/50), el=Math.min(f.aspect/4,1),
        tr=f.territory_overlap, bo=f.touches_border?0.6:1;
  return lt*(0.4+0.6*el)*(0.3+0.7*tr)*bo;
}
function renderThumbSync(mask,ew,eh){
  const cv=document.createElement("canvas");cv.width=ew;cv.height=eh;
  const ctx=cv.getContext("2d");
  if(bgImgEl&&bgImgEl.complete)ctx.drawImage(bgImgEl,0,0,ew,eh);else{ctx.fillStyle="#080c0b";ctx.fillRect(0,0,ew,eh);}
  const id=ctx.getImageData(0,0,ew,eh),d=id.data;
  for(let y=0;y<eh;y++)for(let x=0;x<ew;x++){const i=y*ew+x;if(!mask[i])continue;
    const edge=x===0||x===ew-1||y===0||y===eh-1||!mask[i-1]||!mask[i+1]||!mask[i-ew]||!mask[i+ew];
    if(edge){const j=i*4;d[j]=255;d[j+1]=74;d[j+2]=170;d[j+3]=255;}}
  ctx.putImageData(id,0,0);return cv.toDataURL("image/png");
}
function splitIntoCards(){
  const c=curCase();if(!c)return;
  const parts=ensureLabels(c).filter(nonEmpty);
  if(parts.length<2){flash("colour at least two myotubes first (⟳ Hypothesis, or + separate myotube + Assign)");return;}
  const ew=c.geom.edit_w,eh=c.geom.edit_h,letters="abcdefghij",newCases=[];
  parts.forEach((m,idx)=>{
    const feats=childFeatures(m,ew,eh,c.geom,c),id=c.id+"_"+letters[idx];
    newCases.push({id,features:feats,prior:+priorJS(feats).toFixed(3),
      thumb:renderThumbSync(m,ew,eh),edit_img:c.edit_img,
      mask_rle:encodeRle(m,eh,ew),geom:c.geom});
    state[id]={action:DATA.blind?"ambiguous":(priorJS(feats)>=0.5?"accept":"reject"),note:"",
      labels:[Uint8Array.from(m)],active:0,edited:true,hyp:0};
  });
  const i=DATA.cases.findIndex(x=>x.id===c.id);
  DATA.cases.splice(i,1,...newCases);
  delete state[c.id];
  lbId=newCases[0].id;
  build();paintLb();renderCounts();save();
  flash("split into "+newCases.length+" separate cards");
}

function assignSegmentAt(c,clientX,clientY){
  const mk=document.getElementById("lbMask"),r=mk.getBoundingClientRect(),ew=c.geom.edit_w,eh=c.geom.edit_h;
  const x=Math.floor((clientX-r.left)/r.width*ew),y=Math.floor((clientY-r.top)/r.height*eh);
  if(x<0||y<0||x>=ew||y>=eh)return;
  const seg=segMap(c)[y*ew+x];if(seg<=0)return;
  const labels=ensureLabels(c),active=state[c.id].active;
  c._segpix[seg-1].forEach(i=>{for(let g=0;g<labels.length;g++)labels[g][i]=(g===active)?1:0;});
  state[c.id].edited=true;renderOverlay(c);markEdited(c);save();}

function esc(s){return (s||"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]));}
function fmt(x){return (typeof x==="number")?(Number.isInteger(x)?x:x.toFixed(x<10?2:1)):x;}

let flt="all";
function visible(c){const a=state[c.id].action;
  if(flt==="all")return true;
  if(flt==="todo")return a===suggested(c) && a!=="ambiguous";  // untouched suggestions
  return a===flt;}

function counts(){
  let a=0,r=0,m=0;DATA.cases.forEach(c=>{const x=state[c.id].action;
    if(x==="accept")a++;else if(x==="reject")r++;else m++;});
  return {a,r,m,total:DATA.cases.length};
}
function renderCounts(){
  const k=counts();
  document.getElementById("counts").innerHTML=
    `<span class="chip accept"><span class="dot"></span>${k.a} accept</span>`+
    `<span class="chip reject"><span class="dot"></span>${k.r} reject</span>`+
    `<span class="chip ambig"><span class="dot"></span>${k.m} ambiguous</span>`;
  document.getElementById("footsummary").textContent=
    `${k.a} accepted · ${k.r} rejected · ${k.m} ambiguous of ${k.total} proposals`;
}

function card(c){
  const st=state[c.id], f=c.features, sug=suggested(c);
  const conf = DATA.blind ? "decide fresh"
    : (c.learned_action ? `model: ${c.learned_action} · p=${c.learned_proba}`
                        : `suggested: ${sug} · prior ${c.prior}`);
  const border = f.touches_border ? `<span class="feat warn">edge</span>` : "";
  const split = hasSplit(c) ? `<span class="feat splitbadge">⟳ splittable</span>` : "";
  const dark = tooDark(c) ? `<span class="feat toodark">too dark</span>` : "";
  return `<div class="card${tooDark(c)?' toodark':''}" data-act="${st.action}" id="card_${c.id}">
    <div class="card-top">
      <span class="thumbwrap">
        <img class="thumb" src="${c.thumb}" alt="proposal ${c.id}" loading="lazy"
             onclick="openLb('${c.id}')">
        <span class="zoom">⤢ enlarge</span>
      </span>
      <div class="meta">
        <div class="idrow"><span class="pid mono">${c.id.replace("myotube_","#")}</span>
          <span class="suggest ${sug}">${sug}</span></div>
        <div class="sub mono" style="font-size:11px">${conf}</div>
        <div class="feats mono">
          <span class="feat">len <b>${fmt(f.length_um)}</b>µm</span>
          <span class="feat">wid <b>${fmt(f.width_um)}</b>µm</span>
          <span class="feat">aspect <b>${fmt(f.aspect)}</b></span>
          <span class="feat">terr <b>${Math.round(f.territory_overlap*100)}</b>%</span>
          <span class="feat">int <b>${fmt(f.fiber_mean)}</b></span>
          ${border}${split}${dark}
        </div>
      </div>
    </div>
    <div class="decide" id="dec_${c.id}">
      <button data-a="accept">Accept</button>
      <button data-a="ambiguous">Ambiguous</button>
      <button data-a="reject">Reject</button>
    </div></div>`;
}

function paintDecide(id){
  const st=state[id];
  document.querySelectorAll(`#dec_${id} button`).forEach(b=>{
    const a=b.dataset.a; b.className="";
    if(a===st.action) b.classList.add(a==="accept"?"sel-accept":a==="reject"?"sel-reject":"sel-ambig");
  });
  const card=document.getElementById("card_"+id); if(card) card.dataset.act=st.action;
}

function build(){
  const grid=document.getElementById("grid");
  grid.innerHTML=DATA.cases.filter(visible).map(card).join("");
  DATA.cases.forEach(c=>{
    const dec=document.getElementById("dec_"+c.id); if(!dec) return;
    dec.querySelectorAll("button").forEach(b=>b.onclick=()=>{
      state[c.id].action=b.dataset.a; stampDecision(c.id); paintDecide(c.id); renderCounts(); save();
      if(flt!=="all"){ const card=document.getElementById("card_"+c.id);
        if(card && !visible(c)) card.style.display="none"; }
    });
    paintDecide(c.id);
  });
  renderCounts();
}

function acceptSuggested(){DATA.cases.forEach(c=>{state[c.id].action=suggested(c);stampDecision(c.id);});build();save();}
function resetSuggested(){acceptSuggested();}

/* ---- lightbox editor: click a thumbnail to enlarge, then ADD / REMOVE mask ---- */
let lbId=null, tool="brush", painting=false, bgImgEl=null, overlayVisible=true;
function visibleList(){return DATA.cases.filter(visible);}
function curCase(){return DATA.cases.find(x=>x.id===lbId);}
function openLb(id){lbId=id;
  document.getElementById("lb").classList.add("open");
  document.getElementById("lb").setAttribute("aria-hidden","false");
  paintLb();}
function closeLb(){document.getElementById("lb").classList.remove("open");
  document.getElementById("lb").setAttribute("aria-hidden","true");lbId=null;}
function setTool(t){tool=t;
  document.getElementById("tBrush").classList.toggle("on",t==="brush");
  document.getElementById("tErase").classList.toggle("on",t==="erase");
  const a=document.getElementById("tAssign");if(a)a.classList.toggle("on",t==="assign");}

function layoutCanvases(c){
  const wrap=document.getElementById("lbWrap"),bg=document.getElementById("lbBg"),mk=document.getElementById("lbMask");
  const ew=c.geom.edit_w,eh=c.geom.edit_h;
  bg.width=ew;bg.height=eh;mk.width=ew;mk.height=eh;
  const bw=wrap.clientWidth,bh=wrap.clientHeight,s=Math.min(bw/ew,bh/eh);
  const dw=Math.round(ew*s),dh=Math.round(eh*s),lx=Math.round((bw-dw)/2),ly=Math.round((bh-dh)/2);
  [bg,mk].forEach(cv=>{cv.style.width=dw+"px";cv.style.height=dh+"px";cv.style.left=lx+"px";cv.style.top=ly+"px";});
}
function renderOverlay(c){
  const mk=document.getElementById("lbMask"),ew=c.geom.edit_w,eh=c.geom.edit_h;
  const ctx=mk.getContext("2d");
  if(!overlayVisible){ctx.clearRect(0,0,ew,eh);return;}
  const labels=ensureLabels(c),img=ctx.createImageData(ew,eh),d=img.data;
  for(let li=0;li<labels.length;li++){const col=LABEL_COLORS[li%LABEL_COLORS.length];
    const m=labels[li],a=(li===state[c.id].active)?165:110;
    for(let i=0;i<m.length;i++){if(m[i]){const j=i*4;d[j]=col[0];d[j+1]=col[1];d[j+2]=col[2];d[j+3]=a;}}}
  ctx.putImageData(img,0,0);
}
function paintAt(c,clientX,clientY){
  const mk=document.getElementById("lbMask"),r=mk.getBoundingClientRect();
  const ew=c.geom.edit_w,eh=c.geom.edit_h,st=state[c.id],labels=ensureLabels(c);
  const mask=labels[st.active];
  const x=Math.round((clientX-r.left)/r.width*ew),y=Math.round((clientY-r.top)/r.height*eh);
  const rad=Math.max(1,+document.getElementById("tSize").value);   // radius in mask pixels
  const add=tool==="brush";
  for(let dy=-rad;dy<=rad;dy++)for(let dx=-rad;dx<=rad;dx++){
    if(dx*dx+dy*dy>rad*rad)continue;const px=x+dx,py=y+dy;
    if(px<0||py<0||px>=ew||py>=eh)continue;const k=py*ew+px;
    if(add){for(let o=0;o<labels.length;o++)if(o!==st.active)labels[o][k]=0; mask[k]=1;} // labels don't overlap
    else mask[k]=0;}
  st.edited=true;renderOverlay(c);markEdited(c);
}
function markEdited(c){const el=document.getElementById("lbEdited");if(!el)return;
  const n=ensureLabels(c).filter(nonEmpty).length;
  el.innerHTML=(state[c.id].edited?'<span class="edited-flag">● edited</span> ':'')+
    (n>1?`<span class="edited-flag">${n} myotubes</span>`:'');}
function renderLabelChips(c){
  const wrap=document.getElementById("lbLabels");if(!wrap)return;
  const labels=ensureLabels(c),st=state[c.id];
  let h="";labels.forEach((m,li)=>{const col=LABEL_COLORS[li%LABEL_COLORS.length];
    h+=`<button class="lbchip ${li===st.active?'on':''}" onclick="setActiveLabel(${li})"
      style="--c:rgb(${col[0]},${col[1]},${col[2]})"><span class="dot">●</span> myotube ${li+1}${nonEmpty(m)?'':' (empty)'}</button>`;});
  h+=`<button class="lbchip add" onclick="addLabel()">+ separate myotube</button>`;
  if(labels.length>1)h+=`<button class="lbchip" onclick="deleteLabel()">🗑 remove myotube ${st.active+1}</button>`;
  if(labels.filter(nonEmpty).length>=2)
    h+=`<button class="lbchip splitcards" onclick="splitIntoCards()">✂ Split into ${labels.filter(nonEmpty).length} cards</button>`;
  wrap.innerHTML=h;
}
function setActiveLabel(i){const c=curCase();if(!c)return;state[c.id].active=i;
  renderLabelChips(c);renderOverlay(c);}
function addLabel(){const c=curCase();if(!c)return;const labels=ensureLabels(c);
  labels.push(new Uint8Array(c.geom.edit_w*c.geom.edit_h));state[c.id].active=labels.length-1;
  state[c.id].edited=true;renderLabelChips(c);renderOverlay(c);markEdited(c);save();}
function deleteLabel(){const c=curCase();if(!c)return;const st=state[c.id],labels=ensureLabels(c);
  if(labels.length<=1)return;labels.splice(st.active,1);st.active=0;st.edited=true;
  renderLabelChips(c);renderOverlay(c);markEdited(c);save();}
function resetMask(){const c=curCase();if(!c)return;
  state[c.id].labels=[decodeRle(c.mask_rle)];state[c.id].active=0;state[c.id].edited=false;
  renderLabelChips(c);renderOverlay(c);markEdited(c);save();}
function clearMask(){const c=curCase();if(!c)return;
  ensureLabels(c)[state[c.id].active].fill(0);state[c.id].edited=true;overlayVisible=true;
  document.getElementById("tOverlay").classList.add("on");renderOverlay(c);markEdited(c);save();}
function toggleOverlay(){overlayVisible=!overlayVisible;
  document.getElementById("tOverlay").classList.toggle("on",overlayVisible);
  const c=curCase();if(c)renderOverlay(c);}
function applyBC(){                    // brightness/contrast is a view aid on the fibre image only
  const b=document.getElementById("tBright").value,c=document.getElementById("tContrast").value;
  const bg=document.getElementById("lbBg");if(bg)bg.style.filter=`brightness(${b}) contrast(${c})`;}
function resetBC(){document.getElementById("tBright").value=1;document.getElementById("tContrast").value=1;applyBC();}

function paintLb(){
  const c=curCase(); if(!c) return;
  const f=c.features, st=state[c.id], sug=suggested(c);
  layoutCanvases(c);
  if(bgImgEl){bgImgEl.onload=null;}
  bgImgEl=new Image();
  bgImgEl.onload=()=>{const bg=document.getElementById("lbBg");
    bg.getContext("2d").drawImage(bgImgEl,0,0,c.geom.edit_w,c.geom.edit_h);renderOverlay(c);applyBC();};
  bgImgEl.src=c.edit_img;
  renderLabelChips(c);
  document.getElementById("lbId").textContent=c.id.replace("myotube_","proposal #");
  document.getElementById("lbSuggest").textContent = DATA.blind ? "blind repeat — no hint shown"
    : (c.learned_action ? `model suggests ${c.learned_action} · p=${c.learned_proba}`
                        : `suggested ${sug} · shape prior ${c.prior}`);
  const F=[["length",fmt(f.length_um)+" µm"],["width",fmt(f.width_um)+" µm"],
    ["area",fmt(f.area_um2)+" µm²"],["aspect",fmt(f.aspect)],
    ["solidity",fmt(f.solidity)],["in territory",Math.round(f.territory_overlap*100)+"%"],
    ["fiber signal",fmt(f.fiber_mean)],["edge",f.touches_border?"yes (truncated)":"no"]];
  document.getElementById("lbFeats").innerHTML=F.map(([k,v])=>
    `<div class="lb-feat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  document.querySelectorAll(".lb-decide button").forEach(b=>b.classList.remove("on"));
  const map={accept:".b-accept",ambiguous:".b-ambig",reject:".b-reject"};
  const el=document.querySelector(".lb-decide "+map[st.action]); if(el)el.classList.add("on");
  document.getElementById("tOverlay").classList.toggle("on",overlayVisible);
  document.getElementById("lbReason").value=st.reason||"";
  // hypothesis + assign controls only when this proposal can be split
  const split=hasSplit(c);
  const hb=document.getElementById("lbHyp"),ta=document.getElementById("tAssign");
  hb.style.display=split?"":"none"; ta.style.display=split?"":"none";
  if(split)hb.textContent=hypLabel(c); else if(tool==="assign")setTool("brush");
  markEdited(c);
  const vis=visibleList(),i=vis.findIndex(x=>x.id===lbId);
  document.getElementById("lbPos").textContent=i>=0?`${i+1} / ${vis.length}`:"";
}
function setLb(action){if(!lbId)return;state[lbId].action=action;stampDecision(lbId);paintLb();paintDecide(lbId);renderCounts();save();}
function stepLb(d){const vis=visibleList();let i=vis.findIndex(x=>x.id===lbId);
  if(i<0)return;i=(i+d+vis.length)%vis.length;lbId=vis[i].id;paintLb();}

(function wireEditor(){
  const mk=document.getElementById("lbMask");
  mk.addEventListener("pointerdown",e=>{const c=curCase();if(!c)return;
    if(tool==="assign"){assignSegmentAt(c,e.clientX,e.clientY);return;}
    painting=true;mk.setPointerCapture(e.pointerId);paintAt(c,e.clientX,e.clientY);});
  mk.addEventListener("pointermove",e=>{if(!painting)return;const c=curCase();if(c)paintAt(c,e.clientX,e.clientY);});
  mk.addEventListener("pointerup",()=>{if(painting){painting=false;save();}});
  mk.addEventListener("pointerleave",()=>{if(painting){painting=false;save();}});
})();
document.getElementById("lb").addEventListener("click",e=>{if(e.target.id==="lb")closeLb();});
window.addEventListener("resize",()=>{if(lbId)paintLb();});
function decideNext(action){if(!lbId)return;
  const vis=visibleList(),i=vis.findIndex(x=>x.id===lbId);
  const nxt=(i>=0&&vis.length>1)?vis[(i+1)%vis.length].id:null;
  setLb(action);
  if(nxt&&DATA.cases.some(x=>x.id===nxt)){lbId=nxt;paintLb();}}
function nudgeBright(d){const el=document.getElementById("tBright");
  el.value=Math.min(3.5,Math.max(0.3,(+el.value)+d));applyBC();}
document.addEventListener("keydown",e=>{
  if(e.target&&/^(INPUT|TEXTAREA)$/.test(e.target.tagName))return;   // don't hijack typing
  const km=document.getElementById("kbmap");
  if(e.key==="?"){e.preventDefault();km.classList.toggle("open");return;}
  if(km.classList.contains("open")){if(e.key==="Escape")km.classList.remove("open");return;}
  if(!document.getElementById("lb").classList.contains("open"))return;
  const k=e.key.toLowerCase();
  if(e.key==="Escape")closeLb();
  else if(e.key==="ArrowRight")stepLb(1);else if(e.key==="ArrowLeft")stepLb(-1);
  else if(e.key==="ArrowUp"){e.preventDefault();nudgeBright(0.15);}
  else if(e.key==="ArrowDown"){e.preventDefault();nudgeBright(-0.15);}
  else if(k==="a"||e.key==="Enter")decideNext("accept");
  else if(k==="r")decideNext("reject");
  else if(k==="x")decideNext("ambiguous");
  else if(k==="b")setTool("brush");else if(k==="e")setTool("erase");else if(k==="g")setTool("assign");
  else if(k==="h"){e.preventDefault();cycleHypothesis();}
  else if(k==="s"){e.preventDefault();splitIntoCards();}
  else if(k==="l")toggleOverlay();
  else if(k==="u"){e.preventDefault();resetMask();}
  else if(e.key==="Backspace"||e.key==="Delete"){e.preventDefault();clearMask();}
  else if(k>="1"&&k<="5"){const c=curCase(),idx=+k-1;if(c&&idx<ensureLabels(c).length)setActiveLabel(idx);}
});

document.getElementById("filters").querySelectorAll("button").forEach(b=>b.onclick=()=>{
  flt=b.dataset.f;
  document.querySelectorAll("#filters button").forEach(x=>x.classList.toggle("on",x===b));
  build();
});

// Idea 2: capture (original proposal mask, corrected mask, reason) as a correction pair.
function correctionInfo(c,labels){
  const N=c.geom.edit_w*c.geom.edit_h,orig=decodeRle(c.mask_rle),u=new Uint8Array(N);
  labels.forEach(m=>{for(let i=0;i<N;i++)if(m[i])u[i]=1;});
  let added=0,removed=0,origN=0,edN=0;
  for(let i=0;i<N;i++){if(orig[i])origN++;if(u[i])edN++;
    if(u[i]&&!orig[i])added++;if(!u[i]&&orig[i])removed++;}
  let kind;
  if(labels.length>1)kind="split";
  else if(added>origN*0.15&&added>=removed)kind="too_short";   // machine label stopped short
  else if(removed>origN*0.15&&removed>added)kind="spillover";
  else kind="reshape";
  return {kind,added_px:added,removed_px:removed,orig_px:origN,edited_px:edN,n_labels:labels.length};
}
function decisions(){
  const out={stem:DATA.stem,
    reviewer:DATA.reviewer||"",                    // WHO reviewed (provenance)
    session_started_at:SESSION_T0, exported_at:nowISO(),   // WHEN (washout verification)
    decisions:{}};
  DATA.cases.forEach(c=>{const st=state[c.id];
    const d={action:st.action,note:st.note,features:c.features,edited:!!st.edited,decided_at:st.t||null};
    if(st.edited&&st.labels){
      const nz=st.labels.filter(nonEmpty);
      const masks=nz.map(m=>encodeRle(m,c.geom.edit_h,c.geom.edit_w));
      if(masks.length){
        d.labels_rle=masks; d.geom=c.geom;
        d.original_rle=c.mask_rle;                 // machine proposal, paired with the correction
        d.correction=correctionInfo(c,nz);
        d.reason=st.reason||d.correction.kind;     // your override, else auto-inferred
      }
    }
    out.decisions[c.id]=d;});
  return out;
}
function downloadJson(){
  const blob=new Blob([JSON.stringify(decisions(),null,2)],{type:"application/json"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);
  a.download=DATA.stem+".decisions.json";a.click();
}
function copyJson(){navigator.clipboard&&navigator.clipboard.writeText(JSON.stringify(decisions(),null,2));
  flash("copied to clipboard");}
function sameGeom(a,b){return a&&b&&a.edit_h===b.edit_h&&a.edit_w===b.edit_w;}
function applyImport(obj){
  if(!obj||!obj.decisions){flash("⚠ not a decisions file");return;}
  let n=0,masks=0;
  DATA.cases.forEach(c=>{const d=obj.decisions[c.id];if(!d)return;
    const st=state[c.id];
    if(d.action)st.action=d.action; st.note=d.note||""; st.t=d.decided_at||st.t||null;
    st.labels=null;st.active=0;st.edited=!!d.edited;
    if(d.edited&&sameGeom(d.geom,c.geom)){
      if(d.labels_rle&&d.labels_rle.length){st.labels=d.labels_rle.map(r=>decodeRle(r));masks++;}
      else if(d.mask_rle){st.labels=[decodeRle(d.mask_rle)];masks++;}
    } else if(d.edited){st.edited=false;}   // geom changed -> keep the decision, drop stale mask
    n++;});
  build();save();
  flash("✓ restored "+n+" decisions"+(masks?(" ("+masks+" with edited masks)"):""));
}
function importFromFile(e){const f=e.target.files[0];if(!f)return;
  const r=new FileReader();r.onload=()=>{try{applyImport(JSON.parse(r.result));}catch(err){flash("⚠ bad JSON file");}};
  r.readAsText(f);e.target.value="";}
function toggleIO(){const p=document.getElementById("ioPanel");p.hidden=!p.hidden;
  if(!p.hidden){document.getElementById("ioOut").value=JSON.stringify(decisions(),null,2);}}
function restoreFromBox(){const t=document.getElementById("ioIn").value.trim();
  if(!t){flash("paste your saved text first");return;}
  try{applyImport(JSON.parse(t));}catch(e){flash("⚠ could not read that text");}}
function pasteRestore(){restoreFromBox();}
function stamp(){try{return new Date().toLocaleTimeString();}catch(e){return "";}}
function nowISO(){try{return new Date().toISOString();}catch(e){return "";}}
const SESSION_T0=nowISO();                 // when this review session started (for washout verification)
function stampDecision(id){if(state[id])state[id].t=nowISO();}   // per-decision timestamp
function flash(msg){const el=document.getElementById("saved");if(el)el.textContent=msg;}
function save(){try{
  const ser={};                          // persist masks as compact RLE (typed arrays don't survive JSON)
  for(const id in state){const s=state[id],c=DATA.cases.find(x=>x.id===id);
    const o={action:s.action,note:s.note,edited:!!s.edited,hyp:s.hyp||0,active:s.active||0,reason:s.reason||"",t:s.t||null};
    if(s.edited&&s.labels&&c)o.labels_rle=s.labels.filter(nonEmpty).map(m=>encodeRle(m,c.geom.edit_h,c.geom.edit_w));
    ser[id]=o;}
  localStorage.setItem(LSKEY,JSON.stringify({when:stamp(),state:ser}));
}catch(e){}
  const p=document.getElementById("ioPanel");
  if(p&&!p.hidden)document.getElementById("ioOut").value=JSON.stringify(decisions(),null,2);}
function load(){try{const raw=localStorage.getItem(LSKEY);if(!raw)return null;
  const o=JSON.parse(raw);if(o&&o.state){DATA.cases.forEach(c=>{const s=o.state[c.id];if(!s)return;
    state[c.id]={action:s.action||suggested(c),note:s.note||"",edited:!!s.edited,hyp:s.hyp||0,active:s.active||0,
      reason:s.reason||"",t:s.t||null,labels:(s.labels_rle&&s.labels_rle.length)?s.labels_rle.map(r=>decodeRle(r)):null};});
    return o.when;}}catch(e){}return null;}

function renderModel(){
  const m=DATA.model||{};
  const el=document.getElementById("model");
  if(DATA.blind){
    el.innerHTML=`<span class="badge cold">blind repeat · reliability check</span>
      <div style="line-height:1.6">These are <b>${DATA.cases.length} cases you reviewed before</b>, shown in
      random order with <b>no hint of your earlier call and no model suggestion</b> — on purpose. Decide each
      one <b>fresh</b>, as if for the first time.
      <ul style="margin:6px 0 0;padding-left:18px">
        <li><b>Accept</b> only if you are confident it is one real, complete myotube.</li>
        <li><b>Reject</b> if it is clearly not a myotube (debris, fragment, background).</li>
        <li><b>Ambiguous</b> if you genuinely cannot tell — <b>when in doubt, do NOT Accept.</b> Ambiguous or Reject is the safe choice.</li>
        <li>Edge fibre (<span class="feat warn">edge</span>)? Still <b>Accept</b> it — it is recorded as
          <b>border-truncated automatically</b>. Do not guess its missing length.</li>
      </ul>
      Every card starts on <b>Ambiguous</b> so nothing counts as accepted until you actively choose it.
      When finished, click <b>⇅ Save / Restore → Copy all</b> and send me the text.</div>
      ${DATA.blind_note?`<div style="margin-top:8px;padding:8px 12px;border-radius:8px;background:var(--accept-bg);
        border:1px solid var(--accept);color:var(--ink);line-height:1.6"><b>This round's rule:</b> ${DATA.blind_note}</div>`:""}`;
    return;
  }
  if(m.status==="fitted"){
    el.innerHTML=`<span class="badge learn">model active</span>
      <div><b>Learned rule:</b> ${esc(m.rule)} &nbsp;·&nbsp;
      trained on ${m.n} decisions · accuracy ${m.train_accuracy}. Suggestions below are the
      model's, with probabilities.</div>`;
  }else{
    const need=Math.max(0,12-(m.n||0));
    el.innerHTML=`<span class="badge cold">learning</span>
      <div>The model isn't set yet — it starts pre-deciding after <b>${need} more</b>
      accept/reject decisions. For now, suggestions come from a transparent shape prior
      (elongated, in-territory, non-edge proposals rank first). Confirm the obvious ones fast.</div>`;
  }
}

document.getElementById("stem").textContent=DATA.stem;
document.getElementById("sub").textContent= DATA.blind
  ? `${DATA.cases.length} cases · random order · blind second pass`
  : `${DATA.cases.length} proposals${DATA.total>DATA.cases.length?" (top "+DATA.cases.length+" of "+DATA.total+" by likelihood)":""} · sorted most-myotube-like first`;
const when=load();
document.getElementById("intFilter").max=Math.ceil(MAX_FIBER);
try{const v=localStorage.getItem(LSKEY+"_int");if(v){document.getElementById("intFilter").value=v;intensityMin=+v;}}catch(e){}
document.getElementById("intVal").textContent=intensityMin;
document.getElementById("intCount").textContent=intensityMin>0
  ?`${DATA.cases.filter(tooDark).length} of ${DATA.cases.length} below threshold (too dark)`:"off — show all";
renderModel();build();
if(when) flash("restored from "+when);
</script></body></html>"""


def build_page(stem: str, cases: list[dict], total: int, out_path: str | Path,
               pixel_um: float = 0.6493, blind: bool = False, blind_note: str = "",
               reviewer: str = "") -> str:
    """Attach model suggestions to cases and write review.html. Returns the path.

    ``blind=True`` builds a blind-repeat page: NO model suggestion, NO shape prior
    shown, and the default disposition is the conservative ``ambiguous`` (nothing is
    pre-accepted). Used for the single-operator G-SO1 reliability check.
    """
    summary = M.load_summary()
    model_state = {"status": summary.get("status", "cold"), "n": summary.get("n", 0),
                   "rule": summary.get("rule", ""),
                   "train_accuracy": summary.get("train_accuracy")}
    if not blind:                                       # blind: never leak a model hint
        for c in cases:
            action, proba = M.predict_default(c["features"])
            if action is not None:
                c["learned_action"] = action
                c["learned_proba"] = proba
    data = {"stem": stem, "cases": cases, "total": total, "model": model_state,
            "pixel_um": pixel_um, "blind": bool(blind), "blind_note": blind_note,
            "reviewer": reviewer}
    html = (PAGE.replace("__DATA__", json.dumps(data))
                .replace("__STEM__", stem))
    out_path = Path(out_path)
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)
