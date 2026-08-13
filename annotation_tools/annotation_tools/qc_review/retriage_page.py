"""Static re-triage page: 839 ambiguous proposals -> six actionable categories.

Deliberately a separate page from `page.py`. That one is hardwired to three
actions and is the workflow that produced the frozen 1,800-decision first pass;
it is not worth destabilising for a different task with a different vocabulary.

Design goals, in order:

1. **Confirm, don't author.** Every card opens on the machine's predicted
   category with its reason visible, so a correct suggestion is zero clicks and a
   wrong one is one click.
2. **Highest yield first.** Cases arrive ordered by expected promotion rate, so
   stopping after any batch still captures most of the recoverable data.
3. **No silent incompleteness.** A category that was never explicitly confirmed
   exports `decided_at: null`. The page counts explicit decisions and refuses to
   present itself as finished until every card is touched -- this is backlog item
   D from the lane resume notes, and it is why the first G-SO1 round failed
   provenance.
4. **Full provenance.** Reviewer identity, session start, export time, and a
   per-decision UTC timestamp, matching the R04 contract.

Nothing here mutates the frozen first-pass artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

CATEGORY_UI = [
    ("complete", "1", "Complete", "One whole, fully measurable myotube", "#22c55e"),
    ("branched_one_myotube", "2", "Branched (still one)",
     "Branches, but it is one myotube and measurable", "#4ade80"),
    ("fragment_too_short", "3", "Fragment (too short)",
     "A piece of a longer fibre, not a whole one", "#f59e0b"),
    ("merged_too_long", "4", "Merged (too long)",
     "Two or more fibres stuck together", "#fb923c"),
    ("unresolvable", "5", "Unresolvable",
     "Cannot be separated in 2-D; leave it out", "#a78bfa"),
    ("not_myotube", "6", "Not a myotube", "Debris or background", "#ef4444"),
]

_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#0b0f0e;color:#e8f0ee;
     font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:20;background:#0f1614;border-bottom:1px solid #1f2d29;
       padding:12px 20px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:600}
.sub{color:#8aa39c;font-size:12px}
.counts{display:flex;gap:10px;flex-wrap:wrap;margin-left:auto;align-items:center}
.pill{padding:3px 9px;border-radius:999px;font-size:12px;background:#16211e;border:1px solid #24332e}
.bar{height:6px;background:#16211e;border-radius:999px;overflow:hidden;width:180px}
.bar>i{display:block;height:100%;background:#22c55e;width:0}
button{font:inherit;cursor:pointer;border-radius:8px;border:1px solid #24332e;
       background:#16211e;color:#e8f0ee;padding:6px 12px}
button:hover{border-color:#3a5049}
button.primary{background:#22c55e;color:#04170d;border-color:#22c55e;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
      gap:14px;padding:18px 20px 80px}
.card{background:#111a18;border:1px solid #1f2d29;border-radius:12px;overflow:hidden;
      display:flex;flex-direction:column}
.card.undecided{border-color:#3f3a1f}
.card img{width:100%;display:block;background:#060a09;cursor:zoom-in}
.meta{padding:8px 10px;font-size:12px;color:#9fb5af;border-bottom:1px solid #1a2422}
.why{padding:7px 10px;font-size:11px;color:#7f978f;background:#0d1413;font-style:italic}
.cats{display:grid;grid-template-columns:1fr 1fr;gap:4px;padding:8px}
.cats button{padding:5px 6px;font-size:11.5px;text-align:left;line-height:1.25}
.cats button.on{color:#04170d;font-weight:700}
dialog{border:none;background:#0f1614;color:#e8f0ee;border-radius:14px;padding:0;max-width:94vw}
dialog::backdrop{background:rgba(0,0,0,.72)}
.lb{padding:14px;display:flex;flex-direction:column;gap:10px}
.lb img{max-width:min(78vw,900px);max-height:70vh;border-radius:8px;background:#060a09}
footer{position:fixed;bottom:0;left:0;right:0;background:#0f1614;border-top:1px solid #1f2d29;
       padding:10px 20px;display:flex;gap:12px;align-items:center;z-index:20}
.warn{color:#fbbf24}
textarea{width:100%;height:38vh;background:#060a09;color:#cfe;border:1px solid #24332e;
         border-radius:8px;padding:10px;font-family:ui-monospace,Consolas,monospace;font-size:12px}
"""

_JS = r"""
const CATS = DATA.categories;
const state = {};
// Keyed by `uid` (well/id), never by `id`: proposal ids repeat across wells, so
// keying by id would make two wells' cards share one decision.
DATA.cases.forEach(c => state[c.uid] = {cat: c.machine_category, t: null});

function nowISO(){ return new Date().toISOString().replace(/\.\d+Z$/, 'Z'); }
function decidedCount(){ return DATA.cases.filter(c => state[c.uid].t).length; }

function renderCounts(){
  const n = DATA.cases.length, d = decidedCount();
  const tally = {};
  DATA.cases.forEach(c => { const k = state[c.uid].cat; tally[k] = (tally[k]||0)+1; });
  document.getElementById('counts').innerHTML =
    CATS.map(([k,,label,,color]) =>
      `<span class="pill" style="border-color:${color}66">${label}: <b>${tally[k]||0}</b></span>`
    ).join('') +
    `<span class="pill" ${d<n?'style="border-color:#fbbf24"':''}>decided <b>${d}</b>/${n}</span>`;
  document.getElementById('progress').style.width = (100*d/n) + '%';
  document.getElementById('exportBtn').textContent =
    d < n ? `Export (${n-d} still untouched)` : 'Export all decisions';
}

const BY_UID = {};
DATA.cases.forEach(c => BY_UID[c.uid] = c);

function setCat(uid, cat, explicit){
  state[uid].cat = cat;
  if (explicit) state[uid].t = nowISO();
  const card = document.getElementById('card_' + BY_UID[uid].dom_id);
  if (card){
    card.classList.toggle('undecided', !state[uid].t);
    card.querySelectorAll('.cats button').forEach(b => {
      const on = b.dataset.c === cat;
      b.classList.toggle('on', on);
      b.style.background = on ? b.dataset.color : '';
      b.style.borderColor = on ? b.dataset.color : '';
    });
  }
  renderCounts(); save();
}

function cardHTML(c){
  const f = c.features;
  const u = encodeURIComponent(c.uid);
  return `<div class="card undecided" id="card_${c.dom_id}">
    <img src="${c.thumb}" onclick="openLb(decodeURIComponent('${u}'))" loading="lazy" alt="">
    <div class="meta"><b>${c.well}</b> · ${c.id}<br>
      ${f.length_um} µm · aspect ${f.aspect} · solidity ${f.solidity}</div>
    <div class="why">machine: ${c.machine_why}</div>
    <div class="cats">${CATS.map(([k,key,label,,color]) =>
      `<button data-c="${k}" data-color="${color}" title="${key}: ${label}"
        onclick="setCat(decodeURIComponent('${u}'),'${k}',true)">${key}. ${label}</button>`).join('')}</div>
  </div>`;
}

function build(){
  document.getElementById('grid').innerHTML = DATA.cases.map(cardHTML).join('');
  DATA.cases.forEach(c => setCat(c.uid, state[c.uid].cat, false));
  restore();
}

let lbUid = null;
function openLb(uid){
  lbUid = uid;
  const c = BY_UID[uid];
  document.getElementById('lbImg').src = c.edit_img || c.thumb;
  document.getElementById('lbTitle').textContent = `${c.well} · ${c.id}`;
  document.getElementById('lbWhy').textContent = c.machine_why;
  document.getElementById('lb').showModal();
}
document.addEventListener('keydown', e => {
  const hit = CATS.find(([,key]) => key === e.key);
  if (!hit) return;
  if (lbUid && document.getElementById('lb').open){ setCat(lbUid, hit[0], true); return; }
});

const KEY = 'retriage_' + DATA.batch_id;
function save(){
  const out = {};
  DATA.cases.forEach(c => out[c.uid] = state[c.uid]);
  try { localStorage.setItem(KEY, JSON.stringify({s: out, started: DATA.session_started_at})); }
  catch(_){}
}
function restore(){
  try {
    const raw = localStorage.getItem(KEY); if(!raw) return;
    const saved = JSON.parse(raw).s || {};
    Object.keys(saved).forEach(uid => { if(state[uid]){ state[uid] = saved[uid]; } });
    DATA.cases.forEach(c => setCat(c.uid, state[c.uid].cat, false));
  } catch(_){}
}

function acceptAllMachine(){
  if(!confirm('Confirm the machine suggestion for every UNTOUCHED card?\n\n'+
              'Only do this after spot-checking - it stamps them as your decisions.')) return;
  DATA.cases.forEach(c => { if(!state[c.uid].t) setCat(c.uid, state[c.uid].cat, true); });
}

function exportAll(){
  const n = DATA.cases.length, d = decidedCount();
  if (d < n && !confirm(
      `${n-d} of ${n} cards were never explicitly confirmed.\n\n`+
      `They will export with decided_at = null and will be EXCLUDED from training `+
      `promotion. Export anyway?`)) return;
  const payload = {
    schema: 'retriage.v1',
    batch_id: DATA.batch_id,
    reviewer: DATA.reviewer,
    session_started_at: DATA.session_started_at,
    exported_at: nowISO(),
    n_cases: n, n_explicitly_decided: d,
    categories: CATS.map(c => c[0]),
    decisions: {}
  };
  DATA.cases.forEach(c => {
    // keyed by uid; `id` alone repeats across wells and would overwrite
    payload.decisions[c.uid] = {
      well: c.well, id: c.id, category: state[c.uid].cat, decided_at: state[c.uid].t,
      machine_category: c.machine_category, first_pass: 'ambiguous'
    };
  });
  const text = JSON.stringify(payload, null, 1);
  document.getElementById('outText').value = text;
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
  a.download = DATA.batch_id + '.retriage.json';
  a.click();
}
build();
"""


def build_retriage_page(cases: list[dict], out_path: str | Path, *, batch_id: str,
                        reviewer: str, session_started_at: str,
                        batch_index: int = 1, batch_total: int = 1,
                        note: str = "") -> str:
    """Write one bounded re-triage batch as a self-contained static page.

    ``cases`` must already be ordered (highest expected yield first) and carry
    ``machine_category`` / ``machine_why`` from
    :func:`annotation_tools.qc_review.retriage.classify`.
    """
    if not reviewer:
        raise ValueError("reviewer is required: an export without an identified "
                         "reviewer cannot be used as evidence")

    payload = {
        "cases": cases,
        "categories": CATEGORY_UI,
        "batch_id": batch_id,
        "reviewer": reviewer,
        "session_started_at": session_started_at,
    }
    promotable = sum(1 for c in cases
                     if c["machine_category"] in ("complete", "branched_one_myotube"))
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Re-triage {batch_id}</title><style>{_CSS}</style></head><body>
<header>
  <div>
    <h1>Ambiguous re-triage &mdash; batch {batch_index} of {batch_total}</h1>
    <div class="sub">{len(cases)} cases &middot; reviewer <b>{reviewer}</b> &middot;
      machine suggests {promotable} promotable &middot; highest expected yield first</div>
  </div>
  <div class="counts" id="counts"></div>
  <div class="bar"><i id="progress"></i></div>
</header>
{f'<div class="why" style="padding:10px 20px">{note}</div>' if note else ''}
<div class="grid" id="grid"></div>
<footer>
  <button onclick="acceptAllMachine()">Confirm machine suggestion for untouched</button>
  <button class="primary" id="exportBtn" onclick="exportAll()">Export all decisions</button>
  <span class="sub">Click a thumbnail to enlarge; keys <b>1</b>&ndash;<b>6</b> decide the
    enlarged card. Progress autosaves in this browser.</span>
</footer>
<dialog id="lb"><div class="lb">
  <b id="lbTitle"></b><img id="lbImg" alt=""><div class="why" id="lbWhy"></div>
  <div><button onclick="document.getElementById('lb').close()">Close</button>
    <span class="sub">press 1&ndash;6 to categorise</span></div>
</div></dialog>
<dialog id="out"><div class="lb">
  <b>Re-triage export</b>
  <span class="sub">Copy or download this and send it back.</span>
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
