"""Stage 4 -- build the interactive review page from flags.json.

Serverless. The crop shows the RAW image by default; MY proposed overlay (traces + split marks) is
an opt-in VECTOR layer you can toggle, and you can:
  * click a proposed split to toggle it -- it highlights on the figure (not just a coordinate),
  * click to drop your OWN split point(s),
  * "Reject & draw my own" -- draw the correct trace(s) yourself; the pipeline applies YOUR drawing
    (decisions.json action 'redraw') and the learner records the correction.
On submit it produces decisions.json; save it into the stage4 folder and re-run with --resume.

Usage: python build_review_html.py --out <stage4 dir>
"""
from __future__ import annotations
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Myotube QC review — __STEM__</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#111;color:#eee}
 header{position:sticky;top:0;background:#1b1b1b;padding:10px 16px;border-bottom:1px solid #333;z-index:5}
 header h1{font-size:16px;margin:0 0 6px} .muted{color:#9aa;font-size:12px}
 button{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:5px;padding:5px 9px;cursor:pointer;margin:2px 4px 2px 0}
 button:hover{background:#333} button.on{background:#274; border-color:#4a7}
 .case{display:flex;gap:14px;padding:14px 16px;border-bottom:1px solid #2a2a2a;align-items:flex-start}
 .imgwrap{position:relative;display:inline-block}
 .imgwrap img{max-width:540px;max-height:540px;border:1px solid #444;background:#000;display:block}
 .imgwrap canvas{position:absolute;left:0;top:0;cursor:crosshair}
 .ctrl{min-width:330px} .ctrl h3{margin:0 0 4px;font-size:14px}
 .split h3{color:#ff8a8a} .merge h3{color:#9ad0ff} .occluded h3{color:#ffd27a}
 .badge{font-size:10px;padding:1px 6px;border-radius:8px;background:#444;margin-left:6px}
 .badge.auto{background:#664} .badge.draw{background:#274}
 label{display:block;margin:3px 0;font-size:13px} .pt{margin-left:18px;font-size:12px;color:#fbb}
 .upt{margin-left:18px;font-size:12px;color:#8ff;margin-top:2px}
 textarea{width:100%;height:150px;background:#000;color:#9f9;font-family:monospace;font-size:12px}
 .conf{font-size:11px;color:#8c8} .hint{font-size:11px;color:#8ff;margin:4px 0}
</style></head><body>
<header>
 <h1>Myotube QC review — <span id="stem"></span></h1>
 <div class="muted" id="summary"></div>
 <div style="margin-top:8px">
   <button onclick="setAll('proposal')">Accept all proposals</button>
   <button onclick="setAll('safe')">Reject all edits</button>
   <button onclick="render()">Refresh JSON</button>
   <button onclick="download()">⬇ Download decisions.json</button>
   &nbsp;show: <button onclick="setFilter('all')">All</button>
   <button onclick="setFilter('merge')">Merges</button>
   <button onclick="setFilter('split')">Splits</button>
   <button onclick="setFilter('occluded')">Occluded</button>
 </div>
 <div class="muted" style="margin-top:5px">
   <span id="saved">your progress auto-saves in this browser</span> &nbsp;
   <button onclick="clearProgress()">Clear saved progress</button>
 </div>
</header>
<div id="cases"></div>
<div style="padding:16px">
 <h3>decisions.json (save into the stage4_qc folder, then run the orchestrator with --resume)</h3>
 <textarea id="out" readonly></textarea>
</div>
<script>
const DATA = __DATA__;
const state = {};
function esc(s){return (s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function gapKey(p){return p[0]+','+p[1];}
function gapMask(c){const g=new Set((c.gap_splits||[]).map(gapKey));return (c.proposed_splits||[]).map(p=>g.has(gapKey(p)));}
function noneMask(c){return (c.proposed_splits||[]).map(()=>false);}
function defForCase(c){
  const base={note:'', showProposal:false, drawMode:false, polys:[], cur:[]};
  if(c.type==='merge') return {...base, action:c.learned_default||(c.auto?'merge':'separate')};
  if(c.type==='occluded') return {...base, action:c.learned_default||'drop'};
  const hasGap=(c.gap_splits||[]).length>0;
  let act=c.learned_default||(c.auto?'split':'keep');
  if(act==='split' && !hasGap) act='keep';   // kink-only learned split isn't auto-applicable -> keep
  return {...base, action:act, points:(act==='split'?gapMask(c):noneMask(c)), n:2, user_points:[]};
}
DATA.cases.forEach(c=>state[c.id]=defForCase(c));

// --- persistence: auto-save the whole review state to this browser (survives close/reopen) ---
const LSKEY='myoqc_review_'+DATA.stem;
function stamp(){try{return new Date().toLocaleTimeString();}catch(e){return '';}}
function saveProgress(){
  try{ localStorage.setItem(LSKEY, JSON.stringify({when:stamp(), state:state}));
       const el=document.getElementById('saved'); if(el)el.textContent='✓ progress saved '+stamp(); }
  catch(e){ const el=document.getElementById('saved'); if(el)el.textContent='⚠ could not save ('+e+')'; }
}
function fixCaseState(s,c){
  if(!s) return defForCase(c);
  if(typeof s.action!=='string') s.action=defForCase(c).action;
  if(c.type==='split'){ const n=(c.proposed_splits||[]).length;
    s.points=Array.from({length:n},(_,i)=>!!(s.points&&s.points[i]));
    if(!Array.isArray(s.user_points)) s.user_points=[]; if(typeof s.n!=='number') s.n=2; }
  if(!Array.isArray(s.polys)) s.polys=[]; if(!Array.isArray(s.cur)) s.cur=[];
  s.showProposal=!!s.showProposal; s.drawMode=!!s.drawMode; if(typeof s.note!=='string') s.note='';
  return s;
}
function loadProgress(){
  try{ const raw=localStorage.getItem(LSKEY); if(!raw) return null; const obj=JSON.parse(raw);
       if(obj&&obj.state){ DATA.cases.forEach(c=>{ if(obj.state[c.id]) state[c.id]=fixCaseState(obj.state[c.id],c); }); return obj.when||'a previous session'; } }
  catch(e){} return null;
}
function clearProgress(){
  try{ localStorage.removeItem(LSKEY); }catch(e){}
  DATA.cases.forEach(c=>state[c.id]=defForCase(c)); buildCards(); render();
  const el=document.getElementById('saved'); if(el)el.textContent='saved progress cleared';
}
function setAll(mode){
  DATA.cases.forEach(c=>{const st=state[c.id];
    st.drawMode=false; st.polys=[]; st.cur=[];            // clear any hand-drawn traces...
    if(c.type==='split') st.user_points=[];               // ...and hand-clicked split points
    if(c.type==='merge') st.action=(mode==='proposal')?'merge':'separate';
    else if(c.type==='occluded') st.action='drop';
    else { if(mode==='proposal' && c.longest_dark_px>0){st.action='split';st.points=gapMask(c);} else {st.action='keep';st.points=noneMask(c);} }
  });
  buildCards(); render();
}
let flt='all';
function setFilter(t){flt=t;buildCards();}

// click<->coordinate mapping
function imgToFull(c,img,clientX,clientY){const r=img.getBoundingClientRect();
  const natX=(clientX-r.left)*(img.naturalWidth/r.width), natY=(clientY-r.top)*(img.naturalHeight/r.height);
  return [Math.round((c.bbox[0]+natX/c.crop_scale)*10)/10, Math.round((c.bbox[1]+natY/c.crop_scale)*10)/10];}
function fullToDisp(c,img,fx,fy){const r=img.getBoundingClientRect();
  return [(fx-c.bbox[0])*c.crop_scale*(r.width/img.naturalWidth), (fy-c.bbox[1])*c.crop_scale*(r.height/img.naturalHeight)];}

function drawCanvas(c){
  const img=document.getElementById('img_'+c.id), canvas=document.getElementById('cv_'+c.id);
  if(!img||!canvas) return;
  const r=img.getBoundingClientRect();
  if(!r.width){ img.onload=()=>drawCanvas(c); return; }
  canvas.width=r.width; canvas.height=r.height; canvas.style.width=r.width+'px'; canvas.style.height=r.height+'px';
  const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,r.width,r.height); const st=state[c.id];
  const D=(fx,fy)=>fullToDisp(c,img,fx,fy);
  if(st.showProposal){ ctx.strokeStyle='rgba(255,235,0,0.85)'; ctx.lineWidth=2.5;
    (c.trace_polys||[]).forEach(poly=>{ctx.beginPath();poly.forEach((p,k)=>{const d=D(p[0],p[1]);k?ctx.lineTo(d[0],d[1]):ctx.moveTo(d[0],d[1]);});ctx.stroke();}); }
  if(c.type==='split'){
    (c.proposed_splits||[]).forEach((p,i)=>{const d=D(p[0],p[1]); const on=st.points[i];
      ctx.strokeStyle=on?'#ff3b3b':'rgba(255,150,150,0.45)'; ctx.lineWidth=on?4:2;
      ctx.beginPath();ctx.moveTo(d[0]-11,d[1]-11);ctx.lineTo(d[0]+11,d[1]+11);ctx.moveTo(d[0]-11,d[1]+11);ctx.lineTo(d[0]+11,d[1]-11);ctx.stroke();
      if(on){ctx.fillStyle='#ff6b6b';ctx.font='12px sans-serif';ctx.fillText('split '+(i+1),d[0]+13,d[1]-3);}});
    (st.user_points||[]).forEach(p=>{const d=D(p[0],p[1]);ctx.strokeStyle='#19e0ff';ctx.lineWidth=2;
      ctx.beginPath();ctx.arc(d[0],d[1],7,0,2*Math.PI);ctx.stroke();
      ctx.beginPath();ctx.moveTo(d[0]-11,d[1]);ctx.lineTo(d[0]+11,d[1]);ctx.moveTo(d[0],d[1]-11);ctx.lineTo(d[0],d[1]+11);ctx.stroke();});
  }
  const drawPoly=(poly,col,dots)=>{if(!poly.length)return;ctx.strokeStyle=col;ctx.lineWidth=3;ctx.beginPath();
    poly.forEach((p,k)=>{const d=D(p[0],p[1]);k?ctx.lineTo(d[0],d[1]):ctx.moveTo(d[0],d[1]);});ctx.stroke();
    if(dots)poly.forEach(p=>{const d=D(p[0],p[1]);ctx.fillStyle=col;ctx.beginPath();ctx.arc(d[0],d[1],3.5,0,2*Math.PI);ctx.fill();});};
  (st.polys||[]).forEach((poly,k)=>{const col=COLORS[k%COLORS.length]; drawPoly(poly.pts,col,true);
    if(poly.pts.length){const m=poly.pts[Math.floor(poly.pts.length/2)];const d=D(m[0],m[1]);
      ctx.fillStyle=col;ctx.font='bold 15px sans-serif';ctx.fillText(String(k+1),d[0]+7,d[1]-7);}});
  if(st.cur&&st.cur.length)drawPoly(st.cur,'#aaffaa',true);
}
const COLORS=['#33ff77','#ffd11a','#19e0ff','#ff7ae0','#ff9933','#9b8cff','#7dff9b'];
function setRadio(c){const el=document.querySelector('[data-act="'+c.id+':'+state[c.id].action+'"]'); if(el)el.checked=true;}

function onCanvasClick(c,e){
  const st=state[c.id], img=document.getElementById('img_'+c.id);
  const full=imgToFull(c,img,e.clientX,e.clientY);
  if(st.drawMode){ st.cur.push(full); drawCanvas(c); render(); return; }
  if(c.type!=='split'){ return; }
  const r=img.getBoundingClientRect(), cx=e.clientX-r.left, cy=e.clientY-r.top;
  let best=-1,bd=14;
  (c.proposed_splits||[]).forEach((p,i)=>{const d=fullToDisp(c,img,p[0],p[1]);const dist=Math.hypot(d[0]-cx,d[1]-cy);if(dist<bd){bd=dist;best=i;}});
  if(best>=0){ st.points[best]=!st.points[best]; const cb=document.querySelector('[data-pt="'+c.id+':'+best+'"]'); if(cb)cb.checked=st.points[best]; }
  else { st.user_points.push(full); }
  st.action=(st.points.some(Boolean)||(st.user_points||[]).length)?'split':'keep'; setRadio(c);  // keep radio == emitted action
  drawCanvas(c); render();
}

function buildCards(){
  const root=document.getElementById('cases'); root.innerHTML='';
  DATA.cases.forEach(c=>{
    if(flt!=='all' && c.type!==flt) return;
    const st=state[c.id];
    const div=document.createElement('div'); div.className='case';
    const wrap=document.createElement('div'); wrap.className='imgwrap';
    const img=document.createElement('img'); img.id='img_'+c.id; img.loading='lazy'; img.src=c.crop_raw;
    const canvas=document.createElement('canvas'); canvas.id='cv_'+c.id;
    canvas.onclick=(e)=>onCanvasClick(c,e);
    wrap.appendChild(img); wrap.appendChild(canvas);
    img.onload=()=>drawCanvas(c); if(img.complete&&img.naturalWidth) drawCanvas(c);
    const ctrl=document.createElement('div'); ctrl.className='ctrl '+c.type;
    let h='<h3>'+c.type.toUpperCase()+' '+c.id+(c.auto?'<span class="badge auto">auto</span>':'')+
          (st.drawMode?'<span class="badge draw">drawing</span>':'')+'</h3>';
    h+='<div class="conf">confidence '+c.confidence+' — '+esc(c.reason)+
       (c.learned_default?(' · <span style="color:#caf">learned: '+c.learned_default+' (p='+c.learned_proba+')</span>'):'')+'</div>';
    h+='<button class="'+(st.showProposal?'on':'')+'" data-prop="'+c.id+'">'+(st.showProposal?'✓ my overlay shown':'Show my proposed overlay')+'</button>';
    h+='<button class="'+(st.drawMode?'on':'')+'" data-draw="'+c.id+'">'+(st.drawMode?'Stop drawing':'Reject mine & draw my own')+'</button>';
    if(st.drawMode){
      const nM=st.polys.length;
      h+='<div class="hint">Drawing <b>myotube '+(nM+1)+'</b>: click to place its vertices, then "Finish this myotube" and draw the next separate one.</div>';
      h+='<button data-finish="'+c.id+'">✓ Finish this myotube</button><button data-undo="'+c.id+'">Undo point</button><button data-clrdraw="'+c.id+'">Clear all</button>';
      st.polys.forEach((p,k)=>{h+='<div class="upt">▦ myotube '+(k+1)+': <input type="text" data-mlabel="'+c.id+':'+k+'" value="'+esc(p.label)+'" style="width:150px"> <a href="#" data-rmpoly="'+c.id+':'+k+'">delete</a></div>';});
      if(st.cur.length) h+='<div class="upt" style="color:#aaffaa">myotube '+(nM+1)+' in progress: '+st.cur.length+' point(s)</div>';
    } else if(c.type==='split'){
      h+='<div class="hint">Click a red ✕ to toggle my split (it highlights), or click elsewhere to add YOUR split point.</div>';
      h+=radio(c.id,'split','Split at the marked point(s)',st.action);
      (c.proposed_splits||[]).forEach((p,i)=>{h+='<div class="pt"><label><input type="checkbox" data-pt="'+c.id+':'+i+'" '+(st.points[i]?'checked':'')+'> my split '+(i+1)+'</label></div>';});
      if((st.user_points||[]).length) h+='<div class="pt" style="color:#8ff">your split points: '+st.user_points.length+' <a href="#" data-clrupt="'+c.id+'">clear</a></div>';
      h+=radio(c.id,'split_n','Split into N equal pieces',st.action)+' <input type="number" min="2" data-n="'+c.id+'" value="'+(st.n||2)+'" style="width:48px">';
      h+=radio(c.id,'keep','Keep as ONE myotube',st.action);
      h+=radio(c.id,'reject','Reject (not a real myotube)',st.action);
    } else if(c.type==='merge'){
      h+=radio(c.id,'merge','Merge into ONE myotube',st.action)+radio(c.id,'separate','Keep as SEPARATE',st.action);
    } else {
      h+=radio(c.id,'restore','Restore — a separate dim fibre',st.action)+radio(c.id,'drop','Drop — a bright fragment',st.action);
    }
    h+='<label>note: <input type="text" data-note="'+c.id+'" value="'+esc(st.note)+'" style="width:220px"></label>';
    ctrl.innerHTML=h; div.appendChild(wrap); div.appendChild(ctrl); root.appendChild(div);
  });
  bind();
}
function radio(id,val,label,cur){return '<label><input type="radio" name="act_'+id+'" data-act="'+id+':'+val+'" '+(cur===val?'checked':'')+'> '+label+'</label>';}
function bind(){
  document.querySelectorAll('[data-act]').forEach(el=>el.onchange=e=>{const[id,v]=e.target.dataset.act.split(':');state[id].action=v;state[id].drawMode=false;render();});
  document.querySelectorAll('[data-pt]').forEach(el=>el.onchange=e=>{const[id,i]=e.target.dataset.pt.split(':');const c=DATA.cases.find(x=>x.id===id);const st=state[id];st.points[+i]=e.target.checked;st.action=(st.points.some(Boolean)||(st.user_points||[]).length)?'split':'keep';setRadio(c);drawCanvas(c);render();});
  document.querySelectorAll('[data-n]').forEach(el=>el.onchange=e=>{state[e.target.dataset.n].n=+e.target.value;render();});
  document.querySelectorAll('[data-note]').forEach(el=>el.oninput=e=>{state[e.target.dataset.note].note=e.target.value;render();});
  document.querySelectorAll('[data-prop]').forEach(el=>el.onclick=e=>{const id=e.target.dataset.prop;state[id].showProposal=!state[id].showProposal;buildCards();render();});
  document.querySelectorAll('[data-draw]').forEach(el=>el.onclick=e=>{const id=e.target.dataset.draw;const c=DATA.cases.find(x=>x.id===id);const st=state[id];st.drawMode=!st.drawMode;
    if(st.drawMode){ st.action='redraw'; }
    else { st.action=(c.type==='split' && ((st.points&&st.points.some(Boolean))||(st.user_points||[]).length)) ? 'split' : defForCase(c).action; }  // resync action to live state on exit
    buildCards();render();});
  document.querySelectorAll('[data-finish]').forEach(el=>el.onclick=e=>{const st=state[e.target.dataset.finish];if(st.cur.length>=2){st.polys.push({pts:st.cur,label:'Myotube '+(st.polys.length+1)});st.cur=[];}buildCards();render();});
  document.querySelectorAll('[data-undo]').forEach(el=>el.onclick=e=>{const st=state[e.target.dataset.undo];if(st.cur.length)st.cur.pop();else if(st.polys.length)st.cur=st.polys.pop().pts;buildCards();render();});
  document.querySelectorAll('[data-clrdraw]').forEach(el=>el.onclick=e=>{const st=state[e.target.dataset.clrdraw];st.polys=[];st.cur=[];buildCards();render();});
  document.querySelectorAll('[data-mlabel]').forEach(el=>el.oninput=e=>{const[id,k]=e.target.dataset.mlabel.split(':');state[id].polys[+k].label=e.target.value;render();});
  document.querySelectorAll('[data-rmpoly]').forEach(el=>el.onclick=e=>{e.preventDefault();const[id,k]=e.target.dataset.rmpoly.split(':');state[id].polys.splice(+k,1);buildCards();render();});
  document.querySelectorAll('[data-clrupt]').forEach(el=>el.onclick=e=>{e.preventDefault();state[e.target.dataset.clrupt].user_points=[];buildCards();render();});
}
function redrawPayload(st){
  const polys=st.polys.map(p=>p.pts.slice()), labels=st.polys.map(p=>p.label||'');
  if(st.cur&&st.cur.length>=2){polys.push(st.cur.slice());labels.push('Myotube '+(st.polys.length+1));}
  return {polys,labels};
}
function decisions(){
  const out={stem:DATA.stem, decisions:{}};
  DATA.cases.forEach(c=>{
    const st=state[c.id];
    if(st.action==='redraw'){
      const pl=redrawPayload(st);
      if(pl.polys.length) out.decisions[c.id]={action:'redraw', polys:pl.polys, labels:pl.labels, note:st.note};
      // draw mode but nothing drawn -> apply the safe action but DON'T train on it (not a real decision)
      else out.decisions[c.id]={action:(c.type==='merge'?'separate':c.type==='occluded'?'drop':'keep'), note:(st.note||'')+' [draw mode, nothing drawn]', skip_learn:true};
    } else if(c.type==='split'){
      if(st.action==='keep') out.decisions[c.id]={action:'keep',note:st.note};
      else if(st.action==='reject') out.decisions[c.id]={action:'reject',note:st.note};
      else if(st.action==='split_n'&&st.n>1) out.decisions[c.id]={action:'split_n',n:st.n,note:st.note};
      else if(st.action==='split_n') out.decisions[c.id]={action:'keep',note:(st.note||'')+' [split_n N<2 ignored]'};
      else {const pts=(c.proposed_splits||[]).filter((p,i)=>st.points[i]).concat(st.user_points||[]);
            out.decisions[c.id]={action: pts.length?'split':'keep', points:pts, note:st.note};}
    } else out.decisions[c.id]={action:st.action, note:st.note};
  });
  return out;
}
function render(){ document.getElementById('out').value=JSON.stringify(decisions(),null,2); saveProgress(); }
function download(){const blob=new Blob([JSON.stringify(decisions(),null,2)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='decisions.json';a.click();}
document.getElementById('stem').textContent=DATA.stem;
document.getElementById('summary').textContent=DATA.n_combined+' candidate myotubes — '+DATA.cases.length+' flagged ('+
  DATA.cases.filter(c=>c.type==='split').length+' splits, '+DATA.cases.filter(c=>c.type==='merge').length+' merges, '+
  DATA.cases.filter(c=>c.type==='occluded').length+' occluded). Raw image shown; toggle my overlay per case.';
const _restoredWhen=loadProgress();
buildCards(); render();
if(_restoredWhen){ const el=document.getElementById('saved'); if(el)el.textContent='✓ restored your progress from '+_restoredWhen; }
window.addEventListener('resize',()=>DATA.cases.forEach(c=>{if(flt==='all'||c.type===flt)drawCanvas(c);}));
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    flags = json.load(open(os.path.join(a.out, "flags.json"), encoding="utf-8"))
    html = PAGE.replace("__DATA__", json.dumps(flags)).replace("__STEM__", flags.get("stem", ""))
    path = os.path.join(a.out, "review.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"review.html written: {len(flags['cases'])} cases -> {path}")


if __name__ == "__main__":
    main()
