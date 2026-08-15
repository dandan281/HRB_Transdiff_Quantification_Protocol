"""The relabelling page. Vanilla JS, no build step, no CDN.

Interaction is built around the fact that a myotube is a ribbon and the operator
knows where it goes: click down the spine, commit, move on. Width is one control
for the whole trace and `snap` recovers the rest from the image, so the operator
is never asked to paint an edge.

Every commit POSTs immediately. There is no "save" button on purpose -- a save
button is a thing you forget to press.
"""
from __future__ import annotations

HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Myotube relabelling — __WELL__</title>
<style>
  :root{--bg:#0f1115;--panel:#171a21;--line:#272c36;--ink:#e8e8ea;--mut:#9aa0ab;
        --acc:#2a78d6;--new:#1baf7a;--rej:#d03b3b;--warn:#eda100;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:13px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
       display:flex;height:100vh;overflow:hidden}
  #stage{flex:1;position:relative;overflow:hidden;background:#000;cursor:crosshair}
  canvas{position:absolute;top:0;left:0}
  aside{width:310px;background:var(--panel);border-left:1px solid var(--line);
        padding:14px;overflow-y:auto;flex-shrink:0}
  h1{font-size:15px;margin:0 0 2px}
  .sub{color:var(--mut);font-size:11px;margin-bottom:12px;word-break:break-all}
  .box{background:#12151b;border:1px solid var(--line);border-radius:7px;
       padding:9px 10px;margin-bottom:10px}
  .row{display:flex;justify-content:space-between;gap:8px;padding:2px 0}
  .row b{font-variant-numeric:tabular-nums}
  label{display:block;color:var(--mut);font-size:11px;margin:8px 0 3px}
  input[type=range]{width:100%}
  input[type=text],select{width:100%;background:#0d1014;color:var(--ink);
       border:1px solid var(--line);border-radius:5px;padding:6px}
  button{width:100%;padding:8px;margin-top:6px;border-radius:6px;cursor:pointer;
         border:1px solid var(--line);background:#1d222b;color:var(--ink);font:inherit}
  button:hover{background:#242a35}
  button.pri{background:var(--acc);border-color:var(--acc);font-weight:600}
  button.dan{background:var(--rej);border-color:var(--rej)}
  kbd{background:#0d1014;border:1px solid var(--line);border-radius:4px;
      padding:1px 5px;font:11px ui-monospace,monospace}
  .help{color:var(--mut);font-size:11.5px}
  .help div{padding:2px 0;display:flex;justify-content:space-between;gap:10px}
  #toast{position:absolute;left:50%;bottom:22px;transform:translateX(-50%);
    background:#1d222b;border:1px solid var(--line);border-radius:7px;
    padding:8px 14px;opacity:0;transition:opacity .25s;pointer-events:none}
  #toast.on{opacity:1}
  .sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}
</style></head><body>
<div id="stage">
  <canvas id="cv"></canvas>
  <div id="toast"></div>
</div>
<aside>
  <h1>Myotube relabelling</h1>
  <div class="sub" id="wellName">__WELL__</div>

  <div class="box">
    <div class="row"><span><i class="sw" style="background:#7a7f8a"></i>existing certified</span><b id="nExist">0</b></div>
    <div class="row"><span><i class="sw" style="background:var(--new)"></i>you added</span><b id="nNew">0</b></div>
    <div class="row"><span><i class="sw" style="background:var(--rej)"></i>you rejected</span><b id="nRej">0</b></div>
  </div>

  <label>Reviewer</label>
  <input type="text" id="reviewer" placeholder="your name">

  <label>Width <b id="wLab">8</b> px (<span id="wUm">5.2</span> µm) — <kbd>[</kbd> <kbd>]</kbd></label>
  <input type="range" id="width" min="3" max="40" step="1" value="8">

  <label>Mask fitting</label>
  <select id="mode">
    <option value="snap">snap to signal (recommended)</option>
    <option value="ribbon">constant-width ribbon</option>
  </select>

  <button class="pri" id="commit">Commit trace &nbsp;<kbd>Enter</kbd></button>
  <button id="cancel">Cancel trace &nbsp;<kbd>Esc</kbd></button>
  <button class="dan" id="reject">Reject selected existing &nbsp;<kbd>X</kbd></button>
  <button id="undo">Undo my last &nbsp;<kbd>Ctrl+Z</kbd></button>

  <div class="box" style="margin-top:12px">
    <div class="help">
      <div><span>add point</span><kbd>click</kbd></div>
      <div><span>commit</span><kbd>Enter</kbd> / <kbd>dbl-click</kbd></div>
      <div><span>cancel</span><kbd>Esc</kbd></div>
      <div><span>width</span><kbd>[</kbd> <kbd>]</kbd></div>
      <div><span>pan</span><kbd>drag w/ space</kbd></div>
      <div><span>zoom</span><kbd>scroll</kbd></div>
      <div><span>fit</span><kbd>F</kbd></div>
      <div><span>hide existing</span><kbd>H</kbd></div>
      <div><span>select existing</span><kbd>click on it</kbd></div>
    </div>
  </div>

  <label>Well</label>
  <select id="wellSel">__WELLOPTS__</select>
  <button id="finish">Finish batch &rarr; summary</button>
  <div class="sub" style="margin-top:10px">Every commit saves to disk instantly.
  There is no save button.</div>
</aside>
<script>
const WELL="__WELL__", PIXEL_UM=__PIXEL_UM__;
const cv=document.getElementById("cv"), ctx=cv.getContext("2d");
const stage=document.getElementById("stage");
let field=new Image(), labels=new Image();
let view={s:1,tx:0,ty:0}, pts=[], traces=[], existing=[], rejected={},
    width=8, selected=null, showExist=true, panning=false, spaceDown=false,
    last={x:0,y:0}, imgW=0, imgH=0;

function toast(m){const t=document.getElementById("toast");t.textContent=m;
  t.classList.add("on");clearTimeout(t._t);t._t=setTimeout(()=>t.classList.remove("on"),1600);}
function resize(){cv.width=stage.clientWidth;cv.height=stage.clientHeight;draw();}
function fit(){if(!imgW)return;const s=Math.min(cv.width/imgW,cv.height/imgH)*0.96;
  view={s:s,tx:(cv.width-imgW*s)/2,ty:(cv.height-imgH*s)/2};draw();}
function toImg(x,y){return [(y-view.ty)/view.s,(x-view.tx)/view.s];}   // -> row,col
function toScr(r,c){return [c*view.s+view.tx, r*view.s+view.ty];}

function draw(){
  ctx.setTransform(1,0,0,1,0,0);
  ctx.fillStyle="#000";ctx.fillRect(0,0,cv.width,cv.height);
  if(!imgW)return;
  ctx.imageSmoothingEnabled = view.s < 1;
  ctx.setTransform(view.s,0,0,view.s,view.tx,view.ty);
  ctx.drawImage(field,0,0);
  if(showExist && labels.complete) {ctx.globalAlpha=0.55;ctx.drawImage(labels,0,0);ctx.globalAlpha=1;}
  ctx.setTransform(1,0,0,1,0,0);

  // committed traces
  for(const t of traces) strokeTrace(t.points,t.width_px,"#1baf7a",0.9);
  // rejected existing markers
  for(const id in rejected){const e=existing.find(v=>v.label==id);if(!e)continue;
    const [x,y]=toScr(e.cy,e.cx);ctx.strokeStyle="#d03b3b";ctx.lineWidth=2;
    ctx.beginPath();ctx.arc(x,y,11,0,7);ctx.stroke();
    ctx.beginPath();ctx.moveTo(x-7,y-7);ctx.lineTo(x+7,y+7);
    ctx.moveTo(x+7,y-7);ctx.lineTo(x-7,y+7);ctx.stroke();}
  // selection
  if(selected!=null){const e=existing.find(v=>v.label==selected);
    if(e){const [x0,y0]=toScr(e.r0,e.c0),[x1,y1]=toScr(e.r1,e.c1);
      ctx.strokeStyle="#eda100";ctx.lineWidth=2;ctx.setLineDash([5,4]);
      ctx.strokeRect(x0,y0,x1-x0,y1-y0);ctx.setLineDash([]);}}
  // in-progress
  if(pts.length){strokeTrace(pts,width,"#2a78d6",1);
    for(const p of pts){const [x,y]=toScr(p[0],p[1]);
      ctx.fillStyle="#fff";ctx.beginPath();ctx.arc(x,y,3,0,7);ctx.fill();}}
}
function strokeTrace(points,w,color,alpha){
  if(points.length<1)return;
  ctx.globalAlpha=alpha;ctx.strokeStyle=color;
  ctx.lineWidth=Math.max(w*view.s,1.5);ctx.lineCap="round";ctx.lineJoin="round";
  ctx.beginPath();
  points.forEach((p,i)=>{const [x,y]=toScr(p[0],p[1]);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
  ctx.stroke();ctx.globalAlpha=1;
}

async function loadWell(w){
  const st=await (await fetch(`/api/state/${w}`)).json();
  existing=st.existing;traces=Object.values(st.traces);rejected=st.rejected||{};
  imgW=st.width;imgH=st.height;
  field=new Image();labels=new Image();
  await new Promise(r=>{field.onload=r;field.src=`/img/${w}/field.png?t=${Date.now()}`;});
  labels.src=`/img/${w}/labels.png?t=${Date.now()}`;
  labels.onload=draw;
  counts();fit();
}
function counts(){
  document.getElementById("nExist").textContent=existing.length-Object.keys(rejected).length;
  document.getElementById("nNew").textContent=traces.length;
  document.getElementById("nRej").textContent=Object.keys(rejected).length;
}
async function post(rec){
  const r=await fetch(`/api/trace/${WELL}`,{method:"POST",
    headers:{"Content-Type":"application/json"},body:JSON.stringify(rec)});
  if(!r.ok){toast("SAVE FAILED — "+r.status);return null;}
  return await r.json();
}
async function commit(){
  if(pts.length<2){toast("need at least 2 points");return;}
  const rec={kind:"add",points:pts,width_px:width,
    mode:document.getElementById("mode").value,
    reviewer:document.getElementById("reviewer").value||null};
  const saved=await post(rec);
  if(saved){traces.push(saved);pts=[];counts();draw();toast("saved · "+traces.length+" added");}
}
async function rejectSel(){
  if(selected==null){toast("click an existing instance first");return;}
  const saved=await post({kind:"reject_existing",source_label:selected,
    reviewer:document.getElementById("reviewer").value||null});
  if(saved){rejected[selected]=saved;selected=null;counts();draw();toast("rejected");}
}
async function undo(){
  if(!traces.length){toast("nothing to undo");return;}
  const t=traces[traces.length-1];
  const saved=await post({kind:"delete",replaces:t.trace_id,
    reviewer:document.getElementById("reviewer").value||null});
  if(saved){traces.pop();counts();draw();toast("undone");}
}

stage.addEventListener("mousedown",e=>{
  if(spaceDown||e.button===1){panning=true;last={x:e.clientX,y:e.clientY};return;}
  if(e.button!==0)return;
  const [r,c]=toImg(e.offsetX,e.offsetY);
  if(pts.length===0){ // maybe selecting an existing instance
    const hit=existing.find(v=>r>=v.r0&&r<=v.r1&&c>=v.c0&&c<=v.c1);
    if(hit&&e.shiftKey===false&&e.detail===1&&e.altKey){selected=hit.label;draw();return;}
  }
  pts.push([r,c]);draw();
});
stage.addEventListener("dblclick",e=>{e.preventDefault();commit();});
window.addEventListener("mousemove",e=>{
  if(!panning)return;view.tx+=e.clientX-last.x;view.ty+=e.clientY-last.y;
  last={x:e.clientX,y:e.clientY};draw();});
window.addEventListener("mouseup",()=>panning=false);
stage.addEventListener("wheel",e=>{e.preventDefault();
  const f=e.deltaY<0?1.15:1/1.15;
  const [r,c]=toImg(e.offsetX,e.offsetY);
  view.s*=f;
  const [nx,ny]=toScr(r,c);
  view.tx+=e.offsetX-nx;view.ty+=e.offsetY-ny;draw();},{passive:false});

window.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT"||e.target.tagName==="SELECT")return;
  if(e.code==="Space"){spaceDown=true;stage.style.cursor="grab";e.preventDefault();}
  else if(e.key==="Enter")commit();
  else if(e.key==="Escape"){pts=[];selected=null;draw();}
  else if(e.key==="["){setW(width-1);}
  else if(e.key==="]"){setW(width+1);}
  else if(e.key==="f"||e.key==="F")fit();
  else if(e.key==="h"||e.key==="H"){showExist=!showExist;draw();}
  else if(e.key==="x"||e.key==="X")rejectSel();
  else if(e.key==="z"&&(e.ctrlKey||e.metaKey))undo();
});
window.addEventListener("keyup",e=>{if(e.code==="Space"){spaceDown=false;stage.style.cursor="crosshair";}});
function setW(v){width=Math.max(3,Math.min(40,v));
  document.getElementById("width").value=width;
  document.getElementById("wLab").textContent=width;
  document.getElementById("wUm").textContent=(width*PIXEL_UM).toFixed(1);draw();}
document.getElementById("width").addEventListener("input",e=>setW(+e.target.value));
document.getElementById("commit").onclick=commit;
document.getElementById("cancel").onclick=()=>{pts=[];draw();};
document.getElementById("reject").onclick=rejectSel;
document.getElementById("undo").onclick=undo;
document.getElementById("finish").onclick=()=>location.href="/summary";
document.getElementById("wellSel").onchange=e=>location.href="/?well="+e.target.value;
window.addEventListener("resize",resize);
resize();setW(8);loadWell(WELL);
</script></body></html>
"""


def render(well: str, wells: list[str], pixel_um: float) -> str:
    opts = "".join(
        f'<option value="{w}"{" selected" if w == well else ""}>{w}</option>'
        for w in wells)
    return (HTML.replace("__WELL__", well)
                .replace("__WELLOPTS__", opts)
                .replace("__PIXEL_UM__", f"{pixel_um:.6f}"))


SUMMARY = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Relabelling summary</title><style>
 body{background:#0f1115;color:#e8e8ea;font:14px/1.6 system-ui,sans-serif;
      max-width:820px;margin:40px auto;padding:0 20px}
 h1{font-size:20px} table{width:100%;border-collapse:collapse;margin:18px 0}
 th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #272c36}
 th{color:#9aa0ab;font-weight:600;font-size:12px}
 td b{font-variant-numeric:tabular-nums}
 code{background:#171a21;padding:2px 6px;border-radius:4px;font-size:12.5px}
 .n{color:#1baf7a} .r{color:#d03b3b}
 a{color:#2a78d6}
</style></head><body>
<h1>Relabelling batch — summary</h1>
<table><tr><th>well</th><th>existing kept</th><th class="n">added</th>
<th class="r">rejected</th><th>log records</th></tr>__ROWS__</table>
<p>Traces are already on disk. To fold them into a new training corpus:</p>
<pre><code>__APPLYCMD__</code></pre>
<p><a href="/">&larr; back to annotating</a></p>
</body></html>
"""
