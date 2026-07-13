// stage1/adjust_primary.ijm -- produce the SHARED display-scaled primary + signal map at the
// brightness/contrast chosen by threshold.py. These are the single source of truth for every
// downstream stage's display scaling.
// arg: work=<stage1 dir>; primary=<n>; min=<int>; max=<int>
setBatchMode(true);
function arg(k){ s=getArgument(); p=split(s,";"); for(i=0;i<p.length;i++){ if(p[i]==""){continue;} kv=split(p[i],"="); if(kv.length>=2 && kv[0]==k) return kv[1]; } return ""; }
work=arg("work"); primary=arg("primary"); dmin=parseInt(arg("min")); dmax=parseInt(arg("max"));

// adjusted 8-bit primary (raw scaling, NO bg subtraction) -- the "duplicate" downstream renders share
open(work+"ch"+primary+"_raw16.tif"); run("Set Scale...","distance=0 known=0 pixel=1 unit=pixel");
run("Duplicate...","title=adj"); setMinAndMax(dmin,dmax); run("8-bit");
saveAs("Tiff", work+"ch"+primary+"_adjusted8.tif");

// signal map: background-subtracted primary scaled 0..max -> 8-bit (used by merge + flag)
selectWindow("ch"+primary+"_raw16.tif");
run("Subtract Background...","rolling=50 disable");
setMinAndMax(0,dmax); run("8-bit");
saveAs("PNG", work+"signal.png");

File.saveString("adjusted primary=ch"+primary+" min="+dmin+" max="+dmax+"\n", work+"adjust_log.txt");
run("Quit");
