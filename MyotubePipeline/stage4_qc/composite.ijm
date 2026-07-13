// stage4/composite.ijm -- 3-channel composite using the SAME primary display max from Stage 1.
// overlap -> red, primary(fiber) -> green @ metadata max, dapi -> blue. Co-localised fiber+marker
// reads yellow; nuclei read blue/cyan. Crops for the review page are cut from this image.
// arg: stage1=<stage1 dir>; out=<stage4 dir>; stem=<stem>; primary=<n>; overlap=<n>; dapi=<n>; max=<int>
setBatchMode(true);
function arg(k){ s=getArgument(); p=split(s,";"); for(i=0;i<p.length;i++){ if(p[i]==""){continue;} kv=split(p[i],"="); if(kv.length>=2 && kv[0]==k) return kv[1]; } return ""; }
stage1=arg("stage1"); out=arg("out"); stem=arg("stem");
primary=arg("primary"); overlap=arg("overlap"); dapi=arg("dapi"); dmax=parseInt(arg("max"));

open(stage1+"ch"+overlap+"_raw16.tif"); run("Subtract Background...","rolling=50 disable");
resetMinAndMax(); run("Enhance Contrast","saturated=0.3"); run("8-bit"); rename("red8");
open(stage1+"ch"+primary+"_raw16.tif"); run("Subtract Background...","rolling=50 disable");
setMinAndMax(0,dmax); run("8-bit"); rename("grn8");
open(stage1+"ch"+dapi+"_raw16.tif"); run("Subtract Background...","rolling=50 disable");
resetMinAndMax(); run("Enhance Contrast","saturated=0.3"); run("8-bit"); rename("dap8");

run("Merge Channels...", "c1=red8 c2=grn8 c3=dap8 create");
run("RGB Color");
saveAs("PNG", out+stem+"_composite.png");
run("Duplicate...","title=prev"); run("Size...","width=1500 height=1500 average interpolation=Bilinear");
saveAs("PNG", out+stem+"_composite_preview.png");
run("Quit");
