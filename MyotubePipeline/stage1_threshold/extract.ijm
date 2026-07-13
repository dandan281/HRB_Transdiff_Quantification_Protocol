// stage1/extract.ijm -- extract 3 channels of an ND2 + previews into the stage1 dir.
// arg: src=<nd2>; work=<stage1 dir>
setBatchMode(true);
function arg(k){ s=getArgument(); p=split(s,";"); for(i=0;i<p.length;i++){ if(p[i]==""){continue;} kv=split(p[i],"="); if(kv.length>=2 && kv[0]==k) return kv[1]; } return ""; }
src = arg("src"); work = arg("work");
File.makeDirectory(work);
logf = work + "extract_log.txt"; File.saveString("EXTRACT src=" + src + "\n", logf);
function log(m){ File.append(m+"\n", logf); }

run("Bio-Formats Importer", "open=[" + src + "] color_mode=Default view=Hyperstack stack_order=XYCZT");
title = getTitle(); getDimensions(w,h,c,z,t);
log("w=" + w + " h=" + h + " c=" + c + " z=" + z + " t=" + t);
for (ch=1; ch<=c; ch++){
    selectWindow(title);
    run("Duplicate...", "title=chdup duplicate channels=" + ch + "-" + ch);
    getStatistics(area, mean, min, max, std);
    log("ch" + (ch-1) + " mean=" + mean + " std=" + std + " max=" + max);
    saveAs("Tiff", work + "ch" + (ch-1) + "_raw16.tif");
    cur = getTitle();
    run("Duplicate...", "title=prev"); resetMinAndMax(); run("Enhance Contrast", "saturated=0.35"); run("8-bit");
    run("Size...", "width=900 height=900 average interpolation=Bilinear");
    saveAs("PNG", work + "ch" + (ch-1) + "_preview.png");
    close(); selectWindow(cur); close();
}
log("DONE");
run("Quit");
