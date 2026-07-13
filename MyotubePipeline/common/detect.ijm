// common/detect.ijm -- Ridge Detection on the primary channel; export raw centreline segments.
// Reads ch{primary}_raw16.tif from the stage1 dir (read-only); writes segments to the caller's dir.
//
// arg: work=<stage1 dir>; primary=<n>; out=<segments.txt>;
//      scalemode=<reset|fixed>; max=<int, used when scalemode=fixed>;
//      low_contrast=<int>; high_contrast=<int>; lower=<float>; upper=<float>;
//      line_width=<int>; sigma=<float>; min_len=<float export floor px>
//
// Defaults match the validated gen_detect.ijm (reset scaling, lw=16, sigma=5.12, 0.2/0.6).
// 'reset' scaling = resetMinAndMax (the reliable choice; never stalls). 'fixed' brightens dim
// wells (max < p975) to recover faint fibers -- used by Stage 3's dim-boost pass.
setBatchMode(true);
function arg(k){ s=getArgument(); p=split(s,";"); for(i=0;i<p.length;i++){ if(p[i]==""){continue;} kv=split(p[i],"="); if(kv.length>=2 && kv[0]==k) return kv[1]; } return ""; }
function argd(k,dv){ v=arg(k); if(v=="") return dv; return v; }
work=arg("work"); primary=arg("primary"); outf=arg("out");
scalemode=argd("scalemode","reset"); dmax=parseInt(argd("max","0"));
lowC=parseInt(argd("low_contrast","40")); highC=parseInt(argd("high_contrast","200"));
lower=parseFloat(argd("lower","0.2")); upper=parseFloat(argd("upper","0.6"));
lineW=parseInt(argd("line_width","16")); sigma=parseFloat(argd("sigma","5.12"));
minLen=parseFloat(argd("min_len","25"));

logf=File.getParent(outf)+File.separator+"detect_log.txt";
File.saveString("DETECT primary="+primary+" scalemode="+scalemode+" max="+dmax+
    " lowC="+lowC+" highC="+highC+" lower="+lower+" upper="+upper+"\n", logf);

open(work+"ch"+primary+"_raw16.tif");
run("Set Scale...", "distance=0 known=0 pixel=1 unit=pixel");
title=getTitle();

// detection image: bg-subtract -> 8-bit (reset or fixed scaling) -> CLAHE
run("Subtract Background...", "rolling=50 disable");
run("Duplicate...", "title=det");
if(scalemode=="fixed" && dmax>0){ setMinAndMax(0,dmax); } else { resetMinAndMax(); }
run("8-bit");
run("Enhance Local Contrast (CLAHE)", "blocksize=200 histogram=256 maximum=3 mask=*None*");

run("Ridge Detection",
    "line_width="+lineW+" high_contrast="+highC+" low_contrast="+lowC+" estimate_width extend_line displayresults" +
    " add_to_manager method_for_overlap_resolution=SLOPE sigma="+sigma+" lower_threshold="+lower +
    " upper_threshold="+upper+" minimum_line_length=0 maximum=0");
nAll=roiManager("count");
File.append("raw segments="+nAll+"\n", logf);

File.saveString("", outf);
selectWindow("det");
written=0;
for(i=0;i<nAll;i++){
    roiManager("select", i);
    if(getValue("Length") < minLen) continue;
    Roi.getCoordinates(xp,yp); np=xp.length;
    if(np<2) continue;
    // Roi.getCoordinates returns PARALLEL per-vertex arrays (xp[k],yp[k] = vertex k); emit every
    // vertex once. (The old gen_detect stepped j by 2, silently dropping half the centreline.)
    line=""+d2s(xp[0],2)+","+d2s(yp[0],2);
    for(j=1;j<np;j++) line=line+","+d2s(xp[j],2)+","+d2s(yp[j],2);
    File.append(line, outf);
    written++;
}
File.append("exported="+written+"\n", logf);
run("Quit");
