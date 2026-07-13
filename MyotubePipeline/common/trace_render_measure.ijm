// trace_render_measure.ijm -- canonical render + measure for one trace set.
// Loads primary/overlap/dapi channels, builds polyline ROIs from a traces.txt (IN INPUT ORDER --
// the calling Python stage is responsible for spatial ordering), measures Length + mean on each
// channel, and writes: <out>_rois.zip, <out>_results.csv, <out>_overlay_clean.png,
// <out>_overlay_labeled.png, <out>_overlay_preview.png.  Reused by stages 2,3,4(final),5.
//
// arg: work=<dir with ch*_raw16.tif>; traces=<traces.txt>; out=<output path prefix>;
//      primary=<n>; overlap=<n>; dapi=<n>; max=<int>; label=<0|1>; rgb=<R-G-B (line color)>
setBatchMode(true);
function arg(k){ s=getArgument(); p=split(s,";"); for(i=0;i<p.length;i++){ if(p[i]==""){continue;} kv=split(p[i],"="); if(kv.length>=2 && kv[0]==k) return kv[1]; } return ""; }
work=arg("work"); tracesFile=arg("traces"); out=arg("out");
primary=arg("primary"); overlap=arg("overlap"); dapi=arg("dapi");
dmax=parseInt(arg("max")); doLabel=parseInt(arg("label"));
rgbArg=arg("rgb"); if(rgbArg==""){ rgbArg="255-255-0"; }
rc=split(rgbArg,"-"); cR=parseInt(rc[0]); cG=parseInt(rc[1]); cB=parseInt(rc[2]);
UM=2360.74/3636.0; strokeW=4; fontSz=34;

// ---- green base for overlay (primary @ display max) ----
open(work+"ch"+primary+"_raw16.tif"); run("Set Scale...","distance=0 known=0 pixel=1 unit=pixel");
run("Duplicate...","title=grn8"); setMinAndMax(0,dmax); run("8-bit");
run("Green"); run("RGB Color"); rename("green");

// ---- background-subtracted measurement images ----
selectWindow("ch"+primary+"_raw16.tif"); run("Subtract Background...","rolling=50 disable"); rename("bgP");
open(work+"ch"+overlap+"_raw16.tif"); run("Set Scale...","distance=0 known=0 pixel=1 unit=pixel");
run("Subtract Background...","rolling=50 disable"); rename("bgO");
open(work+"ch"+dapi+"_raw16.tif"); run("Set Scale...","distance=0 known=0 pixel=1 unit=pixel");
run("Subtract Background...","rolling=50 disable"); rename("bgD");

// ---- load traces -> polyline ROIs (preserve input order) ----
str=File.openAsString(tracesFile); lines=split(str,"\n"); roiManager("reset");
nLoaded=0;
for(i=0;i<lines.length;i++){
    v=split(lines[i],","); if(v.length<4) continue; np=floor(v.length/2);
    x=newArray(np); y=newArray(np);
    for(k=0;k<np;k++){ x[k]=parseFloat(v[2*k]); y[k]=parseFloat(v[2*k+1]); }
    makeSelection("polyline",x,y); roiManager("add"); nLoaded++;
}
n=roiManager("count");

// ---- measure ----
len=newArray(n); pm=newArray(n); om=newArray(n); dm=newArray(n); mx=newArray(n); my=newArray(n);
for(i=0;i<n;i++){
    selectWindow("bgP"); roiManager("select",i); len[i]=getValue("Length"); pm[i]=getValue("Mean");
    Roi.getCoordinates(xp,yp); m=floor(xp.length/2); mx[i]=xp[m]; my[i]=yp[m];
    selectWindow("bgO"); roiManager("select",i); om[i]=getValue("Mean");
    selectWindow("bgD"); roiManager("select",i); dm[i]=getValue("Mean");
}

// ---- results.csv ----
csvp=out+"_results.csv";
File.saveString("id,mid_x,mid_y,length_px,length_um,primary_mean,overlap_mean,dapi_mean\n", csvp);
for(i=0;i<n;i++){
    File.append((i+1)+","+mx[i]+","+my[i]+","+len[i]+","+(len[i]*UM)+","+pm[i]+","+om[i]+","+dm[i], csvp);
}

// ---- overlay (clean) ----
selectWindow("green"); run("Duplicate...","title=ov");
setColor(cR,cG,cB); setLineWidth(strokeW);
for(i=0;i<n;i++){ roiManager("select",i); Roi.getCoordinates(xp,yp); for(j=1;j<xp.length;j++) drawLine(xp[j-1],yp[j-1],xp[j],yp[j]); }
run("Select None"); saveAs("PNG", out+"_overlay_clean.png");

// ---- overlay (labeled) ----
if(doLabel==1){
    setFont("SansSerif",fontSz,"bold"); setJustification("left");
    for(i=0;i<n;i++){ lx=mx[i]+7; ly=my[i]-7; setColor(0,0,0); drawString(""+(i+1),lx+2,ly+2); setColor(255,255,255); drawString(""+(i+1),lx,ly); }
    run("Select None"); saveAs("PNG", out+"_overlay_labeled.png");
}
run("Duplicate...","title=prev"); run("Size...","width=1500 height=1500 average interpolation=Bilinear");
saveAs("PNG", out+"_overlay_preview.png"); close();
if(isOpen("ov")){ selectWindow("ov"); close(); }

// ---- ROI set (names = padded ids). Skip when empty (empty save blocks on a dialog in -batch). ----
if(n>0){
    for(i=0;i<n;i++){ roiManager("select",i); roiManager("rename", IJ.pad(i+1,3)); }
    roiManager("Deselect"); roiManager("save", out+"_rois.zip");
}

File.saveString("rendered="+n+" loaded="+nLoaded+"\n", out+"_render_log.txt");
run("Quit");
