#!/bin/bash
cd /c/Users/liqig/Documents/HRB_Transdiff/Conversion_Efficiency
P=../Q_PLATES/Q_Plates/PLATE_23
for w in 19_B06_act104_trka 22_B03_act104_egfrc 29_C05_br223_egfrc 32_C08_br223_igf1r 33_C09_br223_trka; do
  echo "=== $w ==="
  cpenv/Scripts/python.exe fusion_index.py --nd2 "$P/$w.nd2" --masks "plate23_nuclei/${w}_masks.npy" --outdir plate23_fusion 2>&1 | grep FUSION_DONE
done
echo "FUSION_ALLDONE"
