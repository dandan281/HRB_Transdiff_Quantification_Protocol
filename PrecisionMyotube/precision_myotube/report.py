"""QC overlays and a self-contained human review package."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from .io import load_run_channel

STATUS_COLORS = {
    "complete": (0, 255, 255),
    "border_truncated": (255, 180, 0),
    "occluded": (190, 80, 255),
    "ambiguous": (255, 50, 50),
}


def _stretch(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, (1, 99.7))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def create_overlay(run_dir: str | Path, analysis: dict) -> Path:
    run = Path(run_dir)
    fiber = _stretch(load_run_channel(run, "fiber"))
    rgb = np.stack([fiber // 3, fiber, fiber // 3], axis=-1)
    territory = analysis["territory"]
    rgb[territory, 1] = np.maximum(rgb[territory, 1], 100)

    row_by_id = {row["id"]: row for row in analysis["myotubes"]}
    for instance_id, (bbox, mask) in analysis["instance_masks"].items():
        r0, c0, r1, c1 = bbox
        edge = ndi.binary_dilation(mask) ^ mask
        color = STATUS_COLORS[row_by_id[instance_id]["status"]]
        crop = rgb[r0:r1, c0:c1]
        crop[edge] = color

    nuclei = analysis["nuclei_labels"]
    valid = analysis["valid_nuclei"]
    valid_pixels = valid[nuclei]
    inside = np.zeros_like(valid_pixels)
    assigned = np.zeros_like(valid_pixels)
    ambiguous = np.zeros_like(valid_pixels)
    for row in analysis["nuclei"]:
        nucleus_id = int(row["id"])
        if row["in_myotube_50"]:
            inside[nuclei == nucleus_id] = True
        if row["assignment_status"] == "assigned":
            assigned[nuclei == nucleus_id] = True
        elif row["assignment_status"] == "assignment_ambiguous":
            ambiguous[nuclei == nucleus_id] = True
    nucleus_edge = ndi.binary_dilation(valid_pixels) ^ valid_pixels
    rgb[nucleus_edge & ~inside] = (40, 100, 255)
    rgb[nucleus_edge & inside & ~assigned] = (255, 80, 220)
    rgb[nucleus_edge & assigned] = (255, 255, 255)
    rgb[nucleus_edge & ambiguous] = (255, 0, 0)

    path = run / "qc_overlay.png"
    image = Image.fromarray(rgb)
    image.save(path)
    preview = image.copy()
    preview.thumbnail((1800, 1800))
    preview.save(run / "qc_overlay_preview.png")
    return path


def create_review_html(run_dir: str | Path, analysis: dict) -> Path:
    run = Path(run_dir)
    summary = analysis["summary"]
    flags = analysis["qc_flags"]
    flag_rows = []
    for index, flag in enumerate(flags):
        subject = flag.get("instance_id", f"nucleus {flag.get('nucleus_id', '')}")
        detail = html.escape(json.dumps({k: v for k, v in flag.items()
                                         if k not in {"type", "severity", "instance_id"}}))
        flag_rows.append(
            f"<tr><td>{index + 1}</td><td>{html.escape(flag['severity'])}</td>"
            f"<td>{html.escape(flag['type'])}</td><td>{html.escape(str(subject))}</td>"
            f"<td>{detail}</td><td><select data-flag='{index}'>"
            "<option value='pending'>Pending</option><option value='accept'>Accept</option>"
            "<option value='corrected'>Corrected</option><option value='exclude'>Exclude</option>"
            "</select></td></tr>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Precision Myotube Review</title>
<style>
body{{font:14px system-ui;background:#111;color:#eee;margin:20px}} img{{max-width:100%;border:1px solid #555}}
table{{border-collapse:collapse;width:100%;margin-top:16px}} th,td{{border:1px solid #555;padding:5px;text-align:left}}
th{{background:#263238}} .legend span{{margin-right:16px}} button{{padding:8px 14px;margin:12px 0}}
code{{color:#9ee}}
</style></head><body>
<h1>{html.escape(summary['image_id'])}: precision review</h1>
<p><b>This page never changes masks automatically.</b> Correct masks in napari, update the instance
manifest, rerun analysis, then record the disposition of every required flag.</p>
<div class="legend"><span style="color:#0ff">complete</span><span style="color:#fb0">truncated</span>
<span style="color:#c5f">occluded</span><span style="color:#f44">ambiguous/assignment</span>
<span style="color:#fff">assigned nucleus</span></div>
<p>Total nuclei: {summary['total_nuclei']} &nbsp; Conversion efficiency (50%):
{100 * summary['conversion_efficiency_50']:.2f}% &nbsp; Authoritative instances:
{summary['instances_authoritative']} &nbsp; Required flags: {summary['required_qc_flags']}</p>
<img src="qc_overlay_preview.png" alt="QC overlay">
<h2>Review queue</h2><table><thead><tr><th>#</th><th>Severity</th><th>Type</th><th>Subject</th>
<th>Details</th><th>Decision</th></tr></thead><tbody>{''.join(flag_rows)}</tbody></table>
<p>Reviewer: <input id="reviewer"> <button onclick="save()">Download decisions.json</button></p>
<script>
function save(){{
 const decisions=[...document.querySelectorAll('select[data-flag]')].map(x=>({{flag_index:+x.dataset.flag,decision:x.value}}));
 const payload={{schema_version:'1.0',image_id:{json.dumps(summary['image_id'])},reviewer:document.getElementById('reviewer').value,decisions}};
 const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});
 const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='review_decisions.json'; a.click();
}}
</script></body></html>"""
    path = run / "review.html"
    path.write_text(document, encoding="utf-8")
    return path


def create_reports(run_dir: str | Path, analysis: dict) -> tuple[Path, Path]:
    overlay = create_overlay(run_dir, analysis)
    review = create_review_html(run_dir, analysis)
    return overlay, review
