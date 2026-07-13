# Wet Lab Quantification Skill Spec

Fill this for one assay first. Narrow and validated is better than broad and fragile.

## 1. Skill Name
Preferred name:

## 2. Scope
Assay type:
What this first version should quantify:
What it should not handle yet:

## 3. Trigger Examples
Example requests that should activate the skill:
- 
- 
- 

## 4. Input Files
Image/file types:
Folder structure:
Naming convention:
Metadata files, if any:
Representative input folder:

## 5. Manual FIJI/ImageJ Protocol
Write the exact manual steps you currently perform.

1. 
2. 
3. 

Menus/tools used:
Measurements required:
ROI handling:
Lane/band/cell/object detection rule:
Thresholding method and settings:
Background subtraction/correction:
Channel splitting/merging:
Particle analysis settings:
Scale calibration:
Batch behavior:

## 6. Raw FIJI/ImageJ Outputs
Files FIJI currently exports:
Columns expected:
Units:
Known quirks:
What should be preserved exactly:

## 7. Python / Excel Processing
What Python should do after FIJI:
Normalization rules:
Grouping rules:
Summary statistics:
Excel sheets needed:
Excel formulas needed:
Charts needed:
Example final workbook:

## 8. QC And Stop Rules
Failed image criteria:
Failed measurement criteria:
Saturation/overexposure rule:
Missing scale/metadata rule:
Acceptable ranges:
Warnings to flag:
Cases that require manual review:

## 9. Final Outputs
Desired output folder structure:
Final file names:
Required report format:
Processing log contents:

## 10. Validation Dataset
Sample images available:
Gold-standard finished Excel file:
Existing FIJI macro:
Existing Python script:
Expected result for one known image or lane:

## 11. Non-Negotiables
The automation must never:
- Overwrite raw images
- Silently change quantification parameters

The automation must always:
- Save raw FIJI/ImageJ outputs before cleanup
- Record assumptions, exclusions, warnings, and software versions

