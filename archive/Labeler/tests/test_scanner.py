from pathlib import Path

from wb_annotator.scanner import preserved_extension, scan_image_files


def test_scan_supported_wb_image_extensions(tmp_path: Path) -> None:
    names = [
        "a.tif",
        "b.tiff",
        "c.raw16tif",
        "d.raw16.tif",
        "e.jpg",
        "f.jpeg",
        "notes.txt",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"data")

    scanned = [path.name for path in scan_image_files(tmp_path)]

    assert scanned == ["a.tif", "b.tiff", "c.raw16tif", "d.raw16.tif", "e.jpg", "f.jpeg"]


def test_preserved_extension_keeps_raw16_compound_suffix() -> None:
    assert preserved_extension("sample.raw16.tif") == ".raw16.tif"
    assert preserved_extension("sample.raw16tif") == ".raw16tif"
    assert preserved_extension("sample.TIF") == ".TIF"
