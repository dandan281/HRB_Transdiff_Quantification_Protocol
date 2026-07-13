from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".raw16tif", ".raw16.tif", ".jpg", ".jpeg")


def is_supported_image(path: Path | str) -> bool:
    name = Path(path).name.lower()
    return any(name.endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def preserved_extension(filename: str) -> str:
    lower_name = filename.lower()
    for extension in sorted(SUPPORTED_EXTENSIONS, key=len, reverse=True):
        if lower_name.endswith(extension):
            return filename[-len(extension) :]
    return Path(filename).suffix


def scan_image_files(folder: Path | str) -> list[Path]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {root}")

    files = [path for path in root.iterdir() if path.is_file() and is_supported_image(path)]
    return sorted(files, key=lambda path: path.name.casefold())
