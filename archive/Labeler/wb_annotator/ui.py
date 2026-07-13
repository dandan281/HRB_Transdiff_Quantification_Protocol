from __future__ import annotations

from datetime import date
from pathlib import Path
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from wb_annotator.auto_annotate import auto_annotate_files, experiment_group_key
from wb_annotator.batch_rules import CellLineBlockRule, ExperimentRule, apply_batch_rules, parse_mixed_batch_commands
from wb_annotator.dose_mapper import build_lane_annotations_from_experiments, parse_dose_series
from wb_annotator.manifest import write_label_export
from wb_annotator.renamer import apply_rename_plan, build_rename_plan
from wb_annotator.scanner import SUPPORTED_EXTENSIONS, scan_image_files
from wb_annotator.schema import (
    FILE_KIND_DEFINITIONS,
    LANE_DIRECTION_DEFINITIONS,
    LANE_ROLE_DEFINITIONS,
    PROTEIN_ROLE_DEFINITIONS,
    CellLineBlock,
    ExperimentMetadata,
    FileAnnotation,
    LaneAnnotation,
    RenameRecord,
)


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget, height: int = 180) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, height=height, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas)

        self.content.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.bind("<Configure>", self._resize_content)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def _resize_content(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)


def _parse_optional_int(value: str, field_name: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer: {value}") from exc


def _parse_required_int(value: str, field_name: str) -> int:
    if not value:
        raise ValueError(f"{field_name} is required")
    return _parse_optional_int(value, field_name) or 0


def _experiment_row_var_name(field: str) -> str:
    aliases = {
        "date": "date",
        "experiment_id": "experiment_id",
        "cell_line": "cell_line",
        "modification": "modification",
        "treatment_name": "treatment_name",
        "dose_series": "dose_series",
        "treatment_time": "treatment_time",
        "lane_direction": "lane_direction",
        "loading_control": "loading_control",
    }
    return aliases.get(field, field)


def _parse_lane_remove_selector(selector: str) -> tuple[str | None, int]:
    cleaned = selector.strip()
    if not cleaned:
        raise ValueError("Enter a lane number to remove, for example 3 or E02:3.")

    match = re.fullmatch(r"(?:(?P<experiment>E\d+)\s*[:\-\s])?\s*(?:lane\s*)?(?P<lane>\d+)", cleaned, re.IGNORECASE)
    if not match:
        raise ValueError("Use a lane number such as 3, or experiment plus lane such as E02:3.")

    experiment_key = match.group("experiment")
    lane_number = int(match.group("lane"))
    return (experiment_key.upper() if experiment_key else None), lane_number


def _image_label_summary(annotation: FileAnnotation) -> str:
    protein = annotation.protein_label.strip()
    role = annotation.protein_role.strip().upper() or "UNKNOWN"
    if protein:
        protein_part = f"{role}-{protein}" if role != "UNKNOWN" else protein
    elif role != "UNKNOWN":
        protein_part = role
    else:
        protein_part = "protein-unlabeled"
    return " / ".join(
        [
            annotation.experiment_key.strip() or "E01",
            annotation.blot_id.strip() or "B01",
            annotation.file_kind.strip().upper() or "RAW",
            protein_part,
        ]
    )


class FileRow:
    def __init__(
        self,
        parent: tk.Widget,
        row: int,
        original_name: str,
        experiment_key: str = "E01",
        blot_id: str = "B01",
        file_kind: str = "RAW",
        protein_label: str = "",
        protein_role: str = "UNKNOWN",
        note: str = "",
    ) -> None:
        self.original_name = original_name
        self.experiment_key = tk.StringVar(value=experiment_key)
        self.blot_id = tk.StringVar(value=blot_id)
        self.file_kind = tk.StringVar(value=file_kind)
        self.protein_label = tk.StringVar(value=protein_label)
        self.protein_role = tk.StringVar(value=protein_role)
        self.note = tk.StringVar(value=note)
        self.label_summary = tk.StringVar(value="")

        ttk.Label(parent, text=original_name, anchor="w").grid(row=row, column=0, sticky="ew", padx=4, pady=2)
        self.experiment_combo = ttk.Combobox(
            parent,
            textvariable=self.experiment_key,
            values=["E01"],
            width=8,
        )
        self.experiment_combo.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.blot_id, width=10).grid(row=row, column=2, sticky="ew", padx=4, pady=2)
        ttk.Combobox(
            parent,
            textvariable=self.file_kind,
            values=list(FILE_KIND_DEFINITIONS),
            width=10,
            state="readonly",
        ).grid(row=row, column=3, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.protein_label, width=14).grid(
            row=row, column=4, sticky="ew", padx=4, pady=2
        )
        ttk.Combobox(
            parent,
            textvariable=self.protein_role,
            values=list(PROTEIN_ROLE_DEFINITIONS),
            width=10,
            state="readonly",
        ).grid(row=row, column=5, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.note, width=26).grid(row=row, column=6, sticky="ew", padx=4, pady=2)
        ttk.Label(parent, textvariable=self.label_summary, anchor="w").grid(
            row=row, column=7, sticky="ew", padx=4, pady=2
        )
        for variable in (
            self.experiment_key,
            self.blot_id,
            self.file_kind,
            self.protein_label,
            self.protein_role,
        ):
            variable.trace_add("write", self._update_label_summary)
        self._update_label_summary()

    def _update_label_summary(self, *_args: object) -> None:
        self.label_summary.set(_image_label_summary(self.annotation()))

    def set_experiment_choices(self, choices: list[str]) -> None:
        self.experiment_combo.configure(values=choices)

    def annotation(self) -> FileAnnotation:
        return FileAnnotation(
            original_name=self.original_name,
            experiment_key=self.experiment_key.get().strip(),
            blot_id=self.blot_id.get().strip(),
            file_kind=self.file_kind.get().strip().upper(),
            protein_label=self.protein_label.get().strip(),
            protein_role=self.protein_role.get().strip().upper() or "UNKNOWN",
            note=self.note.get().strip(),
        )


class ExperimentRow:
    def __init__(
        self,
        parent: tk.Widget,
        row: int,
        key: str,
        date_value: str,
        experiment_id: str,
        cell_line: str,
        modification: str,
        treatment_name: str,
        dose_series: str = "",
        treatment_time: str = "",
        lane_direction: str = "LR",
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.key = tk.StringVar(value=key)
        self.date = tk.StringVar(value=date_value)
        self.experiment_id = tk.StringVar(value=experiment_id)
        self.cell_line = tk.StringVar(value=cell_line)
        self.modification = tk.StringVar(value=modification)
        self.treatment_name = tk.StringVar(value=treatment_name)
        self.dose_series = tk.StringVar(value=dose_series)
        self.treatment_time = tk.StringVar(value=treatment_time)
        self.lane_direction = tk.StringVar(value=lane_direction)
        self._on_change = on_change

        ttk.Entry(parent, textvariable=self.key, width=6).grid(row=row, column=0, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.date, width=10).grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.experiment_id, width=12).grid(row=row, column=2, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.cell_line, width=10).grid(row=row, column=3, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.modification, width=34).grid(row=row, column=4, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.treatment_name, width=20).grid(row=row, column=5, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.dose_series, width=20).grid(row=row, column=6, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.treatment_time, width=12).grid(row=row, column=7, sticky="ew", padx=4, pady=2)
        ttk.Combobox(
            parent,
            textvariable=self.lane_direction,
            values=list(LANE_DIRECTION_DEFINITIONS),
            width=8,
            state="readonly",
        ).grid(row=row, column=8, sticky="ew", padx=4, pady=2)
        for variable in (self.key, self.treatment_name, self.dose_series, self.treatment_time, self.lane_direction):
            variable.trace_add("write", self._notify_change)

    def _notify_change(self, *_args: object) -> None:
        if self._on_change:
            self._on_change()

    def metadata(self) -> tuple[str, ExperimentMetadata]:
        key = self.key.get().strip() or "E01"
        return key, ExperimentMetadata(
            date=self.date.get().strip(),
            experiment_id=self.experiment_id.get().strip(),
            cell_line=self.cell_line.get().strip(),
            modification=self.modification.get().strip(),
            treatment_name=self.treatment_name.get().strip(),
            dose_series=self.dose_series.get().strip(),
            treatment_time=self.treatment_time.get().strip(),
            target_protein="",
            loading_control="",
            lane_direction=self.lane_direction.get().strip() or "LR",
        )


class CellLineBlockRow:
    def __init__(
        self,
        parent: tk.Widget,
        row: int,
        experiment_key: str,
        block_number: int,
        cell_line: str,
        modification: str,
        lane_start: str = "",
        lane_end: str = "",
        note: str = "",
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self.experiment_key = tk.StringVar(value=experiment_key)
        self.block_number = tk.StringVar(value=str(block_number))
        self.cell_line = tk.StringVar(value=cell_line)
        self.modification = tk.StringVar(value=modification)
        self.lane_start = tk.StringVar(value=lane_start)
        self.lane_end = tk.StringVar(value=lane_end)
        self.note = tk.StringVar(value=note)
        self._on_change = on_change

        self.experiment_combo = ttk.Combobox(parent, textvariable=self.experiment_key, values=["E01"], width=8)
        self.experiment_combo.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.block_number, width=8).grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.cell_line, width=14).grid(row=row, column=2, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.modification, width=34).grid(row=row, column=3, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.lane_start, width=10).grid(row=row, column=4, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.lane_end, width=10).grid(row=row, column=5, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.note, width=28).grid(row=row, column=6, sticky="ew", padx=4, pady=2)
        for variable in (
            self.experiment_key,
            self.block_number,
            self.cell_line,
            self.modification,
            self.lane_start,
            self.lane_end,
        ):
            variable.trace_add("write", self._notify_change)

    def _notify_change(self, *_args: object) -> None:
        if self._on_change:
            self._on_change()

    def set_experiment_choices(self, choices: list[str]) -> None:
        self.experiment_combo.configure(values=choices)

    def annotation(self) -> CellLineBlock:
        return CellLineBlock(
            experiment_key=self.experiment_key.get().strip() or "E01",
            block_number=_parse_required_int(self.block_number.get().strip(), "cell-line block number"),
            cell_line=self.cell_line.get().strip(),
            modification=self.modification.get().strip(),
            lane_start=_parse_optional_int(self.lane_start.get().strip(), "lane start"),
            lane_end=_parse_optional_int(self.lane_end.get().strip(), "lane end"),
            note=self.note.get().strip(),
        )


class LaneRow:
    def __init__(self, parent: tk.Widget, row: int, lane_number: int, experiment_key: str = "E01") -> None:
        self.experiment_key = tk.StringVar(value=experiment_key)
        self.lane_number = tk.StringVar(value=str(lane_number))
        self.role = tk.StringVar(value="SMP")
        self.condition = tk.StringVar(value="")
        self.concentration = tk.StringVar(value="")
        self.note = tk.StringVar(value="")

        self.experiment_combo = ttk.Combobox(parent, textvariable=self.experiment_key, values=["E01"], width=8)
        self.experiment_combo.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.lane_number, width=8).grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        ttk.Combobox(
            parent,
            textvariable=self.role,
            values=list(LANE_ROLE_DEFINITIONS),
            width=8,
            state="readonly",
        ).grid(row=row, column=2, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.condition, width=24).grid(row=row, column=3, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.concentration, width=16).grid(row=row, column=4, sticky="ew", padx=4, pady=2)
        ttk.Entry(parent, textvariable=self.note, width=24).grid(row=row, column=5, sticky="ew", padx=4, pady=2)

    def set_experiment_choices(self, choices: list[str]) -> None:
        self.experiment_combo.configure(values=choices)

    def annotation(self) -> LaneAnnotation | None:
        lane_text = self.lane_number.get().strip()
        if not lane_text:
            return None
        try:
            lane_number = int(lane_text)
        except ValueError as exc:
            raise ValueError(f"Lane number must be an integer: {lane_text}") from exc

        return LaneAnnotation(
            lane_number=lane_number,
            role=self.role.get().strip().upper(),
            condition=self.condition.get().strip(),
            concentration=self.concentration.get().strip(),
            experiment_key=self.experiment_key.get().strip() or "E01",
            note=self.note.get().strip(),
        )


class WBAutoAnnotatorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("WB Folder Auto-Annotator")
        self.root.geometry("1440x1000")
        self.root.minsize(1180, 820)

        self.folder_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Select a folder to begin.")
        self.meta_vars = {
            "date": tk.StringVar(value=date.today().strftime("%Y%m%d")),
            "experiment_id": tk.StringVar(value="E001"),
            "cell_line": tk.StringVar(value=""),
            "modification": tk.StringVar(value="WT"),
            "treatment_name": tk.StringVar(value=""),
            "dose_series": tk.StringVar(value=""),
            "treatment_time": tk.StringVar(value=""),
            "target_protein": tk.StringVar(value=""),
            "loading_control": tk.StringVar(value=""),
        }

        self.experiment_rows: list[ExperimentRow] = []
        self.cell_line_block_rows: list[CellLineBlockRow] = []
        self.file_rows: list[FileRow] = []
        self.lane_rows: list[LaneRow] = []
        self.current_records: list[RenameRecord] = []
        self.current_metadata: ExperimentMetadata | None = None
        self.current_experiment_sets: dict[str, ExperimentMetadata] = {}
        self.current_cell_line_blocks: list[CellLineBlock] = []
        self.current_files: list[FileAnnotation] = []
        self.current_lanes: list[LaneAnnotation] = []
        self._lane_autofill_after_id: str | None = None
        self._building_ui_tables = False
        self.remove_lane_var = tk.StringVar(value="")
        self.batch_command_text: tk.Text | None = None

        self._build_layout()
        self._populate_default_lanes(8)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.page_scroll = ScrollableFrame(self.root, height=820)
        self.page_scroll.grid(row=0, column=0, sticky="nsew")
        self.page_scroll.content.columnconfigure(0, weight=1)

        main = ttk.Frame(self.page_scroll.content, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(8, weight=1)

        folder_frame = ttk.LabelFrame(main, text="Folder")
        folder_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        folder_frame.columnconfigure(1, weight=1)
        ttk.Button(folder_frame, text="Browse", command=self._browse_folder).grid(row=0, column=0, padx=6, pady=6)
        ttk.Entry(folder_frame, textvariable=self.folder_var).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(folder_frame, text="Pick Image File", command=self._browse_image_file).grid(
            row=0, column=2, padx=6, pady=6
        )
        ttk.Button(folder_frame, text="Scan Images", command=self._scan_folder).grid(row=0, column=3, padx=6, pady=6)
        ttk.Label(
            folder_frame,
            text="Tip: the Browse dialog shows folders only. If it says no items, click Select Folder or use Pick Image File.",
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 6))

        batch_frame = ttk.LabelFrame(main, text="Natural Language Batch Commands")
        batch_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        batch_frame.columnconfigure(0, weight=1)
        self.batch_command_text = tk.Text(batch_frame, height=4, wrap="word")
        self.batch_command_text.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        self.batch_command_text.insert(
            "1.0",
            "# WBScript: table[selector].field = value\n"
            "# files[B07:B12].exp = E02\n"
            "# blocks[E01,1].lanes = 1:2\n"
            "# blocks[E01,2].lanes = 3:13\n"
            "# exp[E02].dose = \"0-10-100nM\"\n",
        )
        batch_buttons = ttk.Frame(batch_frame)
        batch_buttons.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)
        ttk.Button(batch_buttons, text="Apply Commands", command=self._apply_batch_commands).grid(
            row=0, column=0, sticky="ew", pady=(0, 6)
        )
        ttk.Button(batch_buttons, text="Clear Commands", command=self._clear_batch_commands).grid(
            row=1, column=0, sticky="ew"
        )

        top = ttk.Frame(main)
        top.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        top.columnconfigure(0, weight=3)
        top.columnconfigure(1, weight=2)

        metadata_frame = ttk.LabelFrame(top, text="Experiment Metadata")
        metadata_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        for column in (1, 3, 5):
            metadata_frame.columnconfigure(column, weight=1)

        fields = [
            ("Date", "date"),
            ("Experiment ID", "experiment_id"),
            ("Cell line", "cell_line"),
            ("Modification", "modification"),
            ("Treatment", "treatment_name"),
            ("Default dose series", "dose_series"),
            ("Treatment time", "treatment_time"),
            ("Loading control", "loading_control"),
        ]
        for index, (label, key) in enumerate(fields):
            row = index // 3
            col = (index % 3) * 2
            ttk.Label(metadata_frame, text=label).grid(row=row, column=col, sticky="w", padx=6, pady=4)
            ttk.Entry(metadata_frame, textvariable=self.meta_vars[key], width=22).grid(
                row=row, column=col + 1, sticky="ew", padx=6, pady=4
            )

        role_frame = ttk.LabelFrame(top, text="Role Codes")
        role_frame.grid(row=0, column=1, sticky="nsew")
        role_text = "\n".join(f"{code}: {meaning}" for code, meaning in LANE_ROLE_DEFINITIONS.items())
        ttk.Label(role_frame, text=role_text, justify="left").grid(row=0, column=0, sticky="nw", padx=8, pady=6)

        experiment_frame = ttk.LabelFrame(main, text="Experiment Sets")
        experiment_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        experiment_frame.columnconfigure(0, weight=1)
        experiment_buttons = ttk.Frame(experiment_frame)
        experiment_buttons.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
        ttk.Button(experiment_buttons, text="Add Experiment", command=self._add_experiment).grid(
            row=0, column=0, padx=(0, 6)
        )
        ttk.Button(experiment_buttons, text="Clear Experiments", command=self._clear_experiments).grid(row=0, column=1)
        ttk.Label(
            experiment_buttons,
            text="Lane autofill reads Dose series here; edit each experiment row for different dose responses.",
        ).grid(row=0, column=2, sticky="w", padx=12)
        self.experiment_scroll = ScrollableFrame(experiment_frame, height=130)
        self.experiment_scroll.grid(row=1, column=0, sticky="ew", padx=6, pady=6)
        self._draw_experiment_header()

        cell_block_frame = ttk.LabelFrame(main, text="Cell Line Blocks")
        cell_block_frame.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        cell_block_frame.columnconfigure(0, weight=1)
        cell_block_buttons = ttk.Frame(cell_block_frame)
        cell_block_buttons.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
        ttk.Button(cell_block_buttons, text="Add Cell Line", command=self._add_cell_line_block).grid(
            row=0, column=0, padx=(0, 6)
        )
        ttk.Button(cell_block_buttons, text="Clear Cell Lines", command=self._clear_cell_line_blocks).grid(
            row=0, column=1
        )
        ttk.Label(
            cell_block_buttons,
            text="Add one row per cell line/plasmid block in the same blot; lane range is optional for downstream naming.",
        ).grid(row=0, column=2, sticky="w", padx=12)
        self.cell_line_block_scroll = ScrollableFrame(cell_block_frame, height=115)
        self.cell_line_block_scroll.grid(row=1, column=0, sticky="ew", padx=6, pady=6)
        self._draw_cell_line_block_header()

        file_frame = ttk.LabelFrame(main, text="Auto-Labeled Imported Images")
        file_frame.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        file_frame.columnconfigure(0, weight=1)
        self.file_scroll = ScrollableFrame(file_frame, height=180)
        self.file_scroll.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        self._draw_file_header()

        lane_frame = ttk.LabelFrame(main, text="Lane Order Metadata (saved for downstream quantification)")
        lane_frame.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        lane_frame.columnconfigure(0, weight=1)
        button_bar = ttk.Frame(lane_frame)
        button_bar.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
        ttk.Button(button_bar, text="Add Lane", command=self._add_lane).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(button_bar, text="Clear Lanes", command=self._clear_lanes).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(button_bar, text="Auto-Fill Lanes From Experiment Sets", command=self._auto_fill_lanes_from_doses).grid(
            row=0, column=2, padx=(0, 12)
        )
        ttk.Label(button_bar, text="Remove").grid(row=0, column=3, sticky="e", padx=(0, 4))
        ttk.Entry(button_bar, textvariable=self.remove_lane_var, width=10).grid(row=0, column=4, padx=(0, 4))
        ttk.Button(button_bar, text="Remove Lane", command=self._remove_lane).grid(row=0, column=5, padx=(0, 12))
        ttk.Label(
            button_bar,
            text="Use 3 or E02:3. Refill uses Experiment Sets > Dose series.",
        ).grid(row=0, column=6, sticky="w")
        self.lane_scroll = ScrollableFrame(lane_frame, height=165)
        self.lane_scroll.grid(row=1, column=0, sticky="ew", padx=6, pady=6)
        self._draw_lane_header()

        preview_frame = ttk.LabelFrame(main, text="Label Preview")
        preview_frame.grid(row=8, column=0, sticky="nsew", pady=(0, 8))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        columns = ("original", "new", "status", "message")
        self.preview_tree = ttk.Treeview(preview_frame, columns=columns, show="headings", height=10)
        self.preview_tree.heading("original", text="Current image file")
        self.preview_tree.heading("new", text="Applied label filename")
        self.preview_tree.heading("status", text="Status")
        self.preview_tree.heading("message", text="Message")
        self.preview_tree.column("original", width=260, anchor="w")
        self.preview_tree.column("new", width=420, anchor="w")
        self.preview_tree.column("status", width=90, anchor="center")
        self.preview_tree.column("message", width=380, anchor="w")
        preview_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_tree.yview)
        self.preview_tree.configure(yscrollcommand=preview_scroll.set)
        self.preview_tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        preview_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)

        controls = ttk.Frame(main)
        controls.grid(row=9, column=0, sticky="ew")
        controls.columnconfigure(3, weight=1)
        ttk.Button(controls, text="Preview Labeled Filenames", command=self._preview).grid(
            row=0, column=0, padx=(0, 6)
        )
        ttk.Button(controls, text="RENAME FILES", command=self._apply).grid(
            row=0, column=1, padx=(0, 6)
        )
        ttk.Button(controls, text="EXPORT LABEL MAP", command=self._export_label_map).grid(
            row=0, column=2, padx=(0, 12)
        )
        ttk.Label(controls, textvariable=self.status_var).grid(row=0, column=3, sticky="w")

    def _draw_file_header(self) -> None:
        headers = ("Original filename", "Exp Set", "Blot ID", "Data type", "Protein", "Role", "Note", "Auto label")
        for column, text in enumerate(headers):
            ttk.Label(self.file_scroll.content, text=text, font=("", 9, "bold")).grid(
                row=0, column=column, sticky="ew", padx=4, pady=2
            )
        self.file_scroll.content.columnconfigure(0, weight=3)
        self.file_scroll.content.columnconfigure(6, weight=1)
        self.file_scroll.content.columnconfigure(7, weight=1)

    def _draw_experiment_header(self) -> None:
        headers = (
            "Set",
            "Date",
            "Experiment ID",
            "Cell line",
            "Modification / plasmids",
            "Treatment",
            "Dose series",
            "Time",
            "Direction",
        )
        for column, text in enumerate(headers):
            ttk.Label(self.experiment_scroll.content, text=text, font=("", 9, "bold")).grid(
                row=0, column=column, sticky="ew", padx=4, pady=2
            )
        self.experiment_scroll.content.columnconfigure(4, weight=2)
        self.experiment_scroll.content.columnconfigure(6, weight=1)

    def _draw_cell_line_block_header(self) -> None:
        headers = ("Exp Set", "Block #", "Cell line", "Modification / plasmids", "Lane start", "Lane end", "Note")
        for column, text in enumerate(headers):
            ttk.Label(self.cell_line_block_scroll.content, text=text, font=("", 9, "bold")).grid(
                row=0, column=column, sticky="ew", padx=4, pady=2
            )
        self.cell_line_block_scroll.content.columnconfigure(3, weight=2)
        self.cell_line_block_scroll.content.columnconfigure(6, weight=1)

    def _draw_lane_header(self) -> None:
        headers = ("Exp Set", "Lane #", "Role", "Condition / treatment", "Concentration", "Replicate / note")
        for column, text in enumerate(headers):
            ttk.Label(self.lane_scroll.content, text=text, font=("", 9, "bold")).grid(
                row=0, column=column, sticky="ew", padx=4, pady=2
            )
        self.lane_scroll.content.columnconfigure(3, weight=1)
        self.lane_scroll.content.columnconfigure(5, weight=1)

    def _browse_folder(self) -> None:
        selected = filedialog.askdirectory(title="Select WB image folder")
        if selected:
            self.folder_var.set(selected)
            self._scan_folder()

    def _browse_image_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select one WB image file",
            filetypes=[
                ("WB image files", " ".join(f"*{extension}" for extension in SUPPORTED_EXTENSIONS)),
                ("TIFF files", "*.tif *.tiff *.raw16.tif *.raw16tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.folder_var.set(str(Path(selected).parent))
            self._scan_folder()

    def _scan_folder(self) -> None:
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showwarning("No folder", "Choose a folder first.")
            return
        try:
            files = scan_image_files(folder)
        except (FileNotFoundError, NotADirectoryError) as exc:
            messagebox.showerror("Folder error", str(exc))
            return
        self._apply_folder_metadata_guesses(Path(folder))
        annotations = auto_annotate_files(files)
        self._populate_experiment_rows(Path(folder), annotations)
        self._populate_cell_line_blocks_from_experiments()
        self._populate_file_rows(annotations)
        self.current_records = []
        self._clear_preview()
        blot_count = len({row.blot_id.get() for row in self.file_rows})
        experiment_count = len({row.experiment_key.get() for row in self.file_rows})
        self.status_var.set(
            f"Found {len(files)} image file(s). Auto-labeled {experiment_count} experiment set(s), "
            f"{blot_count} blot group(s), data types, and proteins."
        )

    def _populate_file_rows(self, annotations: list[FileAnnotation]) -> None:
        for child in self.file_scroll.content.winfo_children():
            child.destroy()
        self.file_rows.clear()
        self._draw_file_header()
        for index, annotation in enumerate(annotations, start=1):
            self.file_rows.append(
                FileRow(
                    self.file_scroll.content,
                    index,
                    annotation.original_name,
                    annotation.experiment_key,
                    annotation.blot_id,
                    annotation.file_kind,
                    annotation.protein_label,
                    annotation.protein_role,
                    annotation.note,
                )
            )
        self._refresh_experiment_choices()

    def _apply_batch_commands(self) -> None:
        if self.batch_command_text is None:
            return

        command_text = self.batch_command_text.get("1.0", "end").strip()
        if not command_text:
            messagebox.showinfo("No commands", "Enter at least one batch edit command.")
            return

        try:
            commands = parse_mixed_batch_commands(command_text)
        except ValueError as exc:
            messagebox.showerror("Batch command error", str(exc))
            return

        if commands.file_rules and not self.file_rows:
            messagebox.showwarning("No files", "Scan a folder before applying file-edit commands.")
            return

        annotations = self._collect_files() if self.file_rows else []
        updated = annotations
        file_change_count = 0
        if commands.file_rules:
            updated, file_change_count = apply_batch_rules(annotations, commands.file_rules)
            self._ensure_experiment_rows_for_files(updated)
            self._populate_file_rows(updated)
        experiment_change_count = self._apply_experiment_rules(commands.experiment_rules)
        block_change_count = self._apply_cell_line_block_rules(commands.cell_line_block_rules)
        self.current_records = []
        self._clear_preview()
        command_count = (
            len(commands.file_rules)
            + len(commands.experiment_rules)
            + len(commands.cell_line_block_rules)
        )
        self.status_var.set(
            f"Applied {command_count} command(s); "
            f"changed {file_change_count} file value(s), {experiment_change_count} experiment field(s), "
            f"and {block_change_count} cell-line block(s)."
        )

    def _apply_experiment_rules(self, rules: list[ExperimentRule]) -> int:
        change_count = 0
        for rule in rules:
            self._ensure_experiment_row(rule.experiment_key)
            row = self._find_experiment_row(rule.experiment_key)
            if row is None:
                continue
            variable = getattr(row, _experiment_row_var_name(rule.field), None)
            if variable is None:
                continue
            if variable.get() != rule.value:
                variable.set(rule.value)
                change_count += 1
        if rules:
            self._refresh_experiment_choices()
            self._schedule_lane_autofill()
        return change_count

    def _apply_cell_line_block_rules(self, rules: list[CellLineBlockRule]) -> int:
        change_count = 0
        for rule in rules:
            self._ensure_experiment_row(rule.experiment_key)
            row = self._find_cell_line_block_row(rule.experiment_key, rule.block_number)
            if row is None:
                metadata = self._collect_experiment_sets().get(rule.experiment_key)
                row_number = len(self.cell_line_block_rows) + 1
                row = CellLineBlockRow(
                    self.cell_line_block_scroll.content,
                    row_number,
                    rule.experiment_key,
                    rule.block_number,
                    metadata.cell_line if metadata else "",
                    metadata.modification if metadata else "",
                    on_change=self._schedule_lane_autofill,
                )
                self.cell_line_block_rows.append(row)
                self._refresh_experiment_choices()
            row.lane_start.set(str(rule.lane_start))
            row.lane_end.set(str(rule.lane_end))
            change_count += 1
        if rules:
            self._auto_fill_lanes_from_doses(False)
        return change_count

    def _find_experiment_row(self, experiment_key: str) -> ExperimentRow | None:
        for row in self.experiment_rows:
            if row.key.get().strip() == experiment_key:
                return row
        return None

    def _find_cell_line_block_row(self, experiment_key: str, block_number: int) -> CellLineBlockRow | None:
        for row in self.cell_line_block_rows:
            if row.experiment_key.get().strip() != experiment_key:
                continue
            try:
                if int(row.block_number.get().strip()) == block_number:
                    return row
            except ValueError:
                continue
        return None

    def _clear_batch_commands(self) -> None:
        if self.batch_command_text is not None:
            self.batch_command_text.delete("1.0", "end")

    def _ensure_experiment_rows_for_files(self, files: list[FileAnnotation]) -> None:
        existing = set(self._experiment_keys())
        needed = sorted({item.experiment_key for item in files if item.experiment_key})
        for key in needed:
            if self._ensure_experiment_row(key):
                existing.add(key)

    def _ensure_experiment_row(self, key: str) -> bool:
        if key in set(self._experiment_keys()):
            return False
        self._add_experiment_with_key(key)
        return True

    def _add_experiment_with_key(self, key: str) -> None:
        template = self.experiment_rows[0].metadata()[1] if self.experiment_rows else self._collect_metadata()
        row_number = len(self.experiment_rows) + 1
        self.experiment_rows.append(
            ExperimentRow(
                self.experiment_scroll.content,
                row_number,
                key,
                template.date,
                f"{template.experiment_id}-{key}" if key not in template.experiment_id else template.experiment_id,
                template.cell_line,
                template.modification,
                template.treatment_name,
                template.dose_series,
                template.treatment_time,
                template.lane_direction,
                on_change=self._schedule_lane_autofill,
            )
        )
        self._refresh_experiment_choices()

    def _populate_experiment_rows(self, folder: Path, annotations: list[FileAnnotation]) -> None:
        self._building_ui_tables = True
        for child in self.experiment_scroll.content.winfo_children():
            child.destroy()
        self.experiment_rows.clear()
        self._draw_experiment_header()

        date_value = self.meta_vars["date"].get().strip() or date.today().strftime("%Y%m%d")
        base_experiment_id = self._simplified_experiment_id(folder.parent.name)
        cell_line = self._infer_cell_line(folder.parent.name)
        modification = self._infer_modification(folder.parent.name)
        default_dose_series = self.meta_vars["dose_series"].get().strip()
        default_treatment_time = self.meta_vars["treatment_time"].get().strip()

        group_names: dict[str, str] = {}
        for annotation in annotations:
            group_names.setdefault(annotation.experiment_key, experiment_group_key(annotation.original_name))

        if not group_names:
            group_names = {"E01": ""}

        for index, key in enumerate(sorted(group_names), start=1):
            experiment_id = base_experiment_id if len(group_names) == 1 else f"{base_experiment_id}-{key}"
            treatment = group_names[key]
            self.experiment_rows.append(
                ExperimentRow(
                    self.experiment_scroll.content,
                    index,
                    key,
                    date_value,
                    experiment_id,
                    cell_line,
                    modification,
                    treatment,
                    default_dose_series,
                    default_treatment_time,
                    on_change=self._schedule_lane_autofill,
                )
            )
        self._building_ui_tables = False
        self._refresh_experiment_choices()

    def _populate_cell_line_blocks_from_experiments(self) -> None:
        self._clear_cell_line_blocks()
        for row in self.experiment_rows:
            key, metadata = row.metadata()
            self._add_cell_line_block(
                experiment_key=key,
                cell_line=metadata.cell_line,
                modification=metadata.modification,
            )

    def _apply_folder_metadata_guesses(self, folder: Path) -> None:
        guessed_date = self._guess_date_from_path(folder)
        today_text = date.today().strftime("%Y%m%d")
        if guessed_date and self.meta_vars["date"].get().strip() in {"", today_text}:
            self.meta_vars["date"].set(guessed_date)

        if self.meta_vars["experiment_id"].get().strip() in {"", "E001"}:
            self.meta_vars["experiment_id"].set(self._simplified_experiment_id(folder.parent.name))

        if not self.meta_vars["cell_line"].get().strip():
            self.meta_vars["cell_line"].set(self._infer_cell_line(folder.parent.name))

        if self.meta_vars["modification"].get().strip() in {"", "WT"}:
            modification = self._infer_modification(folder.parent.name)
            if modification:
                self.meta_vars["modification"].set(modification)

    def _guess_date_from_path(self, folder: Path) -> str:
        for part in (folder.name, folder.parent.name):
            match = re.fullmatch(r"(\d{1,2})[_\-.](\d{1,2})[_\-.](\d{2,4})", part)
            if match:
                month, day, year = match.groups()
                if len(year) == 2:
                    year = f"20{year}"
                return f"{int(year):04d}{int(month):02d}{int(day):02d}"

            match = re.fullmatch(r"(\d{4})[_\-.](\d{1,2})[_\-.](\d{1,2})", part)
            if match:
                year, month, day = match.groups()
                return f"{int(year):04d}{int(month):02d}{int(day):02d}"
        return ""

    def _infer_cell_line(self, folder_name: str) -> str:
        first_token = re.split(r"[_\-\s]+", folder_name.strip())[0]
        return first_token.upper() if first_token else self.meta_vars["cell_line"].get().strip()

    def _simplified_experiment_id(self, folder_name: str) -> str:
        tokens = [token for token in re.split(r"[_\-\s]+", folder_name.strip()) if token]
        genes = tokens[1:] if len(tokens) > 1 else tokens
        abbreviated = "".join(self._abbreviate_gene_token(token) for token in genes)
        return abbreviated or self.meta_vars["experiment_id"].get().strip() or "E001"

    def _abbreviate_gene_token(self, token: str) -> str:
        match = re.fullmatch(r"([A-Za-z]+?)(\d+)", token)
        if match:
            letters, number = match.groups()
            return f"{letters[0].upper()}{number}"
        return re.sub(r"[^A-Za-z0-9]", "", token).upper()

    def _infer_modification(self, folder_name: str) -> str:
        tokens = [token for token in re.split(r"[_\-\s]+", folder_name.strip()) if token]
        genes = tokens[1:] if len(tokens) > 1 else []
        if not genes:
            return self.meta_vars["modification"].get().strip() or "WT"
        return "; ".join(f"human {gene.upper()} wild-type" for gene in genes)

    def _clear_experiments(self) -> None:
        for child in self.experiment_scroll.content.winfo_children():
            child.destroy()
        self.experiment_rows.clear()
        self._draw_experiment_header()
        self._refresh_experiment_choices()

    def _add_experiment(self) -> None:
        choices = self._experiment_keys()
        next_number = 1
        while f"E{next_number:02d}" in choices:
            next_number += 1
        key = f"E{next_number:02d}"
        row_number = len(self.experiment_rows) + 1
        self.experiment_rows.append(
            ExperimentRow(
                self.experiment_scroll.content,
                row_number,
                key,
                self.meta_vars["date"].get().strip() or date.today().strftime("%Y%m%d"),
                f"{self.meta_vars['experiment_id'].get().strip() or 'EXP'}-{key}",
                self.meta_vars["cell_line"].get().strip(),
                self.meta_vars["modification"].get().strip(),
                self.meta_vars["treatment_name"].get().strip(),
                self.meta_vars["dose_series"].get().strip(),
                self.meta_vars["treatment_time"].get().strip(),
                on_change=self._schedule_lane_autofill,
            )
        )
        self._refresh_experiment_choices()
        self._schedule_lane_autofill()

    def _clear_cell_line_blocks(self) -> None:
        for child in self.cell_line_block_scroll.content.winfo_children():
            child.destroy()
        self.cell_line_block_rows.clear()
        self._draw_cell_line_block_header()
        self._refresh_experiment_choices()

    def _add_cell_line_block(
        self,
        experiment_key: str | None = None,
        cell_line: str | None = None,
        modification: str | None = None,
    ) -> None:
        choices = self._experiment_keys()
        key = experiment_key or choices[0]
        row_number = len(self.cell_line_block_rows) + 1
        block_row = CellLineBlockRow(
            self.cell_line_block_scroll.content,
            row_number,
            key,
            row_number,
            cell_line if cell_line is not None else self.meta_vars["cell_line"].get().strip(),
            modification if modification is not None else self.meta_vars["modification"].get().strip(),
            on_change=self._schedule_lane_autofill,
        )
        self.cell_line_block_rows.append(block_row)
        self._refresh_experiment_choices()
        self._schedule_lane_autofill()

    def _experiment_keys(self) -> list[str]:
        keys = [row.key.get().strip() for row in self.experiment_rows if row.key.get().strip()]
        return keys or ["E01"]

    def _refresh_experiment_choices(self) -> None:
        choices = self._experiment_keys()
        for row in self.file_rows:
            row.set_experiment_choices(choices)
            if row.experiment_key.get().strip() not in choices:
                row.experiment_key.set(choices[0])
        for row in self.lane_rows:
            row.set_experiment_choices(choices)
            if row.experiment_key.get().strip() not in choices:
                row.experiment_key.set(choices[0])
        for row in self.cell_line_block_rows:
            row.set_experiment_choices(choices)
            if row.experiment_key.get().strip() not in choices:
                row.experiment_key.set(choices[0])

    def _schedule_lane_autofill(self) -> None:
        if self._building_ui_tables:
            return
        if self._lane_autofill_after_id is not None:
            self.root.after_cancel(self._lane_autofill_after_id)
        self._lane_autofill_after_id = self.root.after(700, lambda: self._auto_fill_lanes_from_doses(False))

    def _auto_fill_lanes_from_doses(self, show_empty_message: bool = True) -> None:
        self._lane_autofill_after_id = None
        try:
            self._copy_default_dose_to_empty_experiment_rows()
            experiment_sets = self._collect_experiment_sets()
            cell_line_blocks = self._collect_cell_line_blocks()
        except ValueError as exc:
            if show_empty_message:
                messagebox.showerror("Dose autofill error", str(exc))
            return

        lanes = build_lane_annotations_from_experiments(experiment_sets, cell_line_blocks)
        if not lanes:
            if show_empty_message:
                messagebox.showinfo(
                    "No dose series",
                    "Enter a dose series in Experiment Sets, for example 0-10-100nM.",
                )
            return

        self._populate_lane_rows(lanes)
        self.status_var.set(f"Auto-filled {len(lanes)} lane row(s) from experiment dose series.")

    def _copy_default_dose_to_empty_experiment_rows(self) -> None:
        default_dose_series = self.meta_vars["dose_series"].get().strip()
        default_treatment_time = self.meta_vars["treatment_time"].get().strip()
        if not default_dose_series and not default_treatment_time:
            return
        for row in self.experiment_rows:
            if default_dose_series and not row.dose_series.get().strip():
                row.dose_series.set(default_dose_series)
            if default_treatment_time and not row.treatment_time.get().strip():
                row.treatment_time.set(default_treatment_time)

    def _populate_lane_rows(self, lanes: list[LaneAnnotation]) -> None:
        self._clear_lanes()
        for index, lane in enumerate(lanes, start=1):
            lane_row = LaneRow(self.lane_scroll.content, index, lane.lane_number, lane.experiment_key)
            lane_row.role.set(lane.role)
            lane_row.condition.set(lane.condition)
            lane_row.concentration.set(lane.concentration)
            lane_row.note.set(lane.note)
            self.lane_rows.append(lane_row)
        self._refresh_experiment_choices()

    def _populate_default_lanes(self, count: int) -> None:
        self._clear_lanes()
        for _ in range(count):
            self._add_lane()

    def _clear_lanes(self) -> None:
        for child in self.lane_scroll.content.winfo_children():
            child.destroy()
        self.lane_rows.clear()
        self._draw_lane_header()

    def _add_lane(self) -> None:
        row_number = len(self.lane_rows) + 1
        experiment_key = self._experiment_keys()[0]
        lane_row = LaneRow(self.lane_scroll.content, row_number, self._next_lane_number(experiment_key), experiment_key)
        self.lane_rows.append(lane_row)
        self._refresh_experiment_choices()

    def _next_lane_number(self, experiment_key: str) -> int:
        lane_numbers: list[int] = []
        for row in self.lane_rows:
            if row.experiment_key.get().strip() != experiment_key:
                continue
            lane_text = row.lane_number.get().strip()
            if not lane_text:
                continue
            try:
                lane_numbers.append(int(lane_text))
            except ValueError:
                continue
        return max(lane_numbers, default=0) + 1

    def _remove_lane(self) -> None:
        try:
            experiment_key, lane_number = _parse_lane_remove_selector(self.remove_lane_var.get())
            lanes = self._collect_lanes()
        except ValueError as exc:
            messagebox.showerror("Remove lane error", str(exc))
            return

        remaining: list[LaneAnnotation] = []
        removed: LaneAnnotation | None = None
        for lane in lanes:
            matches_lane = lane.lane_number == lane_number
            matches_experiment = experiment_key is None or lane.experiment_key == experiment_key
            if removed is None and matches_lane and matches_experiment:
                removed = lane
                continue
            remaining.append(lane)

        if removed is None:
            prefix = f"{experiment_key} " if experiment_key else ""
            messagebox.showwarning("Lane not found", f"No {prefix}lane {lane_number} row was found.")
            return

        self._populate_lane_rows(remaining)
        self.remove_lane_var.set("")
        self.status_var.set(f"Removed {removed.experiment_key} lane {removed.lane_number}.")

    def _collect_metadata(self) -> ExperimentMetadata:
        experiment_sets = self._collect_experiment_sets()
        if experiment_sets:
            return next(iter(experiment_sets.values()))
        return ExperimentMetadata(
            date=self.meta_vars["date"].get().strip(),
            experiment_id=self.meta_vars["experiment_id"].get().strip(),
            cell_line=self.meta_vars["cell_line"].get().strip(),
            modification=self.meta_vars["modification"].get().strip(),
            treatment_name=self.meta_vars["treatment_name"].get().strip(),
            dose_series=self.meta_vars["dose_series"].get().strip(),
            treatment_time=self.meta_vars["treatment_time"].get().strip(),
            target_protein=self.meta_vars["target_protein"].get().strip(),
            loading_control=self.meta_vars["loading_control"].get().strip(),
        )

    def _collect_experiment_sets(self) -> dict[str, ExperimentMetadata]:
        experiment_sets: dict[str, ExperimentMetadata] = {}
        for row in self.experiment_rows:
            key, metadata = row.metadata()
            experiment_sets[key] = metadata
        return experiment_sets

    def _collect_cell_line_blocks(self) -> list[CellLineBlock]:
        return [row.annotation() for row in self.cell_line_block_rows]

    def _collect_files(self) -> list[FileAnnotation]:
        return [row.annotation() for row in self.file_rows]

    def _collect_lanes(self) -> list[LaneAnnotation]:
        lanes: list[LaneAnnotation] = []
        for row in self.lane_rows:
            annotation = row.annotation()
            if annotation is not None:
                lanes.append(annotation)
        return lanes

    def _preview(self) -> None:
        folder = self.folder_var.get().strip()
        if not folder:
            messagebox.showwarning("No folder", "Choose a folder first.")
            return
        if not self.file_rows:
            messagebox.showwarning("No files", "Scan a folder with supported image files first.")
            return

        try:
            metadata = self._collect_metadata()
            experiment_sets = self._collect_experiment_sets()
            cell_line_blocks = self._collect_cell_line_blocks()
            files = self._collect_files()
            lanes = self._collect_lanes()
        except ValueError as exc:
            messagebox.showerror("Metadata error", str(exc))
            return

        records = build_rename_plan(folder, metadata, files, lanes, experiment_sets)
        self.current_metadata = metadata
        self.current_experiment_sets = experiment_sets
        self.current_cell_line_blocks = cell_line_blocks
        self.current_files = files
        self.current_lanes = lanes
        self.current_records = records
        self._render_preview(records)

        blocked = sum(1 for record in records if record.status != "OK")
        if blocked:
            self.status_var.set(f"Label preview complete: {blocked} row(s) blocked. Fix metadata/table values before labeling.")
        else:
            self.status_var.set(f"Label preview complete: {len(records)} image file(s) ready.")

    def _apply(self) -> None:
        if not self.current_records:
            self._preview()
            if not self.current_records:
                return

        blocked = [record for record in self.current_records if record.status != "OK"]
        if blocked:
            messagebox.showerror("Blocked rows", "Fix blocked preview rows before labeling images.")
            return

        if self.current_metadata is None:
            messagebox.showerror("No preview", "Generate a label preview before labeling images.")
            return

        confirmed = messagebox.askyesno(
            "Label images",
            f"Apply labels to {len(self.current_records)} image file(s) and write wb_metadata.json / wb_rename_log.csv?",
        )
        if not confirmed:
            return

        folder = self.folder_var.get().strip()
        records = apply_rename_plan(
            folder,
            self.current_metadata,
            self.current_files,
            self.current_lanes,
            self.current_records,
            self.current_experiment_sets,
            self.current_cell_line_blocks,
        )
        self.current_records = records
        self._render_preview(records)
        renamed = sum(1 for record in records if record.status == "RENAMED")
        unchanged = sum(1 for record in records if record.status == "UNCHANGED")
        failed = sum(1 for record in records if record.status == "FAILED")
        self.status_var.set(f"Labeling complete: {renamed} labeled, {unchanged} already labeled, {failed} failed.")
        if failed:
            messagebox.showwarning("Labeling finished with failures", "Some files could not be labeled. See preview/log.")
        else:
            messagebox.showinfo("Labeling complete", "Images labeled and logs written.")

    def _export_label_map(self) -> None:
        self._preview()
        if not self.current_records or self.current_metadata is None:
            return

        folder = self.folder_var.get().strip()
        csv_path, json_path = write_label_export(
            folder,
            self.current_metadata,
            self.current_files,
            self.current_lanes,
            self.current_records,
            self.current_experiment_sets,
            self.current_cell_line_blocks,
        )
        self.status_var.set(f"Exported label map: {csv_path.name} and {json_path.name}.")
        messagebox.showinfo(
            "Export complete",
            f"Exported label map files:\n{csv_path.name}\n{json_path.name}",
        )

    def _render_preview(self, records: list[RenameRecord]) -> None:
        self._clear_preview()
        for record in records:
            self.preview_tree.insert(
                "",
                "end",
                values=(record.original_name, record.new_name, record.status, record.message),
            )

    def _clear_preview(self) -> None:
        for item_id in self.preview_tree.get_children():
            self.preview_tree.delete(item_id)


def main() -> None:
    root = tk.Tk()
    app = WBAutoAnnotatorApp(root)
    root.mainloop()


__all__ = ["WBAutoAnnotatorApp", "main"]
