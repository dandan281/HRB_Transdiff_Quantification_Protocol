from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ProteinHit:
    label: str
    role: str
    matched_alias: str


# Intentionally small and editable. Lab-specific aliases come first so they win
# over broad public/common Western blot controls.
LOADING_CONTROL_ALIASES = {
    "H3": [
        "h3",
        "histone h3",
        "histone-h3",
        "total h3",
        "total-h3",
    ],
    "S6": [
        "s6",
        "rps6",
        "ribosomal protein s6",
        "ribosomal-protein-s6",
        "total s6",
        "total-s6",
    ],
    "GAPDH": ["gapdh", "g3pdh"],
    "ACTB": ["actb", "beta actin", "beta-actin", "b-actin", "β-actin", "actin"],
    "Tubulin": ["tubulin", "alpha tubulin", "alpha-tubulin", "beta tubulin", "beta-tubulin", "tuba", "tubb"],
    "Vinculin": ["vinculin"],
    "LaminB1": ["lamin b1", "lamin-b1", "laminb1"],
    "TotalProtein": ["total protein", "total-protein", "ponceau", "stain-free", "stainfree", "coomassie"],
}

TARGET_PROTEIN_ALIASES = {
    "pERK": [
        "perk",
        "p-erk",
        "p erk",
        "phospho erk",
        "phospho-erk",
        "p44/42",
        "p-p44/42",
        "p-erc",
        "perc",
    ],
    "ERK": ["erk", "erk1/2", "erk1-2"],
    "pAKT": [
        "pakt",
        "p-akt",
        "p akt",
        "phospho akt",
        "phospho-akt",
        "mpakt",
        "m pakt",
    ],
    "AKT": ["akt", "akt1", "akt2"],
    "pP38": [
        "pp38",
        "p-p38",
        "p p38",
        "phospho p38",
        "phospho-p38",
        "p38-ish",
        "pp38-ish",
    ],
    "P38": ["p38", "p38 mapk", "p38-mapk"],
    "BMPR2": ["bmpr2", "bmp receptor type 2"],
    "HER2": ["her2", "erbb2"],
    "SMAD1/5": ["smad1/5", "smad1-5", "psmad1/5", "p-smad1/5", "p-smad1-5"],
}


def infer_protein_from_text(text: str) -> ProteinHit | None:
    normalized = _normalize_for_match(text)
    for label, aliases in LOADING_CONTROL_ALIASES.items():
        alias = _first_matching_alias(normalized, aliases)
        if alias:
            return ProteinHit(label=label, role="LC", matched_alias=alias)

    for label, aliases in TARGET_PROTEIN_ALIASES.items():
        alias = _first_matching_alias(normalized, aliases)
        if alias:
            return ProteinHit(label=label, role="TGT", matched_alias=alias)

    return None


def _first_matching_alias(normalized_text: str, aliases: list[str]) -> str:
    for alias in aliases:
        normalized_alias = _normalize_for_match(alias)
        if _contains_tokenish(normalized_text, normalized_alias):
            return alias
    return ""


def _normalize_for_match(text: str) -> str:
    lowered = text.casefold().replace("β", "beta")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return f" {re.sub(r'\\s+', ' ', lowered).strip()} "


def _contains_tokenish(text: str, alias: str) -> bool:
    if not alias.strip():
        return False
    if f" {alias.strip()} " in text:
        return True

    # Handle compact protein labels inside filenames such as "_perk" or "_h3".
    compact_text = re.sub(r"\s+", "", text)
    compact_alias = re.sub(r"\s+", "", alias)
    return bool(re.search(rf"(^|[^a-z0-9]){re.escape(compact_alias)}($|[^a-z0-9])", compact_text))
