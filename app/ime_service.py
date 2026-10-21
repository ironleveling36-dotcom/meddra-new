"""
IME (Important Medical Events) Service — fixed v2.1

Bugs fixed vs v2.0:
  ✗ Filename mismatch: was hunting 'Important-medical-event-29_1.xlsx',
    actual file is '29-0_ime_list-en.xlsx'. Now auto-detects any IME Excel in data dir.
  ✗ Sheet name mismatch: hardcoded '29.1 IME List', actual sheet '29.0 IME List'.
    Now auto-detects the correct sheet regardless of version number.
  ✗ FATAL LOGIC: `if ime_flag == 'X'` only loaded 45 NEW terms (added in 29.0)
    and silently skipped 7,757 existing IME terms. Every row in the file IS an IME;
    the X flag means "new in this version", not "is IME". Now loads ALL rows.

Result: 7,802 IME terms correctly loaded instead of 45.
"""
import os
import logging

import pandas as pd

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("MEDDRA_DATA_DIR", os.path.join(HERE, "..", "data"))


def _find_ime_file(data_dir: str) -> str | None:
    """
    Auto-detect the IME Excel file in data_dir.
    Tries known names first, then any file matching *ime* pattern.
    """
    candidates = [
        # Known filenames across versions (add new ones here as EMA releases them)
        "29-0_ime_list-en.xlsx",
        "29-1_ime_list-en.xlsx",
        "28-1_ime_list-en.xlsx",
        "28-0_ime_list-en.xlsx",
        "Important-medical-event-29_1.xlsx",
        "Important-medical-event-29_0.xlsx",
        "Important-medical-event-28_1.xlsx",
        "IME.xlsx",
        "ime_list.xlsx",
        "important_medical_events.xlsx",
    ]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path

    # Fallback: any xlsx with 'ime' in the name (case-insensitive)
    try:
        for fn in os.listdir(data_dir):
            if fn.lower().endswith(".xlsx") and "ime" in fn.lower():
                return os.path.join(data_dir, fn)
    except OSError:
        pass

    return None


def _detect_ime_sheet(xl: pd.ExcelFile) -> str:
    """Pick the sheet that contains the IME list, regardless of version number."""
    for sheet in xl.sheet_names:
        if "ime list" in sheet.lower() or "ime_list" in sheet.lower():
            return sheet
    # If nothing matches, use the first sheet
    return xl.sheet_names[0]


class IMEIndex:
    """Fast lookup index for Important Medical Events."""

    def __init__(self, excel_file: str | None = None):
        self.ime_ids: set[int] = set()          # MedDRA PT codes
        self.ime_names: set[str] = set()        # lowercase PT names
        self.ime_details: dict[int, dict] = {}  # id → detail record
        self.total_loaded = 0

        path = excel_file or _find_ime_file(DATA_DIR)
        if path:
            self._load(path)
        else:
            logger.warning(
                f"No IME Excel file found in {DATA_DIR}. "
                "IME classification will be disabled. "
                "Expected filename pattern: *ime*.xlsx"
            )

    def _load(self, file_path: str):
        if not os.path.exists(file_path):
            logger.warning(f"IME file not found: {file_path}")
            return

        try:
            logger.info(f"Loading IME data from {file_path} …")
            xl = pd.ExcelFile(file_path)
            sheet = _detect_ime_sheet(xl)
            logger.info(f"Using sheet: '{sheet}'")

            # The file has ~10 metadata rows before the real header row.
            # Read with header=None and skip 11 rows to land right before
            # the column-name row; then drop that row and rename columns.
            df = pd.read_excel(xl, sheet_name=sheet, header=None, skiprows=11)

            # Row 0 is the column header text — drop it, assign clean names
            df.columns = ["code", "pt_name", "soc", "comment", "added_flag", "soc_change"]
            df = df.iloc[1:].reset_index(drop=True)

            # Keep only rows with a valid numeric MedDRA code
            df["code"] = pd.to_numeric(df["code"], errors="coerce")
            df = df.dropna(subset=["code"])
            df["code"] = df["code"].astype(int)

            # EVERY row in this file is an IME term.
            # 'added_flag' == 'X' means it's new in this version — not the IME flag.
            for _, row in df.iterrows():
                tid = int(row["code"])
                name = str(row["pt_name"]).strip() if pd.notna(row["pt_name"]) else ""
                soc = str(row["soc"]).strip() if pd.notna(row["soc"]) else ""

                self.ime_ids.add(tid)
                if name:
                    self.ime_names.add(name.lower())
                self.ime_details[tid] = {
                    "term_id": tid,
                    "name": name,
                    "soc": soc,
                    "is_ime": True,
                    "new_in_version": str(row.get("added_flag", "")).strip() == "X",
                }
                self.total_loaded += 1

            logger.info(f"✅ Loaded {self.total_loaded} IME terms from '{sheet}'")

        except Exception as e:
            logger.error(f"Failed to load IME data from {file_path}: {e}", exc_info=True)

    def is_ime(self, term_id: int | str, term_name: str | None = None) -> bool:
        """
        Return True if term_id OR term_name is in the IME list.

        Lookup order:
          1. term_id (fast O(1) set lookup)
          2. term_name fallback (for cases where ID is not in IME list but name matches)
        """
        try:
            if int(term_id) in self.ime_ids:
                return True
        except (ValueError, TypeError):
            pass

        if term_name:
            return term_name.lower() in self.ime_names

        return False

    def get_ime_info(self, term_id: int | str) -> dict | None:
        try:
            return self.ime_details.get(int(term_id))
        except (ValueError, TypeError):
            return None

    def get_all_imes(self) -> list[dict]:
        return list(self.ime_details.values())

    def count(self) -> int:
        return self.total_loaded


# ── Singleton ────────────────────────────────────────────────────────────────

_ime_index: IMEIndex | None = None


def get_ime_index() -> IMEIndex:
    global _ime_index
    if _ime_index is None:
        _ime_index = IMEIndex()
    return _ime_index


def is_ime(term_id: int | str, term_name: str | None = None) -> bool:
    return get_ime_index().is_ime(term_id, term_name)


def ime_info(term_id: int | str) -> dict | None:
    return get_ime_index().get_ime_info(term_id)
