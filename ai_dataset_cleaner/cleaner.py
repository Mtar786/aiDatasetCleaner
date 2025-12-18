"""Core cleaning utilities for AI Dataset Cleaner.

This module defines the :class:`DatasetCleaner` class which encapsulates the
logic for cleaning tabular data stored in CSV or JSON files. It can be used
programmatically or via the command line interface defined in :mod:`cli`.

The cleaning pipeline performs the following steps:

1. **Schema validation**: If a list of expected columns is provided, the
   cleaner identifies missing columns (added with NaN values) and unexpected
   columns (optionally dropped or kept). Missing or renamed columns break
   downstream pipelines and are a common source of errors【234369309510859†L78-L93】.

2. **Data type conversion**: Columns containing numeric strings are
   converted to numeric dtype and date-like strings are parsed into
   ``datetime64`` objects【234369309510859†L102-L117】.

3. **Missing value handling**: For numeric columns the median is used to
   impute missing values, while for non-numeric columns the mode is used.
   Identifying and handling missing values properly is critical to avoid
   skewing analyses【234369309510859†L120-L149】.

4. **Duplicate removal**: Duplicate records are detected and removed to
   prevent double‑counting and improve data quality【534126459358432†L905-L917】.

5. **Category normalisation**: Categorical values are standardised to
   canonical forms (e.g., ``"Male"`` instead of ``"male"`` or ``"M"``)
   and whitespace/capitalisation inconsistencies are stripped【234369309510859†L170-L177】.

6. **Date normalisation**: Date columns are converted to ISO format
   (``YYYY‑MM‑DD``) using pandas date parsing【234369309510859†L181-L186】.

The cleaner returns the cleaned :class:`pandas.DataFrame` along with a
summary report detailing the operations performed and counts of fixes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from dateutil.parser import ParserError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DatasetCleaner:
    """Encapsulates the logic for cleaning datasets.

    Parameters
    ----------
    expected_columns : Optional[Iterable[str]], optional
        A list or other iterable of expected column names. If provided,
        missing columns will be added (filled with NaN) and unexpected
        columns will be removed. If ``None`` (the default), schema
        validation is skipped.
    drop_duplicates : bool, default True
        Whether to remove duplicate rows.
    strict_schema : bool, default False
        If ``True``, unexpected columns are dropped; otherwise they are
        retained and included in the output. See Pandera documentation for
        discussion of strict schema enforcement【631586573974689†L610-L646】.
    """

    def __init__(
        self,
        expected_columns: Optional[Iterable[str]] = None,
        drop_duplicates: bool = True,
        strict_schema: bool = False,
    ) -> None:
        self.expected_columns = list(expected_columns) if expected_columns else None
        self.drop_duplicates = drop_duplicates
        self.strict_schema = strict_schema

    def clean_file(
        self,
        file_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """Load, clean and optionally save a dataset.

        Parameters
        ----------
        file_path : str or Path
            Path to the CSV or JSON file to clean.
        output_path : Optional[str or Path], optional
            If given, the cleaned dataset will be written to this path.

        Returns
        -------
        tuple
            A tuple ``(df, report)`` where ``df`` is the cleaned DataFrame
            and ``report`` summarises the cleaning operations performed.
        """
        df = self._load_data(file_path)
        logger.info("Loaded dataset with %d rows and %d columns", df.shape[0], df.shape[1])
        report: Dict[str, any] = {
            "rows_before": int(df.shape[0]),
            "columns_before": list(df.columns),
        }
        if self.expected_columns is not None:
            df, schema_report = self._apply_schema(df, self.expected_columns)
            report.update(schema_report)

        df, type_report = self._convert_types(df)
        report.update(type_report)

        df, missing_report = self._fill_missing(df)
        report.update(missing_report)

        df = self._standardise_categories(df)
        # record this but no counts returned
        report.setdefault("category_normalised", True)

        df = self._standardise_dates(df)
        report.setdefault("dates_normalised", True)

        if self.drop_duplicates:
            df, dup_report = self._remove_duplicates(df)
            report.update(dup_report)

        report["rows_after"] = int(df.shape[0])
        report["columns_after"] = list(df.columns)

        if output_path:
            self._write_data(df, output_path)
            report["output_path"] = str(output_path)
        return df, report

    @staticmethod
    def _load_data(file_path: str | Path) -> pd.DataFrame:
        """Load a CSV or JSON file into a DataFrame."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in {".json", ".js", ".ndjson"}:
            try:
                df = pd.read_json(path, lines=True)
            except ValueError:
                # Fallback: load generic JSON
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                df = pd.json_normalize(data)
        else:
            raise ValueError(f"Unsupported file type: {suffix}; only CSV and JSON are supported")
        return df

    @staticmethod
    def _write_data(df: pd.DataFrame, output_path: str | Path) -> None:
        """Write DataFrame to CSV or JSON based on file extension."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        suffix = out.suffix.lower()
        if suffix == ".csv":
            df.to_csv(out, index=False)
        elif suffix == ".json":
            # Write JSON in a line‑delimited format to preserve large datasets
            df.to_json(out, orient="records", lines=True)
        else:
            raise ValueError(f"Unsupported output file type: {suffix}; only CSV and JSON are supported")

    def _apply_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """Ensure that the DataFrame has exactly the expected columns.

        This method attempts to align the dataset's columns to the provided
        ``expected_columns``.  It performs the following steps:

        1. **Rename existing columns** if their normalised form (lowercase,
           alphanumeric only) matches the normalised form of an expected
           column.  This catches simple case differences such as ``"ID"``
           → ``"id"`` and applies basic synonym mappings for common terms
           like ``"years" → "age"`` or ``"pay" → "salary"``.
        2. **Add missing columns** with ``NaN`` values.
        3. **Drop unexpected columns** if ``strict_schema`` is ``True``;
           otherwise unexpected columns are retained but reported.
        4. **Reorder columns** so that expected columns appear first in
           the order provided, followed by any remaining columns.

        Returns the modified DataFrame along with a report summarising
        renames, added columns and unexpected columns.
        """
        report: Dict[str, any] = {"renamed_columns": {}}

        # Build normalised mappings for expected columns
        def normalise(name: str) -> str:
            return "".join(ch for ch in name.lower() if ch.isalnum())

        expected_norm = {normalise(col): col for col in expected_columns}
        # Define simple synonym mappings for common data terms
        synonyms = {
            "years": "age",
            "yrs": "age",
            "year": "age",
            "pay": "salary",
            "wage": "salary",
            "income": "salary",
            "gender": "sex",
            "sex": "gender",  # allow either mapping; whichever exists in expected
            "dob": "dateofbirth",
            "birthdate": "dateofbirth",
        }
        # Attempt to rename columns based on normalised names and synonyms
        new_columns = {}
        for col in list(df.columns):
            norm = normalise(col)
            target = None
            # Check direct normalised match
            if norm in expected_norm:
                target = expected_norm[norm]
            # Check synonym mapping (normalise the synonym key)
            elif norm in synonyms and normalise(synonyms[norm]) in expected_norm:
                target = expected_norm[normalise(synonyms[norm])]
            if target and target != col:
                # Avoid overwriting if target already present
                if target not in df.columns:
                    df = df.rename(columns={col: target})
                    report["renamed_columns"][col] = target
                else:
                    # Drop old column if both exist and names conflict
                    df = df.drop(columns=[col])
                    report.setdefault("dropped_conflicting_columns", []).append(col)

        current_columns = list(df.columns)
        missing = [col for col in expected_columns if col not in current_columns]
        unexpected = [col for col in current_columns if col not in expected_columns]

        if missing:
            for col in missing:
                df[col] = pd.NA
            report["missing_columns_added"] = missing
        else:
            report["missing_columns_added"] = []

        if unexpected:
            if self.strict_schema:
                df = df[[c for c in current_columns if c not in unexpected]]
                report["unexpected_columns_dropped"] = unexpected
            else:
                report["unexpected_columns"] = unexpected
        else:
            report["unexpected_columns"] = []

        # Re-order columns to match expected order and preserve others at the end
        reordered = [c for c in expected_columns if c in df.columns] + [c for c in df.columns if c not in expected_columns]
        df = df.reindex(columns=reordered)
        return df, report

    def _convert_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """Attempt to convert object columns to numeric or datetime types.

        Returns the modified DataFrame and a report dictionary indicating
        which columns were converted.
        """
        report: Dict[str, any] = {"converted_types": {}}
        for col in df.columns:
            series = df[col]
            # skip non-object columns
            if series.dtype != object:
                continue
            # Attempt numeric conversion. We first try coercing to numeric,
            # which will produce NaN for unparseable values. If at least
            # half of the non-null values are successfully converted, we
            # accept the conversion; otherwise we leave as object.
            try:
                numeric_converted = pd.to_numeric(series, errors="coerce")
            except Exception:
                numeric_converted = series
            if pd.api.types.is_numeric_dtype(numeric_converted):
                # Determine fraction of values converted
                original_non_null = series.notna().sum()
                converted_non_null = numeric_converted.notna().sum()
                if original_non_null > 0 and converted_non_null / original_non_null >= 0.5:
                    df[col] = numeric_converted
                    report["converted_types"].setdefault(col, []).append(str(df[col].dtype))
                    continue

            # Attempt datetime conversion. Use coerce and accept if at least half parse.
            try:
                datetime_converted = pd.to_datetime(series, errors="coerce")
            except Exception:
                datetime_converted = series
            if pd.api.types.is_datetime64_any_dtype(datetime_converted):
                original_non_null = series.notna().sum()
                converted_non_null = datetime_converted.notna().sum()
                if original_non_null > 0 and converted_non_null / original_non_null >= 0.5:
                    df[col] = datetime_converted
                    report["converted_types"].setdefault(col, []).append(str(df[col].dtype))
                    continue
        return df, report

    def _fill_missing(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """Fill missing values using simple heuristics.

        Numeric columns: filled with the median. Categorical/textual columns:
        filled with the mode. Returns the modified DataFrame and a report
        containing counts of missing values before and after.
        """
        report: Dict[str, any] = {"missing_values_before": {}, "missing_values_filled": {}}
        for col in df.columns:
            num_missing = df[col].isna().sum()
            if num_missing == 0:
                continue
            report["missing_values_before"][col] = int(num_missing)
            if pd.api.types.is_numeric_dtype(df[col]):
                # Use median; if all values are NaN it returns NaN
                median = df[col].median()
                df[col] = df[col].fillna(median)
            else:
                # Use mode; if multiple modes choose the first
                mode_series = df[col].mode(dropna=True)
                if not mode_series.empty:
                    fill_val = mode_series.iloc[0]
                else:
                    fill_val = pd.NA
                df[col] = df[col].fillna(fill_val)
            report["missing_values_filled"][col] = int(num_missing)
        return df, report

    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """Remove duplicate rows and report the number removed."""
        report: Dict[str, any] = {}
        duplicates = df.duplicated().sum()
        report["duplicates_removed"] = int(duplicates)
        if duplicates > 0:
            df = df.drop_duplicates().reset_index(drop=True)
        return df, report

    @staticmethod
    def _standardise_categories(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise categorical columns by stripping whitespace and title‑casing.

        Additionally attempts to unify common gender values (e.g. "m", "male"
        → "Male"). This operation does not return a report because it is
        non‑destructive and not easily quantifiable.
        """
        def normalise_value(val: any) -> any:
            if isinstance(val, str):
                stripped = val.strip()
                lower = stripped.lower()
                # common gender normalisation
                if lower in {"m", "male", "man"}:
                    return "Male"
                if lower in {"f", "female", "woman"}:
                    return "Female"
                if lower in {"n/a", "nan", "", "none", "unknown"}:
                    return pd.NA
                return stripped.title()
            return val

        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].apply(normalise_value)
        return df

    @staticmethod
    def _standardise_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Normalise date columns to ISO format (YYYY‑MM‑DD).

        This function looks for columns with ``datetime64`` dtype and
        converts them to date only (no time) to ensure consistency. It
        silently ignores columns that cannot be converted.
        """
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = df[col].dt.date.astype("string")
                except Exception:
                    # Fallback: leave as datetime
                    pass
        return df

    def load_schema(self, schema_path: str | Path) -> List[str]:
        """Load a JSON schema file defining expected columns.

        The schema file should contain either a list of column names or a
        dictionary with a ``"columns"`` key mapping to a list of names. This
        method sets the ``expected_columns`` attribute on the instance.
        """
        path = Path(schema_path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "columns" in data:
            cols = data["columns"]
        elif isinstance(data, list):
            cols = data
        else:
            raise ValueError("Schema file must contain a list or a dict with 'columns' key")
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise ValueError("Schema columns must be a list of strings")
        self.expected_columns = cols
        return cols
