"""Top-level package for AI Dataset Cleaner.

This package provides utilities for automatically cleaning messy CSV or JSON
datasets. It exposes a command line interface via ``python -m ai_dataset_cleaner.cli``
and programmatic access through the :class:`DatasetCleaner` class.

Example usage::

    from ai_dataset_cleaner.cleaner import DatasetCleaner
    cleaner = DatasetCleaner(expected_columns=["id", "name", "age"])
    df, report = cleaner.clean_file("data.csv", output_path="cleaned.csv")
    print(report)

The tool detects and fixes schema mismatches (missing or unexpected columns),
converts data types, fills in missing values, normalises categorical values,
removes duplicate records, and generates a summary report describing the
transformations performed.
"""

from .cleaner import DatasetCleaner  # noqa: F401

__all__ = ["DatasetCleaner"]