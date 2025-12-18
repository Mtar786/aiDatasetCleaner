"""Command line interface for AI Dataset Cleaner.

This module exposes an entry point that can be invoked via ``python -m ai_dataset_cleaner.cli``
or ``ai_dataset_cleaner.cli:main``. It wraps the :class:`DatasetCleaner` and provides
options for specifying the input dataset, expected schema file, output path and
cleaning behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from .cleaner import DatasetCleaner


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    required=True,
    help="Path to the input CSV or JSON file to clean.",
)
@click.option(
    "--schema",
    "schema_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    default=None,
    help="Optional JSON file defining expected column names. May contain a list of strings or a dict with a 'columns' key.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True, path_type=str),
    default=None,
    help="Optional path to write the cleaned dataset. The format is inferred from the file extension (.csv or .json).",
)
@click.option(
    "--no-drop-duplicates",
    "drop_duplicates",
    is_flag=True,
    default=False,
    help="If provided, duplicate rows will NOT be removed.",
)
@click.option(
    "--strict",
    "strict_schema",
    is_flag=True,
    default=False,
    help="If set, any unexpected columns not in the schema will be dropped. Otherwise they are retained.",
)
@click.option(
    "--preview",
    is_flag=True,
    default=False,
    help="If set, print a preview of the cleaned DataFrame (first five rows).",
)
def main(
    input_path: str,
    schema_path: Optional[str],
    output_path: Optional[str],
    drop_duplicates: bool,
    strict_schema: bool,
    preview: bool,
) -> None:
    """Clean a CSV or JSON dataset by detecting schema mismatches, filling in missing values,
    converting column types, normalising categories and dates, and optionally removing duplicates.

    The tool outputs a JSON report summarising the operations performed.
    """
    # drop_duplicates flag is inverted because the option is named --no-drop-duplicates
    drop = not drop_duplicates
    cleaner = DatasetCleaner(
        expected_columns=None,
        drop_duplicates=drop,
        strict_schema=strict_schema,
    )
    if schema_path:
        cleaner.load_schema(schema_path)
    # Derive a default output path if none provided
    out_path = output_path
    if not output_path:
        input_p = Path(input_path)
        ext = input_p.suffix
        out_path = str(input_p.with_name(f"{input_p.stem}_cleaned{ext}"))

    df, report = cleaner.clean_file(input_path, out_path)
    click.echo(json.dumps(report, indent=2))
    if preview:
        click.echo("\nPreview of cleaned dataset:")
        click.echo(df.head().to_string())


if __name__ == "__main__":  # pragma: no cover
    main()