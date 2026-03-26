#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

TEXT_CANDIDATES = [
    "complete_text",
    "text",
    "content",
    "body",
    "document_text",
    "full_text",
    "ocr_text",
    "ocr",
    "clean_text",
    "raw_text",
]

def detect_text_column(schema: pa.Schema, forced_col: Optional[str] = None) -> str:
    names = schema.names

    if forced_col:
        if forced_col not in names:
            raise ValueError(
                f"Requested text column '{forced_col}' not found. Available columns: {names}"
            )
        return forced_col

    for candidate in TEXT_CANDIDATES:
        if candidate in names:
            return candidate

    string_cols = []
    for field in schema:
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            string_cols.append(field.name)

    if len(string_cols) == 1:
        return string_cols[0]

    if not string_cols:
        raise ValueError(f"No string-like column found. Available columns: {names}")

    raise ValueError(
        f"Could not auto-detect text column. String-like columns found: {string_cols}. "
        f"Use --text-col to specify one."
    )


def process_parquet_file(path: Path, forced_col: Optional[str] = None, batch_size: int = 128):
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    text_col = detect_text_column(schema, forced_col=forced_col)

    total_rows = 0
    non_empty_docs = 0
    total_chars = 0
    unique_chars = set()
    sample_text = None

    for batch in pf.iter_batches(batch_size=batch_size, columns=[text_col]):
        arr = batch.column(text_col)
        total_rows += len(arr)

        for value in arr.to_pylist():
            if value is None:
                continue
            if not isinstance(value, str):
                value = str(value)

            stripped = value.strip()
            if not stripped:
                continue

            non_empty_docs += 1
            total_chars += len(value)
            unique_chars.update(value)

            if sample_text is None:
                sample_text = value[:300]

    return {
        "file": path.name,
        "text_col": text_col,
        "rows": total_rows,
        "non_empty_docs": non_empty_docs,
        "chars": total_chars,
        "char_tokens": total_chars,
        "bpe_estimate": total_chars // 4,
        "unique_chars": len(unique_chars),
        "sample": sample_text or "",
        "unique_chars_set": unique_chars,
    }


def main():
    parser = argparse.ArgumentParser(description="Sanity check all parquet files in a folder.")
    parser.add_argument(
        "--pattern",
        default="*.parquet",
        help="Glob pattern for parquet files (default: *.parquet)",
    )
    parser.add_argument(
        "--text-col",
        default=None,
        help="Force the text column name if auto-detection is wrong",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for parquet iteration (default: 128)",
    )
    args = parser.parse_args()

    files = sorted(Path(".").glob(args.pattern))
    if not files:
        print(f"No files found for pattern: {args.pattern}")
        return

    global_rows = 0
    global_non_empty_docs = 0
    global_chars = 0
    global_unique_chars = set()
    first_global_sample = None
    processed = 0
    failed = 0

    print(f"Found {len(files)} parquet file(s)\n")

    for path in files:
        try:
            stats = process_parquet_file(
                path,
                forced_col=args.text_col,
                batch_size=args.batch_size,
            )
            processed += 1

            global_rows += stats["rows"]
            global_non_empty_docs += stats["non_empty_docs"]
            global_chars += stats["chars"]
            global_unique_chars.update(stats["unique_chars_set"])
            if first_global_sample is None and stats["sample"]:
                first_global_sample = stats["sample"]

            print("=" * 72)
            print(f"File:            {stats['file']}")
            print(f"Text column:     {stats['text_col']}")
            print(f"Rows:            {stats['rows']:,}")
            print(f"Non-empty docs:  {stats['non_empty_docs']:,}")
            print(f"File size:       {stats['chars']:,} characters")
            print(f"~Char-level tok: {stats['char_tokens']:,} tokens")
            print(f"~BPE tokens:     {stats['bpe_estimate']:,} tokens")
            print(f"Unique chars:    {stats['unique_chars']}")
            if stats["sample"]:
                print(f"Sample:\n{stats['sample']}\n")
            else:
                print("Sample:\n<empty>\n")

        except Exception as e:
            failed += 1
            print("=" * 72)
            print(f"File: {path.name}")
            print(f"ERROR: {e}\n")

    print("=" * 72)
    print("GLOBAL SUMMARY")
    print(f"Files processed: {processed}")
    print(f"Files failed:    {failed}")
    print(f"Total rows:      {global_rows:,}")
    print(f"Non-empty docs:  {global_non_empty_docs:,}")
    print(f"Total chars:     {global_chars:,}")
    print(f"~Char tokens:    {global_chars:,}")
    print(f"~BPE tokens:     {global_chars // 4:,}")
    print(f"Unique chars:    {len(global_unique_chars)}")
    if first_global_sample:
        print(f"Sample:\n{first_global_sample}")


if __name__ == "__main__":
    main()
