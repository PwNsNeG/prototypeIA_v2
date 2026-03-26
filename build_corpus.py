#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
from tqdm import tqdm


TEXT_COL = "complete_text"
META_COLS = ["file_id", "title", "date", "author"]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x0c", "\n")  # form feed/page breaks
    text = text.replace("\u00a0", " ")  # nbsp
    text = text.replace("\t", " ")

    # Strip trailing spaces line by line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse giant blank zones
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse long space runs
    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()


def looks_like_header_footer(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) <= 3 and any(ch.isdigit() for ch in s):
        return True
    if re.fullmatch(r"[IVXLCDM]+\.?", s):
        return True
    if re.fullmatch(r"[-_=*·•~ ]{4,}", s):
        return True
    if re.fullmatch(r"\d+", s):
        return True
    return False


def clean_for_model(text: str) -> str:
    text = normalize_text(text)
    raw_lines = text.split("\n")

    kept = []
    for line in raw_lines:
        s = line.strip()

        # Drop obvious noise-only lines
        if not s:
            kept.append("")
            continue
        if looks_like_header_footer(s):
            continue

        # Drop lines that are mostly punctuation/symbol soup
        punct_like = sum(1 for ch in s if not ch.isalnum() and not ch.isspace())
        if len(s) >= 8 and punct_like / max(len(s), 1) > 0.55:
            continue

        kept.append(s)

    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def text_metrics(text: str) -> dict[str, Any]:
    n = len(text)
    if n == 0:
        return {
            "chars": 0,
            "alpha_ratio": 0.0,
            "digit_ratio": 0.0,
            "space_ratio": 0.0,
            "upper_ratio": 0.0,
            "weird_ratio": 0.0,
            "short_line_ratio": 1.0,
            "headerish_ratio": 1.0,
            "repeated_line_ratio": 0.0,
            "blank_line_ratio": 1.0,
            "line_count": 0,
        }

    alpha = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    spaces = sum(ch.isspace() for ch in text)
    uppers = sum(ch.isupper() for ch in text if ch.isalpha())

    allowed_punct = set(".,;:!?()[]{}'\"-–—_/\\«»…")
    weird = 0
    for ch in text:
        if ch.isalnum() or ch.isspace() or ch in allowed_punct:
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("P"):
            continue
        weird += 1

    lines = text.split("\n")
    non_empty = [ln.strip() for ln in lines if ln.strip()]
    short_lines = [ln for ln in non_empty if len(ln) < 4]
    headerish = [ln for ln in non_empty if looks_like_header_footer(ln)]

    repeated_line_ratio = 0.0
    if non_empty:
        counts = Counter(non_empty)
        repeated = sum(c for _, c in counts.items() if c > 1)
        repeated_line_ratio = repeated / len(non_empty)

    blank_line_ratio = 1.0 - (len(non_empty) / max(len(lines), 1))

    return {
        "chars": n,
        "alpha_ratio": alpha / n,
        "digit_ratio": digits / n,
        "space_ratio": spaces / n,
        "upper_ratio": (uppers / alpha) if alpha else 0.0,
        "weird_ratio": weird / n,
        "short_line_ratio": len(short_lines) / max(len(non_empty), 1),
        "headerish_ratio": len(headerish) / max(len(non_empty), 1),
        "repeated_line_ratio": repeated_line_ratio,
        "blank_line_ratio": blank_line_ratio,
        "line_count": len(lines),
    }


def quality_score(m: dict[str, Any]) -> float:
    score = 0.0

    # Positive signals
    score += min(m["chars"] / 20000.0, 2.5)
    score += 3.0 * m["alpha_ratio"]

    # Negative signals
    score -= 8.0 * m["weird_ratio"]
    score -= 2.5 * m["digit_ratio"]
    score -= 2.0 * m["short_line_ratio"]
    score -= 3.0 * m["headerish_ratio"]
    score -= 2.0 * m["repeated_line_ratio"]

    if m["chars"] > 5000 and m["upper_ratio"] > 0.45:
        score -= 1.5

    if m["blank_line_ratio"] > 0.35:
        score -= 1.0

    return score


def reject_reason(m: dict[str, Any], min_chars: int, min_alpha_ratio: float, max_weird_ratio: float) -> str | None:
    if m["chars"] < min_chars:
        return "too_short"
    if m["alpha_ratio"] < min_alpha_ratio:
        return "low_alpha_ratio"
    if m["weird_ratio"] > max_weird_ratio:
        return "too_many_weird_chars"
    if m["short_line_ratio"] > 0.35:
        return "too_many_short_lines"
    if m["headerish_ratio"] > 0.20:
        return "too_many_headerish_lines"
    if m["repeated_line_ratio"] > 0.20:
        return "too_many_repeated_lines"
    return None


def stable_doc_id(file_id: str | None, title: str | None, author: str | None, idx: int) -> str:
    base = f"{file_id or ''}|{title or ''}|{author or ''}|{idx}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def assign_split(doc_id: str, val_ratio: float) -> str:
    h = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    return "val" if x < val_ratio else "train"


def safe_value(obj: Any) -> Any:
    return None if obj is None else obj


def iter_records(dataset_path: str, pattern: str, batch_size: int):
    dataset = ds.dataset(dataset_path, format="parquet")
    columns = [TEXT_COL] + META_COLS

    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        rows = batch.to_pylist()
        for row in rows:
            yield row


def main():
    parser = argparse.ArgumentParser(description="Build filtered corpus from Gallica parquet files.")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--glob", type=str, default="gallica_mono_*.parquet")
    parser.add_argument("--out-dir", type=str, default="prepared_corpus")
    parser.add_argument("--target-chars", type=int, default=500_000_000)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--min-chars", type=int, default=2000)
    parser.add_argument("--min-alpha-ratio", type=float, default=0.55)
    parser.add_argument("--max-weird-ratio", type=float, default=0.08)
    parser.add_argument("--min-score", type=float, default=1.25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--write-rejected", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    kept_report_path = out_dir / "kept_report.jsonl"
    rejected_report_path = out_dir / "rejected_report.jsonl"
    summary_path = out_dir / "summary.json"

    seen_hashes: set[str] = set()

    total_docs = 0
    kept_docs = 0
    rejected_docs = 0
    duplicate_docs = 0
    kept_chars = 0
    train_chars = 0
    val_chars = 0
    reject_counts = Counter()

    parquet_files = sorted(data_dir.glob(args.glob))
    if not parquet_files:
        raise SystemExit(f"No parquet files found matching {args.glob!r} in {data_dir}")

    dataset = ds.dataset([str(p) for p in parquet_files], format="parquet")

    with (
        train_path.open("w", encoding="utf-8") as f_train,
        val_path.open("w", encoding="utf-8") as f_val,
        kept_report_path.open("w", encoding="utf-8") as f_kept,
        (rejected_report_path.open("w", encoding="utf-8") if args.write_rejected else open("/dev/null", "w")) as f_rejected,
    ):
        pbar = tqdm(desc="Building corpus", unit="doc")

        for idx, row in enumerate(dataset.to_batches(columns=[TEXT_COL] + META_COLS, batch_size=args.batch_size)):
            for rec in row.to_pylist():
                total_docs += 1
                pbar.update(1)

                raw_text = rec.get(TEXT_COL)
                if raw_text is None or not isinstance(raw_text, str):
                    rejected_docs += 1
                    reject_counts["missing_text"] += 1
                    continue

                file_id = safe_value(rec.get("file_id"))
                title = safe_value(rec.get("title"))
                author = safe_value(rec.get("author"))
                date = safe_value(rec.get("date"))

                cleaned = clean_for_model(raw_text)
                metrics = text_metrics(cleaned)
                score = quality_score(metrics)

                reason = reject_reason(
                    metrics,
                    min_chars=args.min_chars,
                    min_alpha_ratio=args.min_alpha_ratio,
                    max_weird_ratio=args.max_weird_ratio,
                )
                if reason is None and score < args.min_score:
                    reason = "low_quality_score"

                if reason is not None:
                    rejected_docs += 1
                    reject_counts[reason] += 1
                    if args.write_rejected:
                        rej = {
                            "file_id": file_id,
                            "title": title,
                            "author": author,
                            "date": date,
                            "reason": reason,
                            "score": round(score, 4),
                            "metrics": metrics,
                            "sample": cleaned[:400],
                        }
                        f_rejected.write(json.dumps(rej, ensure_ascii=False) + "\n")
                    continue

                text_hash = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()
                if text_hash in seen_hashes:
                    duplicate_docs += 1
                    rejected_docs += 1
                    reject_counts["exact_duplicate"] += 1
                    continue
                seen_hashes.add(text_hash)

                doc_id = stable_doc_id(file_id, title, author, total_docs)
                split = assign_split(doc_id, args.val_ratio)

                record = {
                    "doc_id": doc_id,
                    "file_id": file_id,
                    "title": title,
                    "author": author,
                    "date": date,
                    "text": cleaned,
                    "chars": metrics["chars"],
                    "quality_score": round(score, 4),
                }

                if split == "val":
                    f_val.write(json.dumps(record, ensure_ascii=False) + "\n")
                    val_chars += metrics["chars"]
                else:
                    f_train.write(json.dumps(record, ensure_ascii=False) + "\n")
                    train_chars += metrics["chars"]

                f_kept.write(json.dumps({
                    "doc_id": doc_id,
                    "file_id": file_id,
                    "title": title,
                    "author": author,
                    "date": date,
                    "split": split,
                    "chars": metrics["chars"],
                    "score": round(score, 4),
                    "metrics": metrics,
                }, ensure_ascii=False) + "\n")

                kept_docs += 1
                kept_chars += metrics["chars"]

                pbar.set_postfix({
                    "kept_docs": kept_docs,
                    "kept_chars_M": round(kept_chars / 1_000_000, 1),
                    "rejected": rejected_docs,
                })

                if kept_chars >= args.target_chars:
                    pbar.close()
                    summary = {
                        "target_chars": args.target_chars,
                        "kept_chars": kept_chars,
                        "train_chars": train_chars,
                        "val_chars": val_chars,
                        "total_docs_seen": total_docs,
                        "kept_docs": kept_docs,
                        "rejected_docs": rejected_docs,
                        "duplicate_docs": duplicate_docs,
                        "reject_counts": dict(reject_counts),
                        "settings": {
                            "min_chars": args.min_chars,
                            "min_alpha_ratio": args.min_alpha_ratio,
                            "max_weird_ratio": args.max_weird_ratio,
                            "min_score": args.min_score,
                            "val_ratio": args.val_ratio,
                        },
                    }
                    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"\nReached target: {kept_chars:,} cleaned chars")
                    print(f"Wrote:\n  {train_path}\n  {val_path}\n  {kept_report_path}\n  {summary_path}")
                    return

        pbar.close()

    summary = {
        "target_chars": args.target_chars,
        "kept_chars": kept_chars,
        "train_chars": train_chars,
        "val_chars": val_chars,
        "total_docs_seen": total_docs,
        "kept_docs": kept_docs,
        "rejected_docs": rejected_docs,
        "duplicate_docs": duplicate_docs,
        "reject_counts": dict(reject_counts),
        "settings": {
            "min_chars": args.min_chars,
            "min_alpha_ratio": args.min_alpha_ratio,
            "max_weird_ratio": args.max_weird_ratio,
            "min_score": args.min_score,
            "val_ratio": args.val_ratio,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone. Did not reach target.")
    print(f"Kept chars: {kept_chars:,}")
    print(f"Wrote:\n  {train_path}\n  {val_path}\n  {kept_report_path}\n  {summary_path}")


if __name__ == "__main__":
    main()
