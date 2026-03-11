from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


BODY_COLUMN = "Body"
SENTIMENT_COLUMN = "Overall Sentiment"
EMOTION_COLUMN = "Emotion"

OUTPUT_DIR = "donor_analysis_outputs"

SHORT_TOKENS = {"ds", "di", "aid"}

CONTEXT_ANCHORS = [
    "donor",
    "sperm",
    "fertility",
    "ivf",
    "iui",
    "insemination",
    "reproduction",
    "cryobank",
]

SENTIMENT_VARIANTS = {
    "positive": {"positive", "pos", "1", "1.0"},
    "neutral": {"neutral", "neu", "0", "0.0"},
    "negative": {"negative", "neg", "-1", "-1.0"},
}


DONOR_LIST_1 = [
    "donor sperm",
    "sperm donor",
    "donor insemination",
    "donor conceived",
    "donor conception",
    "donor sample",
    "donor samples",
    "donor vials",
    "donor material",
    "donor gametes",
    "third party reproduction",
    "third-party reproduction",
    "ds",
    "di",
    "aid",
    "ivf-ds",
    "ivf ds",
]

DONOR_LIST_2 = [
    "sperm bank",
    "cryobank",
    "fertility bank",
    "donor bank",
    "sperm storage",
    "sperm freezing",
    "purchased sperm",
    "ordered sperm",
    "donor sperm bank",
    "california cryobank",
    "fairfax cryobank",
    "xytex",
    "seattle sperm bank",
    "european sperm bank",
]

DONOR_LIST_3 = [
    "iui with donor",
    "ivf with donor sperm",
    "donor iui",
    "donor ivf",
    "artificial insemination by donor",
    "third party ivf",
    "third-party ivf",
    "assisted reproduction with donor",
    "aid",
    "di",
    "ivf-ds",
    "ivf ds",
]

KEYWORD_SETS = {
    "list1": DONOR_LIST_1,
    "list2": DONOR_LIST_2,
    "list3": DONOR_LIST_3,
}


def infer_row_type_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "comment" in name:
        return "comment"
    if "post" in name:
        return "post"
    raise ValueError(f"Cannot infer post/comment from filename: {path.name}")


def normalise_sentiment(value: str) -> str | None:
    cleaned = value.strip().lower()
    for canonical, variants in SENTIMENT_VARIANTS.items():
        if cleaned in variants:
            return canonical
    return None


def compile_patterns(keywords: List[str]) -> Dict[str, re.Pattern]:
    patterns = {}
    for kw in keywords:
        if kw in SHORT_TOKENS:
            patterns[kw] = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
        else:
            patterns[kw] = re.compile(re.escape(kw), re.IGNORECASE)
    return patterns


def has_context(text: str) -> bool:
    return any(anchor in text for anchor in CONTEXT_ANCHORS)


def row_matches(text: str, patterns: Dict[str, re.Pattern]) -> bool:
    for kw, pattern in patterns.items():
        if pattern.search(text):
            if kw in SHORT_TOKENS:
                if has_context(text):
                    return True
            else:
                return True
    return False


def process_csvs(csv_paths: Iterable[Path]) -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    compiled = {
        name: compile_patterns(keywords)
        for name, keywords in KEYWORD_SETS.items()
    }

    extracted = {
        name: {"post": [], "comment": []}
        for name in KEYWORD_SETS
    }

    aggregated = {
        name: {
            "post": {"total": 0, "sentiments": {"positive": 0, "neutral": 0, "negative": 0}, "emotions": {}},
            "comment": {"total": 0, "sentiments": {"positive": 0, "neutral": 0, "negative": 0}, "emotions": {}},
        }
        for name in KEYWORD_SETS
    }

    for csv_path in csv_paths:
        row_type = infer_row_type_from_filename(csv_path)
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            body = str(row.get(BODY_COLUMN, "")).lower()
            sentiment_raw = str(row.get(SENTIMENT_COLUMN, ""))
            emotion_raw = str(row.get(EMOTION_COLUMN, "")).strip().lower()

            for list_name, patterns in compiled.items():
                if not row_matches(body, patterns):
                    continue

                extracted[list_name][row_type].append(row)
                aggregated[list_name][row_type]["total"] += 1

                sentiment = normalise_sentiment(sentiment_raw)
                if sentiment:
                    aggregated[list_name][row_type]["sentiments"][sentiment] += 1

                if emotion_raw:
                    bucket = aggregated[list_name][row_type]["emotions"]
                    bucket[emotion_raw] = bucket.get(emotion_raw, 0) + 1

    for list_name, buckets in extracted.items():
        for row_type, rows in buckets.items():
            out_path = output_dir / f"{list_name}_{row_type}s.csv"
            pd.DataFrame(rows).to_csv(out_path, index=False)

    sentiment_rows = []
    for list_name, types in aggregated.items():
        for row_type, stats in types.items():
            sentiment_rows.append({
                "List": list_name,
                "Type": row_type,
                "Count": stats["total"],
                "Positive": stats["sentiments"]["positive"],
                "Neutral": stats["sentiments"]["neutral"],
                "Negative": stats["sentiments"]["negative"],
            })

    pd.DataFrame(sentiment_rows).to_csv(
        output_dir / "donor_sentiment_summary.csv", index=False
    )

    all_emotions = sorted(
        {emotion for lists in aggregated.values()
         for types in lists.values()
         for emotion in types["emotions"]}
    )

    emotion_rows = []
    for list_name, types in aggregated.items():
        for row_type, stats in types.items():
            row = {"List": list_name, "Type": row_type}
            for emotion in all_emotions:
                row[emotion] = stats["emotions"].get(emotion, 0)
            emotion_rows.append(row)

    pd.DataFrame(emotion_rows).to_csv(
        output_dir / "donor_emotion_summary.csv", index=False
    )


def main(argv: Sequence[str]) -> None:
    if len(argv) < 2:
        raise SystemExit("Usage: python analyse_donor.py <csv_path> [<csv_path> ...]")

    csv_paths = [Path(arg) for arg in argv[1:]]
    process_csvs(csv_paths)


if __name__ == "__main__":
    main(sys.argv)
