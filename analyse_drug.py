from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd


EXCEL_SOURCE = Path("Keywords.xlsx")
EXCEL_SHEET = "Drug Mention"
BODY_COLUMN = "Body"
SENTIMENT_COLUMN = "Overall Sentiment"
EMOTION_COLUMN = "Emotion"
OUTPUT_FILENAME = "drug_sentiment.csv"
EMOTION_OUTPUT_FILENAME = "drug_emotion.csv"
OUTPUT_DIR = "drug_analysis_outputs"

SENTIMENT_VARIANTS = {
    "positive": {"positive", "pos", "1", "1.0"},
    "neutral":  {"neutral",  "neu", "0", "0.0"},
    "negative": {"negative", "neg", "-1", "-1.0"},
}


def load_drug_keyword_mapping(excel_path: Path | str = EXCEL_SOURCE) -> Dict[str, str]:
    """Return a mapping of drug name to typical Reddit keywords from the Excel workbook."""

    source_path = Path(excel_path)
    df = pd.read_excel(source_path, sheet_name=EXCEL_SHEET, engine="openpyxl", header=2)

    required_columns = {"Drug", "Typical Reddit Keywords"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing expected columns in {source_path} (sheet '{EXCEL_SHEET}'): "
            f"{', '.join(sorted(missing_columns))}"
        )

    filtered = df[["Drug", "Typical Reddit Keywords"]].dropna(subset=["Drug"])
    filtered = filtered[filtered["Drug"].astype(str).str.strip() != ""]
    return dict(
        zip(
            filtered["Drug"].astype(str).str.strip(),
            filtered["Typical Reddit Keywords"].fillna("").astype(str).str.strip(),
        )
    )


def _parse_keywords(keyword_string: str) -> List[str]:
    """Split a comma/semicolon-separated keyword string into individual keywords."""

    if not keyword_string:
        return []
    keywords: List[str] = []
    for fragment in keyword_string.split(","):
        fragment = fragment.strip()
        if not fragment:
            continue
        for sub in fragment.split(";"):
            sub = sub.strip().lower()
            if sub:
                keywords.append(sub)
    return keywords


def _normalise_sentiment(value: str) -> str | None:
    cleaned = value.strip().lower()
    for canonical, variants in SENTIMENT_VARIANTS.items():
        if cleaned in variants:
            return canonical
    return None


def count_keywords_in_csv(
    csv_paths: Iterable[Path], mapping: Mapping[str, str]
) -> Dict[str, Dict[str, Any]]:
    """Aggregate keyword counts, sentiment, and emotion across one or more CSV files."""

    keyword_mapping = {drug: _parse_keywords(kws) for drug, kws in mapping.items()}
    aggregated_counts = {
        drug: {"total": 0, "sentiments": {"positive": 0, "neutral": 0, "negative": 0}, "emotions": {}}
        for drug in keyword_mapping
    }

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        required_columns = {BODY_COLUMN, SENTIMENT_COLUMN, EMOTION_COLUMN}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing expected columns in {csv_path}: {', '.join(sorted(missing_columns))}"
            )

        body_series = df[BODY_COLUMN].fillna("").astype(str).str.lower()
        sentiment_series = df[SENTIMENT_COLUMN].fillna("").astype(str)
        emotion_series = df[EMOTION_COLUMN].fillna("").astype(str)

        for drug, keywords in keyword_mapping.items():
            if not keywords:
                continue

            match_mask = pd.Series(False, index=body_series.index, dtype=bool)
            for keyword in keywords:
                match_mask |= body_series.str.contains(keyword, case=False, na=False, regex=False)

            matches = match_mask.sum()
            if matches == 0:
                continue

            aggregated_counts[drug]["total"] += int(matches)

            for sentiment in sentiment_series[match_mask]:
                canonical = _normalise_sentiment(sentiment)
                if canonical:
                    aggregated_counts[drug]["sentiments"][canonical] += 1

            emotion_bucket = aggregated_counts[drug]["emotions"]
            for emotion in emotion_series[match_mask].str.strip().str.lower():
                if emotion:
                    emotion_bucket[emotion] = emotion_bucket.get(emotion, 0) + 1

    return aggregated_counts


def main(argv: Sequence[str]) -> None:
    if len(argv) < 2:
        raise SystemExit("Usage: python analyse_drug.py <csv_path> [<csv_path> ...]")

    csv_paths = [Path(arg) for arg in argv[1:]]
    missing_paths = [str(p) for p in csv_paths if not p.exists()]
    if missing_paths:
        raise FileNotFoundError(f"CSV files not found: {', '.join(missing_paths)}")

    mapping = load_drug_keyword_mapping()
    aggregated = count_keywords_in_csv(csv_paths, mapping)

    first_csv = csv_paths[0]
    base_name_clean = first_csv.stem.replace("_sentiment_emotion", "")

    output_dir = first_csv.parent / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    sentiment_rows = [
        {
            "Drug": drug,
            "Count": stats["total"],
            "Positive": stats["sentiments"].get("positive", 0),
            "Neutral": stats["sentiments"].get("neutral", 0),
            "Negative": stats["sentiments"].get("negative", 0),
        }
        for drug, stats in aggregated.items()
    ]
    sentiment_path = output_dir / f"{base_name_clean}_{OUTPUT_FILENAME}"
    pd.DataFrame(sentiment_rows).to_csv(sentiment_path, index=False)

    all_emotions = sorted({e for stats in aggregated.values() for e in stats["emotions"]})
    emotion_rows = [
        {"Drug": drug, **{e: stats["emotions"].get(e, 0) for e in all_emotions}}
        for drug, stats in aggregated.items()
    ]
    emotion_df = pd.DataFrame(emotion_rows)
    if emotion_rows:
        emotion_df = emotion_df[["Drug", *all_emotions]]
    emotion_path = output_dir / f"{base_name_clean}_{EMOTION_OUTPUT_FILENAME}"
    emotion_df.to_csv(emotion_path, index=False)

    print(f"Sentiment counts saved to {sentiment_path}")
    print(f"Emotion counts saved to {emotion_path}")


if __name__ == "__main__":
    main()
