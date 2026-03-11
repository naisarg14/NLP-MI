from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd


EXCEL_SOURCE = Path("MaleInfertility_Test_MK290925.xlsx")
BODY_COLUMN = "Body"
SENTIMENT_COLUMN = "Overall Sentiment"
EMOTION_COLUMN = "Emotion"
OUTPUT_FILENAME = "test_sentiment.csv"
EMOTION_OUTPUT_FILENAME = "test_emotion.csv"
OUTPUT_DIR = "test_analysis_outputs"

SENTIMENT_VARIANTS = {
	"positive": {"positive", "pos", "1", "1.0"},
	"neutral": {"neutral", "neu", "0", "0.0"},
	"negative": {"negative", "neg", "-1", "-1.0"},
}


def load_test_keyword_mapping(excel_path: Path | str = EXCEL_SOURCE) -> Dict[str, str]:
	"""Return a mapping of test name to typical Reddit keywords from the Excel workbook."""

	source_path = Path(excel_path)
	df = pd.read_excel(source_path, sheet_name=0, engine="openpyxl")

	required_columns = {"Test", "Typical Reddit Keywords"}
	missing_columns = required_columns.difference(df.columns)
	if missing_columns:
		raise ValueError(
			f"Missing expected columns in {source_path}: {', '.join(sorted(missing_columns))}"
		)

	filtered = df[["Test", "Typical Reddit Keywords"]].dropna(subset=["Test"])
	return dict(
		zip(
			filtered["Test"].astype(str).str.strip(),
			filtered["Typical Reddit Keywords"].astype(str).str.strip(),
		)
	)


def _parse_keywords(keyword_string: str) -> List[str]:
	"""Split the typical keyword string into individual keywords."""

	if not keyword_string:
		return []

	fragments = [frag.strip() for frag in keyword_string.split(",")]
	keywords: List[str] = []
	for fragment in fragments:
		if not fragment:
			continue

		sub_fragments = [sub.strip().lower() for sub in fragment.split(";") if sub.strip()]
		keywords.extend(sub_fragments)
	return keywords


def _normalise_sentiment(value: str) -> str | None:
	"""Map sentiment strings to the canonical buckets."""

	cleaned = value.strip().lower()
	for canonical, variants in SENTIMENT_VARIANTS.items():
		if cleaned in variants:
			return canonical
	return None


def count_keywords_in_csv(
	csv_paths: Iterable[Path], mapping: Mapping[str, str]
) -> Dict[str, Dict[str, Any]]:
	"""Aggregate keyword counts across one or more CSV files."""

	keyword_mapping = {test: _parse_keywords(keywords) for test, keywords in mapping.items()}
	aggregated_counts = {
		test: {"total": 0, "sentiments": {"positive": 0, "neutral": 0, "negative": 0}, "emotions": {}}
		for test in keyword_mapping
	}

	for csv_path in csv_paths:
		df = pd.read_csv(csv_path)
		required_columns = {BODY_COLUMN, SENTIMENT_COLUMN, EMOTION_COLUMN}
		missing_columns = required_columns.difference(df.columns)
		if missing_columns:
			raise ValueError(
				f"Missing expected columns in {csv_path}: {', '.join(sorted(missing_columns))}"
			)

		body_series = df[BODY_COLUMN].fillna(" ").astype(str).str.lower()
		sentiment_series = df[SENTIMENT_COLUMN].fillna("").astype(str)
		emotion_series = df[EMOTION_COLUMN].fillna("").astype(str)

		for test, keywords in keyword_mapping.items():
			if not keywords:
				continue

			match_mask = pd.Series(False, index=body_series.index, dtype=bool)
			for keyword in keywords:
				match_mask |= body_series.str.contains(keyword, case=False, na=False, regex=False)

			matches = match_mask.sum()
			if matches == 0:
				continue

			aggregated_counts[test]["total"] += int(matches)

			matched_sentiments = sentiment_series[match_mask]
			for sentiment in matched_sentiments:
				canonical = _normalise_sentiment(sentiment)
				if canonical:
					aggregated_counts[test]["sentiments"][canonical] += 1

			matched_emotions = emotion_series[match_mask].str.strip().str.lower()
			emotion_bucket = aggregated_counts[test]["emotions"]
			for emotion in matched_emotions:
				if not emotion:
					continue
				emotion_bucket[emotion] = emotion_bucket.get(emotion, 0) + 1

	return aggregated_counts


def main(argv: Sequence[str]) -> None:
	if len(argv) < 2:
		raise SystemExit("Usage: python analyse_test.py <csv_path> [<csv_path> ...]")

	csv_paths = [Path(arg) for arg in argv[1:]]
	missing_paths = [str(path) for path in csv_paths if not path.exists()]
	if missing_paths:
		raise FileNotFoundError(f"CSV files not found: {', '.join(missing_paths)}")

	mapping = load_test_keyword_mapping()
	aggregated = count_keywords_in_csv(csv_paths, mapping)

	first_csv = csv_paths[0]
	base_name = first_csv.stem
	base_name_clean = base_name.replace("_sentiment_emotion", "")

	output_dir = first_csv.parent / OUTPUT_DIR
	output_dir.mkdir(parents=True, exist_ok=True)

	output_path = output_dir / f"{base_name_clean}_{OUTPUT_FILENAME}"
	sentiment_rows = []
	for test, stats in aggregated.items():
		sentiment_rows.append(
			{
				"Test": test,
				"Count": stats["total"],
				"Positive": stats["sentiments"].get("positive", 0),
				"Neutral": stats["sentiments"].get("neutral", 0),
				"Negative": stats["sentiments"].get("negative", 0),
			}
		)

	sentiment_df = pd.DataFrame(sentiment_rows)
	sentiment_df.to_csv(output_path, index=False)

	emotion_output_path = output_dir / f"{base_name_clean}_{EMOTION_OUTPUT_FILENAME}"
	all_emotions = sorted({emotion for stats in aggregated.values() for emotion in stats["emotions"]})
	emotion_rows = []
	for test, stats in aggregated.items():
		row = {"Test": test}
		for emotion in all_emotions:
			row[emotion] = stats["emotions"].get(emotion, 0)
		emotion_rows.append(row)

	emotion_df = pd.DataFrame(emotion_rows)
	if emotion_rows:
		emotion_df = emotion_df[["Test", *all_emotions]]
	emotion_df.to_csv(emotion_output_path, index=False)

	print(f"Sentiment counts saved to {output_path}")
	print(f"Emotion counts saved to {emotion_output_path}")


if __name__ == "__main__":
	main(sys.argv)
