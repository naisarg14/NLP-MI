from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


COMMENTS_FILENAME = Path("extracted/r_maleinfertility_comments_sentiment_emotion.csv")
POSTS_FILENAME = Path("extracted/r_maleinfertility_posts_sentiment_emotion.csv")

USER_COLUMN = "Author"
DATE_COLUMN = "Date"
TYPE_COLUMN = "_SourceType"

COMMENTS_OUTPUT_FILENAME = "user_comments_yearly_activity.csv"
POSTS_OUTPUT_FILENAME = "user_posts_yearly_activity.csv"
TOTAL_OUTPUT_FILENAME = "user_total_yearly_activity.csv"
OUTPUT_DIR = Path("user_activity_outputs")
MIN_ACTIVE_MONTHS = 10  # Minimum distinct months a user must be active to be included
ACTIVE_MONTH_DISTRIBUTION_FILENAME = "user_active_months_distribution.csv"


def _load_dataset(path: Path, source_type: str) -> pd.DataFrame:
	df = pd.read_csv(path)

	required_columns = {USER_COLUMN, DATE_COLUMN}
	missing_columns = required_columns.difference(df.columns)
	if missing_columns:
		raise ValueError(
			f"Missing required columns in {path}: {', '.join(sorted(missing_columns))}"
		)

	df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
	df = df.dropna(subset=[USER_COLUMN, DATE_COLUMN]).copy()
	df[TYPE_COLUMN] = source_type

	return df[[USER_COLUMN, DATE_COLUMN, TYPE_COLUMN]]


def _aggregate_yearly_counts(
	comments_df: pd.DataFrame, posts_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	def _pivot(df: pd.DataFrame, label: str) -> pd.DataFrame:
		counts = (
			df.groupby([USER_COLUMN, df[DATE_COLUMN].dt.year]).size().reset_index(name="Count")
		)
		counts.rename(columns={DATE_COLUMN: "Year"}, inplace=True)
		counts[label] = counts.pop("Count")
		pivot = counts.pivot_table(
			index=USER_COLUMN,
			columns="Year",
			values=label,
			aggfunc="sum",
			fill_value=0,
		)
		pivot.columns = [f"{label.lower()}_{int(year)}" for year in pivot.columns]
		pivot.sort_index(axis=1, inplace=True)
		return pivot

	comments_pivot = _pivot(comments_df, "Comments")
	comments_pivot["total_comments"] = comments_pivot.sum(axis=1)

	posts_pivot = _pivot(posts_df, "Posts")
	posts_pivot["total_posts"] = posts_pivot.sum(axis=1)

	total_pivot = comments_pivot.join(posts_pivot, how="outer").fillna(0)
	comment_year_columns = [col for col in total_pivot.columns if col.startswith("comments_")]
	post_year_columns = [col for col in total_pivot.columns if col.startswith("posts_")]
	all_years = sorted({int(col.split("_")[-1]) for col in comment_year_columns + post_year_columns})

	for year in all_years:
		comment_series = total_pivot.get(
			f"comments_{year}", pd.Series(0, index=total_pivot.index)
		)
		post_series = total_pivot.get(
			f"posts_{year}", pd.Series(0, index=total_pivot.index)
		)
		total_pivot[f"total_{year}"] = comment_series + post_series

	total_pivot = total_pivot.drop(columns=comment_year_columns + post_year_columns, errors="ignore")
	total_year_columns = [f"total_{year}" for year in all_years]
	total_pivot = total_pivot[[*total_year_columns, "total_comments", "total_posts"]]
	total_pivot["grand_total"] = total_pivot["total_comments"] + total_pivot["total_posts"]

	return (
		comments_pivot.reset_index(),
		posts_pivot.reset_index(),
		total_pivot.reset_index(),
	)


def summarise_user_activity(
	comments_path: Path, posts_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	comments_df = _load_dataset(comments_path, "Comments")
	posts_df = _load_dataset(posts_path, "Posts")

	combined = pd.concat([comments_df, posts_df], ignore_index=True)
	combined["YearMonth"] = combined[DATE_COLUMN].dt.to_period("M")
	active_month_counts = combined.groupby(USER_COLUMN)["YearMonth"].nunique()

	if MIN_ACTIVE_MONTHS > 1:
		active_users = active_month_counts[active_month_counts >= MIN_ACTIVE_MONTHS].index
		comments_df = comments_df[comments_df[USER_COLUMN].isin(active_users)]
		posts_df = posts_df[posts_df[USER_COLUMN].isin(active_users)]

	distribution_df = (
		active_month_counts.value_counts().sort_index().reset_index()
	)
	distribution_df.columns = ["MonthsActive", "UserCount"]

	return (*_aggregate_yearly_counts(comments_df, posts_df), distribution_df)


def main(argv: Sequence[str]) -> None:
	if len(argv) == 1:
		comments_path = COMMENTS_FILENAME
		posts_path = POSTS_FILENAME
	elif len(argv) == 3:
		comments_path = Path(argv[1])
		posts_path = Path(argv[2])
	else:
		raise SystemExit(
			"Usage: python track_users.py <comments_csv> <posts_csv>\n"
			"Provide no arguments to use default extracted files."
		)

	for path in (comments_path, posts_path):
		if not Path(path).exists():
			raise FileNotFoundError(f"File not found: {path}")

	(
		comments_summary,
		posts_summary,
		total_summary,
		distribution_summary,
	) = summarise_user_activity(Path(comments_path), Path(posts_path))

	output_base = Path(comments_path).parent / OUTPUT_DIR
	output_base.mkdir(parents=True, exist_ok=True)

	comments_output = output_base / COMMENTS_OUTPUT_FILENAME
	posts_output = output_base / POSTS_OUTPUT_FILENAME
	total_output = output_base / TOTAL_OUTPUT_FILENAME
	distribution_output = output_base / ACTIVE_MONTH_DISTRIBUTION_FILENAME

	comments_summary.to_csv(comments_output, index=False)
	posts_summary.to_csv(posts_output, index=False)
	total_summary.to_csv(total_output, index=False)
	distribution_summary.to_csv(distribution_output, index=False)

	print(f"Comments activity summary written to {comments_output}")
	print(f"Posts activity summary written to {posts_output}")
	print(f"Overall activity summary written to {total_output}")
	print(f"Active months distribution written to {distribution_output}")


if __name__ == "__main__":
	main(sys.argv)
