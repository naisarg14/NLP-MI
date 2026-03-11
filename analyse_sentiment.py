import csv
from helpers import backup
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from process_paragraph import process_paragraph


def file_sentiment(input_file):
    out_file = input_file.replace('.csv', '_sentiment.csv')
    backup(out_file)
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    sia = SentimentIntensityAnalyzer()

    with open(input_file, 'r',  encoding='utf-8') as file:
        with open(out_file, 'w', encoding='utf-8') as output_file:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames + ["Corrected Paragraph", 'Compound Score', 'Overall Sentiment', "Numerical Sentiment"]
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            total_rows = sum(1 for _ in reader)
            file.seek(0)
            next(reader)
            for row in tqdm(reader, total=total_rows, desc=f"Processing {input_file}"):
                if 'Title' in row:
                    paragraph = (row['Title'] + row['Body']).lower().strip()
                else:
                    paragraph = row['Body'].lower().strip()

                if not paragraph or paragraph == '[deleted]' or paragraph == '[removed]':
                    continue
                corrected_paragraph = process_paragraph(paragraph)

                sentiment = sia.polarity_scores(corrected_paragraph)

                if sentiment['compound'] >= 0.05:
                    positive_count += 1
                    overall = "Positive"
                    num = 1

                elif sentiment['compound'] <= -0.05:
                    overall = "Negative"
                    negative_count += 1
                    num = -1

                else:
                    overall = "Neutral"
                    neutral_count += 1
                    num = 0
                
                row["Corrected Paragraph"] = corrected_paragraph
                row['Compound Score'] = sentiment['compound']
                row['Overall Sentiment'] = overall
                row['Numerical Sentiment'] = num

                writer.writerow(row)
    total = positive_count + negative_count + neutral_count
    print(f"Positive: {positive_count}")
    print(f"Negative: {negative_count}")
    print(f"Neutral: {neutral_count}")
    print(f"Total: {total}\n")
    print(f"Positive: {positive_count*100/total:.2f}")
    print(f"Negative: {negative_count*100/total:.2f}")

    return out_file

def main():
    import sys
    files = sys.argv[1:]

    for file in files:
        print(f"Sentiment Analysis for {file}")
        out_file = file_sentiment(file)
        print(f"Completed Sentiment Analysis and saved at {out_file}\n")

if __name__ == '__main__':
    main()
