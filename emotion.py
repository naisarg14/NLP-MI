import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from helpers import backup
import os

class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}


model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
trainer = Trainer(model=model)


import sys
file_list = sys.argv[1:]

folder = None

for f in file_list:
    print(f"Predicting Emotions for {f}")
    df_pred = pd.read_csv(f"{f}", encoding='utf-8')
    pred_texts = df_pred['Body'].dropna().astype('str').tolist()

    tokenized_texts = tokenizer(pred_texts, truncation=True, padding=True)
    pred_dataset = SimpleDataset(tokenized_texts)

    predictions = trainer.predict(pred_dataset)

    preds = predictions.predictions.argmax(-1)
    labels = pd.Series(preds).map(model.config.id2label)
    scores = (np.exp(predictions.predictions) / np.exp(predictions.predictions).sum(-1, keepdims=True)).max(1)

    temp = np.exp(predictions.predictions) / np.exp(predictions.predictions).sum(-1, keepdims=True)

    emotions = {emotion: [temp[i][j] for i in range(len(pred_texts))] for j, emotion in enumerate(['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'])}

    new_df = pd.DataFrame({
        'text': pred_texts,
        'Numerical_Emotion': preds,
        'Emotion': labels,
        'score': scores,
        **emotions
    })

    df_pred = df_pred.merge(new_df, left_on='Body', right_on='text', how='left')
    df_pred = df_pred.drop_duplicates()
    df_pred.drop(columns=['text'], inplace=True)
    outfile = f"{f.removesuffix('.csv')}_emotion.csv"
    if folder is not None:
        outfile = os.path.join(folder, os.path.basename(outfile))
    backup(outfile)
    df_pred.to_csv(outfile, index=False)
    print(f"Emotions saved to {outfile}")