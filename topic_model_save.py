from bertopic import BERTopic
from tqdm import tqdm
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import sys



def main():
    files = sys.argv[1:]
    for file in tqdm(files):
        get_files_topic([file], file.removesuffix(".csv"))


def get_files_topic(files_list, name):
    print(f"Processing {name} files: {files_list}")
    for f in files_list:
        dataset = pd.read_csv(f).dropna(subset=["Body"])
        abstracts = dataset["Body"].tolist()
        model_topic(abstracts, name)


def model_topic(abstracts, f):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=30, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

    prompt = """
    I have a topic that contains the following documents:
    [DOCUMENTS]
    The topic is described by the following keywords: [KEYWORDS]

    Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make sure it is in the following format:
    topic: <topic label>
    """
    #client = openai.OpenAI(api_key="sk-...")
    #openai_model = OpenAI(client, model="gpt-3.5-turbo", exponential_backoff=True, chat=True, prompt=prompt)

    representation_model = KeyBERTInspired()
    topic_model = BERTopic(
        # Pipeline models
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        # Hyperparameters
        top_n_words=10,
        verbose=True
        )
    
    topics, probs = topic_model.fit_transform(abstracts, embeddings)
    topic_model.save(f"topic_models2/{f.replace('extracted/', '').removesuffix('.csv')}_model")
    length = len(topic_model.get_topics())
    with open(f"topic_models2/{f.replace('extracted/', '').removesuffix('.csv')}_model.txt", "a+") as log:
        for i in range(-1, length):
            topic_info = topic_model.get_topic_info()
            try:
                topic_i_count = topic_info[topic_info["Topic"] == i]["Count"].values[0]
            except IndexError:
                continue
            log.write(f"Topic {i+1} [Count: {topic_i_count}]: ")
            print(f"Topic {i+1}: ", end="")
            topic_data = topic_model.get_topic(i)
            items = [item[0] for item in topic_data]
            str_items = " | ".join(items)
            log.write(f"{str_items}\n")
            print(str_items)


if __name__ == "__main__":
    main()