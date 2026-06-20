import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10  # беремо ширше, щоб RRF міг переранжувати

df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

# 1. Побудувати локальний BM25-індекс за заголовками і анотаціями всіх статей.
corpus = df["title"].fillna("") + " " + df["abstract"].fillna("")
bm25 = BM25Okapi([doc.lower().split() for doc in corpus])

# 2. Підключитися до Pinecone і використовувати модель allenai/specter2_base для векторного пошуку.
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)


# 3. Реалізувати Reciprocal Rank Fusion (RRF) для об’єднання ранжованих списків BM25 і векторного пошуку:
# - об’єднувати результати двох методів;
# - формувати загальний топ-K документів.
def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """
    Об'єднує кілька ранжованих списків через RRF.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    query: str,
    corpus: list[str],
    model: SentenceTransformer,
    bm25: BM25Okapi,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    # --- Векторний пошук ---
    corpus_embeddings = model.encode(corpus, normalize_embeddings=True)
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    vector_scores = np.dot(corpus_embeddings, query_embedding)
    vector_ranking = list(np.argsort(vector_scores)[::-1])

    # --- BM25 пошук ---
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranking = list(np.argsort(bm25_scores)[::-1])

    # --- RRF об'єднання ---
    fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking])

    return [(corpus[doc_id], score) for doc_id, score in fused[:top_k]]


# 4. Реалізувати функції пошуку:
# - BM25;
def search_bm25(
    query: str, bm25: BM25Okapi, top_k: int = TOP_K
) -> list[tuple[int, float]]:
    scores = bm25.get_scores(query.lower().split())
    ranking = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in ranking]


# - векторний (Pinecone);
def search_vector(
    query: str, model: SentenceTransformer, index, top_k: int = TOP_K
) -> list[tuple[int, float]]:
    query_embedding = model.encode(query, normalize_embeddings=True)
    results = index.query(vector=np.asarray(query_embedding).tolist(), top_k=top_k)
    # id у Pinecone мають вигляд "paper_<номер_рядка_в_df>" (див. 03_load_to_pinecone.py)
    return [(int(match.id.split("_")[1]), match.score) for match in results.matches]


# - гібридний (BM25 + векторний через RRF).
def search_hybrid(
    query: str,
    bm25: BM25Okapi,
    model: SentenceTransformer,
    index,
    top_k: int = TOP_K,
) -> list[tuple[int, float]]:
    bm25_ranking = [doc_id for doc_id, _ in search_bm25(query, bm25, top_k=top_k)]
    vector_ranking = [
        doc_id for doc_id, _ in search_vector(query, model, index, top_k=top_k)
    ]
    fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking])
    return fused[:top_k]


# 5. Для демонстрації виконати три запити:
TEST_QUERIES = [
    # - точний термін ("BERT fine-tuning");
    "BERT fine-tuning",
    # - ім’я автора ("Yann LeCun convolutional networks");
    "Yann LeCun convolutional networks",
    # - перефразування без явних термінів ("making computers understand human emotions from text").
    "making computers understand human emotions from text",
]

DISPLAY_TOP_K = 5


# 6. Вивести результати для кожного методу і порівняти:
def print_results(label: str, results: list[tuple[int, float]]):
    print(f"{label}:")
    for rank, (doc_id, score) in enumerate(results, start=1):
        title = df.iloc[doc_id]["title"]
        print(f"  {rank}. [{score:.4f}] {title}")
    print()


for query in TEST_QUERIES:
    print(f"=== Запит: '{query}' ===\n")

    # - топ-5 BM25;
    print_results("BM25", search_bm25(query, bm25, top_k=DISPLAY_TOP_K))
    # - топ-5 векторного пошуку;
    print_results(
        "Векторний пошук (Pinecone)",
        search_vector(query, model, index, top_k=DISPLAY_TOP_K),
    )
    # - топ-5 гібридного пошуку з RRF, включаючи RRF-скор.
    print_results(
        "Гібридний пошук (RRF)",
        search_hybrid(query, bm25, model, index, top_k=DISPLAY_TOP_K),
    )

    print(
        "Порівняння: топ-5 BM25 і векторного пошуку майже не перетинаються - "
        "BM25 шукає буквальний збіг слів, вектор - схожість за змістом. "
        "Документи, що потрапляють в топ BM25 або Pinecone, отримують найвищий RRF-скор "
        "у гібридному результаті."
    )
