import os
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, QueryResponse
from sentence_transformers import SentenceTransformer, util

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# 1. Підключитися до індексу arxiv-papers у Pinecone і завантажити модель allenai/specter2_base.
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")  # для отримання повного abstract


# 2. Реалізувати функцію кодування запиту в ембеддинг.
def get_embedding(text: str) -> list[float]:
    embedding = model.encode(text, convert_to_numpy=True)
    return np.asarray(embedding).tolist()


# 3. Виконати чистий семантичний пошук:
# - задати запит (наприклад: "teaching machines to recognize objects in pictures");
query = "teaching machines to recognize objects in pictures"
# - отримати топ-5 найбільш релевантних статей;
results = index.query(
    vector=get_embedding(query),
    top_k=TOP_K,
    include_metadata=True,
)
# - вивести результати з назвою, категорією, роком і частиною абстракту.
print(f"Запит: '{query}'\n")


# Для цього та наступних пунктів створимо функцію тут
def print_results(in_results: QueryResponse) -> None:
    for match in in_results.matches:
        print(f"ID: {match.id} | Score: {match.score:.4f}")
        print(f"  Title: {match.metadata['title']}")
        print(f"  Category: {match.metadata['category']}")
        print(f"  Year: {match.metadata['year']}")
        print(f"  Abstract: {match.metadata['abstract']}...")
        print()


print_results(results)

# 4. Виконати пошук з фільтрацією:
# - приклад A: статті по reinforcement learning за останні 5 років і категорія cs.LG;
query = "reinforcement learning"
results = index.query(
    vector=get_embedding(query),
    top_k=TOP_K,
    include_metadata=True,
    filter={
        "category": "cs.LG",
        "year": {"$gte": datetime.now().year - 5},
    },
)
print(f"Приклад А: статті по '{query}' за останні 5 років і категорія cs.LG\n")
print_results(results)
# - приклад B: більш старі статті (до 2015 року), будь-яка категорія;
results = index.query(
    vector=get_embedding(query),
    top_k=TOP_K,
    include_metadata=True,
    filter={
        "year": {"$lte": 2015},
    },
)
print(f"Приклад B: статті по '{query}' до 2015 року\n")
print_results(results)
# - порівняти видачу і пояснити відмінності.
print("--- ПОРІВНЯННЯ А і Б ПРИКЛАДІВ ---")
print(
    "Приклад А (cs.LG + останні 5 років) повернув 0 результатів, а Приклад Б "
    "(до 2015 року, будь-яка категорія) повернув 5 статей. У датасеті "
    "'arxiv_subset.parquet' всі статті мають дату 2007 року - тобто датасет "
    "не містить свіжих статей (за останні 5 років) взагалі, тому фільтр А не міг "
    "повернути жодного результату. У прикладі Б - нема фільтра по категорії, "
    "тому повернуто 5 статей зі всіх можливих категорій."
)
print()

# 5. Порівняти різні метрики схожості на локальних ембеддингах:
# - завантажити всі ембеддинги з embeddings/embeddings.npy;
embeddings = np.load("embeddings/embeddings.npy")

# - для заданого запиту обчислити:
#   - cosine similarity;
embedding_query = get_embedding(query)
cosine_similarity = util.cos_sim(embeddings, embedding_query).numpy().flatten()
#   - dot product;
dot_product = np.dot(embeddings, embedding_query)
#   - L2-distance;
l2_distance = np.linalg.norm(embeddings - embedding_query, axis=1)


# - вивести топ-5 статей для кожної метрики і порівняти результати.
def print_top_results(name: str, scores: np.ndarray, descending: bool = True) -> None:
    print(f"Метрика: {name}")
    # Сортуємо індекси, а не самі значення
    sorted_indices = np.argsort(scores)
    if descending:
        sorted_indices = sorted_indices[::-1]
    for i in range(TOP_K):
        idx = sorted_indices[i]
        print(f"{i+1}. {df.iloc[idx]["title"]} | Score: {scores[idx]:.4f}")
    print()


print_top_results("Cosine Similarity", cosine_similarity)
print_top_results("Dot Product", dot_product)
# Для cosine similarity і dot product більше = краще (сортуємо в спадаючому порядку),
# а для L2-distance менше = краще (залишаємо зростаючий порядок)
print_top_results("L2-Distance", l2_distance, descending=False)

print("--- ПОРІВНЯННЯ МЕТРИК ---")
print(
    "Усі три метрики (Cosine Similarity, Dot Product, L2-Distance) видали "
    "однаковий список топ-5 статей в однаковому порядку. Тобто в цьому "
    "конкретному прикладі всі три способи рахування схожості 'погодились' між "
    "собою - хоч Score рахування в них різні."
)
