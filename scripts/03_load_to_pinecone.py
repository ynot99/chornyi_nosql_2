import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200  # Pinecone рекомендує батчі до 200 векторів

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# 1. Створити індекс arxiv-papers у Pinecone, якщо він ще не існує, і підключитися до нього.
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )
    print(f"Індекс '{INDEX_NAME}' створено")
else:
    print(f"Індекс '{INDEX_NAME}' вже існує")

index = pc.Index(INDEX_NAME)

# 2. Завантажити дані:
# - прочитати датасет із файлу data/arxiv_subset.parquet;
df = pd.read_parquet(INPUT_PARQUET)
# - завантажити ембеддинги з файлу embeddings/embeddings.npy.
embeddings = np.load(INPUT_EMBEDDINGS)

# 3. Підготувати дані для завантаження:
# - обробляти записи батчами (наприклад, по 200 елементів);
for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Завантаження батчів у Pinecone"):
    # Беремо однаковий батч з обох масивів
    df_batch = df.iloc[i : i + BATCH_SIZE]
    embeddings_batch = embeddings[i : i + BATCH_SIZE]

    vectors = []
    # - для кожного запису сформувати об’єкт із:
    for (id, row), embedding in zip(df_batch.iterrows(), embeddings_batch):
        # - унікальним id вигляду "paper_<номер>";
        vector_id = f"paper_{id}"
        # - ембеддингами;
        embedding_list = embedding.tolist()
        # - метаданими: arxiv_id, title, abstract (до 500 символів), authors (до 200 символів), year, category.
        metadata = {
            "arxiv_id": row["id"],
            "title": row["title"],
            "abstract": row["abstract"][:500],
            "authors": row["authors"][:200],
            "year": row["year"],
            "category": row["category"],
        }

        vector = {
            "id": vector_id,
            "values": embedding_list,
            "metadata": metadata,
        }
        vectors.append(vector)

    # 4. Завантажити дані в Pinecone батчами і показувати прогрес.
    index.upsert(vectors=vectors)

    # Прогрес показується за допомогою tqdm в циклі вище

# 5. Після завершення завантаження вивести в консоль загальну кількість векторів в індексі.
stats = index.describe_index_stats()
print(f"\nВсього векторів в індексі: {stats['total_vector_count']}")
