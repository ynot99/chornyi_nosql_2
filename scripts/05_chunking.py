import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

# 1. Вибрати 30 статей із найдовшими анотаціями.
longest_abstracts = df.sort_values(
    by="abstract", key=lambda x: x.str.len(), ascending=False
).head(30)

# 2. Розбити тексти на чанки двома стратегіями:
# - Fixed-size chunking: фіксована кількість слів з невеликим перекриттям між чанками;
FIXED_CHUNK_WORDS = 40
FIXED_CHUNK_OVERLAP_WORDS = 10


def fixed_size_chunking(
    text: str,
    chunk_size: int = FIXED_CHUNK_WORDS,
    overlap: int = FIXED_CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Повертає список рядків, де кожен рядок - це chunk з фіксованою кількістю слів.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break

    return chunks


# - Semantic chunking: об'єднання речень до досягнення максимальної кількості слів, щоб зберегти зміст.
def semantic_chunking(
    text: str,
    model: SentenceTransformer,
    threshold: float = 0.7,
    min_chunk_size: int = 50,
) -> list[str]:
    """
    Ділить текст на семантично зв'язні блоки.
    Новий chunk починається, коли косинусна схожість
    між сусідніми реченнями падає нижче threshold.
    """
    # Просте розділення на речення
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if len(sentences) < 2:
        return sentences

    # Отримуємо ембеддинги речень
    embeddings = model.encode(sentences, normalize_embeddings=True)

    # Косинусна схожість між сусідніми реченнями
    similarities = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    chunks, current_chunk = [], [sentences[0]]
    for i, sim in enumerate(similarities):
        if sim < threshold and len(" ".join(current_chunk)) >= min_chunk_size:
            chunks.append(". ".join(current_chunk) + ".")
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")

    return chunks


# Розбиваємо всі 30 текстів обома стратегіями
fixed_chunks_data = []
semantic_chunks_data = []

for _, row in longest_abstracts.iterrows():
    text = row["abstract"].strip()

    # Fixed-size chunking
    chunks_fixed = fixed_size_chunking(text)
    # Прив'язуємо метадані статті до кожного чанка тут, бо після цього циклу
    # лишається лише list[str] без зв'язку з конкретною статтею (arxiv_id/title/...)
    for chunk_num, chunk_text in enumerate(chunks_fixed):
        fixed_chunks_data.append(
            {
                "arxiv_id": row["id"],
                "title": row["title"],
                "year": row["year"],
                "category": row["category"],
                "chunk_text": chunk_text,
                "chunk_num": chunk_num,
            }
        )

    # Semantic chunking
    chunks_semantic = semantic_chunking(text, model, threshold=0.6)
    for chunk_num, chunk_text in enumerate(chunks_semantic):
        semantic_chunks_data.append(
            {
                "arxiv_id": row["id"],
                "title": row["title"],
                "year": row["year"],
                "category": row["category"],
                "chunk_text": chunk_text,
                "chunk_num": chunk_num,
            }
        )

print(f"Fixed-size chunks: {len(fixed_chunks_data)}")
print(f"Semantic chunks: {len(semantic_chunks_data)}")

# Приклад виводу перших чанків першої статті
first_article_id = longest_abstracts.iloc[0]["id"]


def print_chunks_preview(chunks_data: list[dict], label: str):
    print(f"\n--- {label} (стаття 1) ---")
    for item in chunks_data:
        if item["arxiv_id"] == first_article_id:
            print(
                f"  Chunk {item["chunk_num"]+1} ({len(item["chunk_text"])} chars): {item["chunk_text"][:80]}..."
            )


print_chunks_preview(fixed_chunks_data, "Fixed-size chunks")
print_chunks_preview(semantic_chunks_data, "Semantic chunks")

# 3. Створити окремі індекси в Pinecone для кожного типу чанків (arxiv-chunks-fixed і arxiv-chunks-semantic).
INDEX_FIXED_NAME = "arxiv-chunks-fixed"
INDEX_SEMANTIC_NAME = "arxiv-chunks-semantic"


def create_index(index_name: str):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
        print(f"Індекс '{index_name}' створено")
    else:
        print(f"Індекс '{index_name}' вже існує")


create_index(INDEX_FIXED_NAME)
create_index(INDEX_SEMANTIC_NAME)

fixed_index = pc.Index(INDEX_FIXED_NAME)
semantic_index = pc.Index(INDEX_SEMANTIC_NAME)


# 4. Для кожного чанка:
# - створити ембеддинг за допомогою моделі allenai/specter2_base;
def get_embedding(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts, convert_to_numpy=True)
    return np.asarray(embeddings).tolist()


fixed_embeddings = np.array(
    get_embedding([chunk["chunk_text"] for chunk in fixed_chunks_data])
)
semantic_embeddings = np.array(
    get_embedding([chunk["chunk_text"] for chunk in semantic_chunks_data])
)


# - сформувати об’єкт з унікальним id, ембеддингом і метаданими: arxiv_id, title, текст чанка, номер чанка, рік, категорія.
def build_vectors(chunks_data: list[dict], embeddings: np.ndarray) -> list[dict]:
    vectors = []
    for i, chunk in enumerate(chunks_data):
        vectors.append(
            {
                "id": f"{chunk["arxiv_id"]}-{chunk["chunk_num"]}",
                "values": embeddings[i].tolist(),
                "metadata": {
                    "arxiv_id": chunk["arxiv_id"],
                    "title": chunk["title"],
                    "chunk_text": chunk["chunk_text"],
                    "chunk_num": chunk["chunk_num"],
                    "year": chunk["year"],
                    "category": chunk["category"],
                },
            }
        )
    return vectors


fixed_vectors = build_vectors(fixed_chunks_data, fixed_embeddings)
semantic_vectors = build_vectors(semantic_chunks_data, semantic_embeddings)

# 5. Завантажувати чанки в Pinecone батчами і відображати прогрес.
batch_size = 100


def upsert_in_batches(index, vectors: list[dict]):
    for i in tqdm(range(0, len(vectors), batch_size)):
        index.upsert(vectors=vectors[i : i + batch_size])


upsert_in_batches(fixed_index, fixed_vectors)
upsert_in_batches(semantic_index, semantic_vectors)


# 6. Реалізувати функцію пошуку по чанках:
# - виконати пошук за кількома тестовими запитами;
# - вивести топ-5 результатів для кожного типу чанків з назвою статті і частиною тексту чанка.
def search_chunks(index, query: str, k: int = 5):
    query_embedding = get_embedding([query])[0]
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True,
    )
    return results


TEST_QUERIES = [
    "What is machine learning?",
    "Neural networks for image recognition",
    "Natural language processing transformers",
]


def print_search_results(index, query: str, label: str):
    results = search_chunks(index, query)
    print(f"{label} search results:")
    for result in results["matches"]:
        print(
            f"  {result["metadata"]["title"]}: {result["metadata"]["chunk_text"][:80]}..."
        )


for query in TEST_QUERIES:
    print(f"\n=== Query: {query} ===")
    print_search_results(fixed_index, query, "Fixed-size")
    print_search_results(semantic_index, query, "Semantic")
