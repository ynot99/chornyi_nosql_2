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
# Створюємо індекс (якщо не існує)

# 2. Завантажити дані:
# - прочитати датасет із файлу data/arxiv_subset.parquet;
# - завантажити ембеддинги з файлу embeddings/embeddings.npy.

# 3. Підготувати дані для завантаження:

# - обробляти записи батчами (наприклад, по 200 елементів);
# - для кожного запису сформувати об’єкт із:
# - унікальним id вигляду "paper_<номер>";
# - ембеддингами;
# - метаданими: arxiv_id, title, abstract (до 500 символів), authors (до 200 символів), year, category.


# 4. Завантажити дані в Pinecone батчами і показувати прогрес.

# 5. Після завершення завантаження вивести в консоль загальну кількість векторів в індексі.
