import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1. Завантажити датасет із файлу data/arxiv_subset.parquet з використанням бібліотеки pandas.
df = pd.read_parquet("data/arxiv_subset.parquet")

print(df.head())
print(df.info())

# 2. Підготувати тексти для кодування:
# - для кожного запису об’єднати поля title і abstract в один рядок у форматі:title + " [SEP] " + abstract
# - важливо: токен [SEP] обов’язковий, оскільки модель навчена працювати саме з таким форматом вхідних даних.
df["text"] = df["title"] + " [SEP] " + df["abstract"]
texts = df["text"].tolist()

# 3. Згенерувати ембеддинги текстів за допомогою моделі allenai/specter2_base з бібліотеки sentence-transformers.
model = SentenceTransformer("allenai/specter2_base")

# 4. Закодувати всі тексти в ембеддинги з урахуванням таких вимог:
# - використовувати батчеву обробку (наприклад, batch_size=64);
# - увімкнути відображення прогресу;
# - нормалізувати ембеддинги (normalize_embeddings=True).
embeddings = model.encode(
    texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
)

# 5. Вивести в консоль:
# - загальну кількість оброблених текстів;
# - розмірність ембеддингів (очікується 768);
# - норму першого ембеддингу (повинна бути близька до 1.0).
print(f"Загальна кількість оброблених текстів: {len(embeddings)}")
print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
norm = np.linalg.norm(embeddings[0])
print(f"Норма першого ембеддингу: {norm:.6f}")

# 6. Зберегти отримані ембеддинги у файл embeddings/embeddings.npy у форматі NumPy.
# 7. Перед збереженням переконатися, що директорія embeddings існує; за потреби створити її.
os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/embeddings.npy", embeddings)
print("Ембеддинги успішно збережено в embeddings/embeddings.npy")
