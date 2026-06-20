          id                                              title  ...  year        category
0  0704.0001  Calculation of prompt diphoton production cros...  ...  2007          hep-ph
1  0704.0002           Sparsity-certifying Graph Decompositions  ...  2007         math.CO
2  0704.0003  The evolution of the Earth-Moon system based o...  ...  2007  physics.gen-ph
3  0704.0004  A determinant of Stirling cycle numbers counts...  ...  2007         math.CO
4  0704.0005  From dyadic $\Lambda_{\alpha}$ to $\Lambda_{\a...  ...  2007         math.CA

[5 rows x 6 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 6 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   id        10000 non-null  object
 1   title     10000 non-null  object
 2   abstract  10000 non-null  object
 3   authors   10000 non-null  object
 4   year      10000 non-null  int64
 5   category  10000 non-null  object
dtypes: int64(1), object(5)
memory usage: 468.9+ KB
None
No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [18:04<00:00,  6.91s/it]
Загальна кількість оброблених текстів: 10000
Розмірність ембеддингів: 768
Норма першого ембеддингу: 1.000000
Ембеддинги успішно збережено в embeddings/embeddings.npy
