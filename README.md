# Лабораторная работа № 2. Вариант 1

Тема: «Обучение без учителя. Понижение размерности. Обнаружение аномалий. Кластеризация»

## Часть 1. Понижение размерности

- Датасет: `Lab2_Dataset_PCA.xlsx`, вариант 1: 243 объекта, признаки `a`, `b`, `c`, `d`, `e`; метка `y` используется только для визуализации (класс 0 — 124 объекта, класс 1 — 119 объектов).
- Корреляция признаков `d` и `e`: −0.9988; ранг матрицы признаков: 4.
- PCA explained variance ratio: 0.4046, 0.2198, 0.1997, 0.1759, 0.0000.
- PCA cumulative explained variance: 0.4046, 0.6244, 0.8241, 1.0000, 1.0000.
- SVD explained variance ratio: 0.4046, 0.2198, 0.1997, 0.1759, 0.0000.
- ICA: 4 компоненты (ранг матрицы равен 4); |excess kurtosis|: 3.9067, 0.8014, 0.5154, 0.3505.
- PCA/SVD: главные компоненты — [1, 2, 3], второстепенные — [4], шумовые — [5].
- ICA: главная компонента — [1], второстепенные — [2, 3], шумовая — [4].
- Графики: `lab2_part1_plots/`.

## Часть 2. Кластеризация

- Датасет: `Lab2_Dataset_Clustering.xlsx`, вариант 1: 222 объекта, признаки `a`, `b`; классы `y`: 0 — 56, 1 — 54, 2 — 55, 3 — 57.
- Признаки стандартизированы с помощью `StandardScaler`.
- Для DBSCAN, HDBSCAN и OPTICS выполнен автоматический подбор параметров по максимуму `adjusted_rand_score` относительно `y`.

| Алгоритм | ARI | Параметры / примечание |
|---|---:|---|
| K-Means | 0.4882 | n_clusters=4 |
| Affinity Propagation | 0.2684 | damping=0.9, preference=default |
| Mean-Shift | 0.0000 | bandwidth=estimated, 1 кластер |
| Spectral Clustering | 0.4573 | n_clusters=4 |
| Ward Hierarchical | 0.4523 | n_clusters=4, linkage=ward |
| Agglomerative | 0.4026 | n_clusters=4, linkage=complete |
| DBSCAN | 0.2934 | eps=0.3, min_samples=6, 5 кластеров, 68 шумовых точек |
| HDBSCAN | 0.4795 | min_cluster_size=14, min_samples=1, 4 кластера, 62 шумовые точки |
| OPTICS | 0.2863 | cluster_method=dbscan, min_samples=6, max_eps=0.3, 5 кластеров, 71 шумовая точка |
| Gaussian Mixtures | 0.7091 | n_components=4 |
| BIRCH | 0.5131 | n_clusters=4 |
| Bisecting K-Means | 0.2832 | n_clusters=4 |

- Лучший плотностный метод: HDBSCAN (ARI = 0.4795).
- Лучший результат среди всех алгоритмов: Gaussian Mixtures (ARI = 0.7091).
- Графики: `clustering_comparison_4x3.png`, `detected_anomalies.png`.
