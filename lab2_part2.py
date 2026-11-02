import os
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import (
    DBSCAN,
    HDBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    BisectingKMeans,
    KMeans,
    MeanShift,
    SpectralClustering,
)
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_FILE = "Lab2_Dataset_Clustering.xlsx"
VARIANT = 1
FEATURES = ["a", "b"]
TARGET = "y"
N_CLUSTERS = 4

start_col = (VARIANT - 1) * 3
usecols = list(range(start_col, start_col + 3))
df = pd.read_excel(DATA_FILE, header=1, usecols=usecols).dropna().reset_index(drop=True)
df.columns = FEATURES + [TARGET]

X = df[FEATURES].to_numpy(dtype=float)
y = df[TARGET].to_numpy(dtype=int)
X_scaled = StandardScaler().fit_transform(X)


def grid_search_dbscan(X_data, y_true):
    best_ari = -np.inf
    best_labels = None
    best_params = None
    for eps in np.round(np.arange(0.1, 1.51, 0.1), 2):
        for min_samples in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_data)
            if len(np.unique(labels)) < 2:
                continue
            ari = adjusted_rand_score(y_true, labels)
            if ari > best_ari:
                best_ari = ari
                best_labels = labels
                best_params = {"eps": float(eps), "min_samples": int(min_samples)}
    return best_labels, best_ari, best_params


def grid_search_hdbscan(X_data, y_true):
    best_ari = -np.inf
    best_labels = None
    best_params = None
    for min_cluster_size in range(10, 31, 2):
        for min_samples in [1, 2, 3, 4, 5, 8]:
            model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=0.0,
            )
            labels = model.fit_predict(X_data)
            if len(np.unique(labels)) < 2:
                continue
            ari = adjusted_rand_score(y_true, labels)
            if ari > best_ari:
                best_ari = ari
                best_labels = labels
                best_params = {
                    "min_cluster_size": int(min_cluster_size),
                    "min_samples": int(min_samples),
                }
    return best_labels, best_ari, best_params


def grid_search_optics(X_data, y_true):
    best_ari = -np.inf
    best_labels = None
    best_params = None
    for min_samples in [2, 3, 4, 5, 6, 8, 10]:
        for xi in [0.01, 0.03, 0.05, 0.08, 0.1, 0.2]:
            labels = OPTICS(
                min_samples=min_samples,
                xi=xi,
                cluster_method="xi",
            ).fit_predict(X_data)
            if len(np.unique(labels)) < 2:
                continue
            ari = adjusted_rand_score(y_true, labels)
            if ari > best_ari:
                best_ari = ari
                best_labels = labels
                best_params = {
                    "cluster_method": "xi",
                    "min_samples": int(min_samples),
                    "xi": float(xi),
                }
        for max_eps in np.round(np.arange(0.1, 2.01, 0.1), 2):
            labels = OPTICS(
                min_samples=min_samples,
                max_eps=max_eps,
                cluster_method="dbscan",
            ).fit_predict(X_data)
            if len(np.unique(labels)) < 2:
                continue
            ari = adjusted_rand_score(y_true, labels)
            if ari > best_ari:
                best_ari = ari
                best_labels = labels
                best_params = {
                    "cluster_method": "dbscan",
                    "min_samples": int(min_samples),
                    "max_eps": float(max_eps),
                }
    return best_labels, best_ari, best_params


def evaluate_labels(labels, ari):
    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels[labels != -1]))
    n_noise = int(np.sum(labels == -1))
    return n_clusters, n_noise, ari


results = {}
results["K-Means"] = {
    "labels": KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0).fit_predict(X_scaled),
    "params": f"n_clusters={N_CLUSTERS}",
}
results["Affinity Propagation"] = {
    "labels": AffinityPropagation(damping=0.9, random_state=0).fit_predict(X_scaled),
    "params": "damping=0.9, preference=default",
}
results["Mean-Shift"] = {
    "labels": MeanShift().fit_predict(X_scaled),
    "params": "bandwidth=estimated",
}
results["Spectral Clustering"] = {
    "labels": SpectralClustering(n_clusters=N_CLUSTERS, random_state=0).fit_predict(X_scaled),
    "params": f"n_clusters={N_CLUSTERS}",
}
results["Ward Hierarchical"] = {
    "labels": AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward").fit_predict(X_scaled),
    "params": f"n_clusters={N_CLUSTERS}, linkage=ward",
}
results["Agglomerative"] = {
    "labels": AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="complete").fit_predict(X_scaled),
    "params": f"n_clusters={N_CLUSTERS}, linkage=complete",
}
results["DBSCAN"] = {
    "labels": None,
    "params": None,
    "_search": grid_search_dbscan,
}
results["HDBSCAN"] = {
    "labels": None,
    "params": None,
    "_search": grid_search_hdbscan,
}
results["OPTICS"] = {
    "labels": None,
    "params": None,
    "_search": grid_search_optics,
}
results["Gaussian Mixtures"] = {
    "labels": GaussianMixture(n_components=N_CLUSTERS, random_state=0).fit_predict(X_scaled),
    "params": f"n_components={N_CLUSTERS}",
}
results["BIRCH"] = {
    "labels": Birch(n_clusters=N_CLUSTERS).fit_predict(X_scaled),
    "params": f"n_clusters={N_CLUSTERS}",
}
results["Bisecting K-Means"] = {
    "labels": BisectingKMeans(n_clusters=N_CLUSTERS, random_state=0).fit_predict(X_scaled),
    "params": f"n_clusters={N_CLUSTERS}",
}

for name, result in results.items():
    if result.get("labels") is None:
        labels, ari, params = result["_search"](X_scaled, y)
        result["labels"] = labels
        result["ari"] = ari
        result["params"] = params
    else:
        result["ari"] = adjusted_rand_score(y, result["labels"])
    result["n_clusters"], result["n_noise"], _ = evaluate_labels(result["labels"], result["ari"])

print(f"Variant: {VARIANT}")
print(f"Samples: {len(df)}")
print(f"Features: {FEATURES}")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print("\nClustering results:")
for name, result in results.items():
    print(
        f"{name:24s} ARI={result['ari']:.4f} "
        f"clusters={result['n_clusters']} noise={result['n_noise']} "
        f"params={result['params']}"
    )

density_names = ["DBSCAN", "HDBSCAN", "OPTICS"]
best_density_name = max(density_names, key=lambda name: results[name]["ari"])
print(f"\nBest density method: {best_density_name} (ARI={results[best_density_name]['ari']:.4f})")

method_order = [
    "K-Means",
    "Affinity Propagation",
    "Mean-Shift",
    "Spectral Clustering",
    "Ward Hierarchical",
    "Agglomerative",
    "DBSCAN",
    "HDBSCAN",
    "OPTICS",
    "Gaussian Mixtures",
    "BIRCH",
    "Bisecting K-Means",
]

fig, axes = plt.subplots(4, 3, figsize=(16, 18))
axes = axes.ravel()
cmap = plt.get_cmap("tab20")

for ax, name in zip(axes, method_order):
    result = results[name]
    labels = np.asarray(result["labels"])
    unique_labels = np.unique(labels)
    cluster_labels = [label for label in unique_labels if label != -1]

    for index, label in enumerate(cluster_labels):
        mask = labels == label
        ax.scatter(
            X_scaled[mask, 0],
            X_scaled[mask, 1],
            s=32,
            color=cmap(index % cmap.N),
            alpha=0.8,
            edgecolors="none",
            label=f"Cluster {label}",
        )

    if np.any(labels == -1):
        noise_mask = labels == -1
        ax.scatter(
            X_scaled[noise_mask, 0],
            X_scaled[noise_mask, 1],
            s=48,
            color="black",
            marker="x",
            linewidths=1.4,
            label="Noise",
        )

    ax.set_title(f"{name}\nARI = {result['ari']:.3f}", fontsize=11)
    ax.set_xlabel("Feature a (scaled)")
    ax.set_ylabel("Feature b (scaled)")
    ax.grid(True, linestyle=":", alpha=0.5)
    handles, legend_labels = ax.get_legend_handles_labels()
    legend_columns = 2 if len(handles) <= 10 else 3
    ax.legend(handles, legend_labels, fontsize=7, loc="best", ncol=legend_columns, framealpha=0.9)

fig.suptitle("Clustering algorithms comparison", fontsize=18)
fig.tight_layout(rect=[0, 0, 1, 0.97])
grid_path = "clustering_comparison_4x3.png"
fig.savefig(grid_path, dpi=150, bbox_inches="tight")
plt.close(fig)

best_result = results[best_density_name]
best_labels = np.asarray(best_result["labels"])
noise_mask = best_labels == -1
true_labels = np.unique(y)
true_cmap = plt.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(9, 7))
for index, label in enumerate(true_labels):
    mask = y == label
    ax.scatter(
        X_scaled[mask, 0],
        X_scaled[mask, 1],
        s=45,
        color=true_cmap(index % true_cmap.N),
        alpha=0.8,
        edgecolors="k",
        linewidths=0.4,
        label=f"True cluster {label}",
    )

ax.scatter(
    X_scaled[noise_mask, 0],
    X_scaled[noise_mask, 1],
    s=90,
    color="red",
    marker="x",
    linewidths=2.2,
    label=f"Anomalies / noise (n={int(noise_mask.sum())})",
)
ax.set_title(
    f"Anomalies detected by {best_density_name}\n"
    f"ARI = {best_result['ari']:.3f}, noise = {int(noise_mask.sum())}, "
    f"params = {best_result['params']}",
    fontsize=12,
)
ax.set_xlabel("Feature a (scaled)")
ax.set_ylabel("Feature b (scaled)")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=9, loc="best")
fig.tight_layout()
anomaly_path = "detected_anomalies.png"
fig.savefig(anomaly_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {os.path.abspath(grid_path)}")
print(f"Saved: {os.path.abspath(anomaly_path)}")
