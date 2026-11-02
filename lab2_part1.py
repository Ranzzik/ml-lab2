import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

DATA_FILE = "Lab2_Dataset_PCA.xlsx"
VARIANT = 1
FEATURES = ["a", "b", "c", "d", "e"]
TARGET = "y"
OUT_DIR = "lab2_part1_plots"

os.makedirs(OUT_DIR, exist_ok=True)

start_col = (VARIANT - 1) * 6
usecols = list(range(start_col, start_col + 6))
df = pd.read_excel(DATA_FILE, header=1, usecols=usecols).dropna().reset_index(drop=True)
df.columns = FEATURES + [TARGET]

X = df[FEATURES].to_numpy(dtype=float)
y = df[TARGET].to_numpy(dtype=int)

print(f"Variant: {VARIANT}")
print(f"Samples: {len(df)}")
print(f"Features: {FEATURES}")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

corr = df[FEATURES].corr()
print("\nCorrelation matrix:")
print(corr.round(4).to_string())

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURES)))
ax.set_yticks(range(len(FEATURES)))
ax.set_xticklabels(FEATURES)
ax.set_yticklabels(FEATURES)
for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
ax.set_title("Feature correlation matrix")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
corr_path = os.path.join(OUT_DIR, "01_correlation_heatmap.png")
fig.savefig(corr_path, dpi=150, bbox_inches="tight")
plt.close(fig)

scatter_axes = pd.plotting.scatter_matrix(
    df[FEATURES],
    c=y,
    figsize=(10, 10),
    diagonal="hist",
    marker="o",
    alpha=0.7,
    cmap="viridis",
)
fig = scatter_axes[0, 0].get_figure()
fig.suptitle("Scatter matrix of features", fontsize=14)
fig.tight_layout()
scatter_path = os.path.join(OUT_DIR, "02_scatter_matrix.png")
fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
plt.close(fig)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=len(FEATURES), random_state=0)
scores_pca = pca.fit_transform(X_scaled)
evr_pca = pca.explained_variance_ratio_
loadings_pca = pca.components_.T

svd = TruncatedSVD(n_components=len(FEATURES), random_state=0)
scores_svd = svd.fit_transform(X_scaled)
evr_svd = svd.explained_variance_ratio_
loadings_svd = svd.components_.T

rank = np.linalg.matrix_rank(X_scaled)
n_ica = min(len(FEATURES), rank)
ica = FastICA(
    n_components=n_ica,
    fun="cube",
    whiten="unit-variance",
    random_state=0,
    max_iter=5000,
    tol=1e-4,
)
scores_ica_raw = ica.fit_transform(X_scaled)
kurt_ica_raw = kurtosis(scores_ica_raw, axis=0)
ica_order = np.argsort(np.abs(kurt_ica_raw))[::-1]
scores_ica = scores_ica_raw[:, ica_order]
loadings_ica = ica.mixing_[:, ica_order]
kurt_ica = kurt_ica_raw[ica_order]
significance_ica = np.abs(kurt_ica)

print(f"\nPCA explained variance ratio: {np.round(evr_pca, 4)}")
print(f"SVD explained variance ratio: {np.round(evr_svd, 4)}")
print(f"Data matrix rank: {rank}; ICA components used: {n_ica}")
print(f"ICA sorted |excess kurtosis|: {np.round(significance_ica, 4)}")


def summarize_variance(name, values):
    main, secondary, noise = [], [], []
    cumulative = 0.0
    for index, value in enumerate(values, start=1):
        previous = cumulative
        cumulative += value
        if previous < 0.80:
            main.append(index)
        elif previous < 0.95:
            secondary.append(index)
        else:
            noise.append(index)
    print(f"\n{name} components:")
    print(f"  Main: {main or 'none'}")
    print(f"  Secondary: {secondary or 'none'}")
    print(f"  Noise: {noise or 'none'}")
    print(f"  Cumulative explained variance: {np.round(np.cumsum(values), 4)}")


def summarize_ica(name, values):
    main = [i for i, value in enumerate(values, start=1) if value >= 1.0]
    secondary = [i for i, value in enumerate(values, start=1) if 0.5 <= value < 1.0]
    noise = [i for i, value in enumerate(values, start=1) if value < 0.5]
    print(f"\n{name} components:")
    print(f"  Main: {main or 'none'}")
    print(f"  Secondary: {secondary or 'none'}")
    print(f"  Noise: {noise or 'none'}")


summarize_variance("PCA", evr_pca)
summarize_variance("SVD", evr_svd)
summarize_ica("ICA", significance_ica)


def plot_variance_significance(name, values, path):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    indices = np.arange(1, len(values) + 1)
    ax1.bar(indices, values, color="steelblue", alpha=0.75, label="Individual explained variance")
    ax1.set_xlabel("Component index")
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_xticks(indices)
    ax2 = ax1.twinx()
    ax2.plot(indices, np.cumsum(values), marker="o", color="darkorange", label="Cumulative explained variance")
    ax2.set_ylabel("Cumulative explained variance ratio")
    ax2.set_ylim(0, 1.05)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="center right")
    ax1.set_title(f"{name} component significance")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ica_significance(values, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    indices = np.arange(1, len(values) + 1)
    ax.bar(indices, values, color="seagreen", alpha=0.8)
    ax.set_xlabel("Independent component index")
    ax.set_ylabel("|Excess kurtosis|")
    ax.set_xticks(indices)
    ax.set_title("ICA component significance (absolute excess kurtosis)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


plot_variance_significance("PCA", evr_pca, os.path.join(OUT_DIR, "03_pca_significance.png"))
plot_variance_significance("SVD", evr_svd, os.path.join(OUT_DIR, "04_svd_significance.png"))
plot_ica_significance(significance_ica, os.path.join(OUT_DIR, "05_ica_significance.png"))


def biplot(scores, loadings, labels=None, targets=None, target_labels=None):
    scores = np.asarray(scores, dtype=float)
    loadings = np.asarray(loadings, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 7))

    score_scale = 0.9 / max(np.abs(scores).max(), 1e-12)
    loading_scale = 0.9 / max(np.abs(loadings).max(), 1e-12)
    scaled_scores = scores * score_scale

    if targets is not None:
        scatter = ax.scatter(
            scaled_scores[:, 0],
            scaled_scores[:, 1],
            c=targets,
            cmap="viridis",
            alpha=0.7,
            edgecolors="k",
        )
        handles, _ = scatter.legend_elements()
        if target_labels is None:
            target_labels = np.unique(targets)
        ax.legend(handles, target_labels, title="Classes")
    else:
        ax.scatter(scaled_scores[:, 0], scaled_scores[:, 1], alpha=0.7, edgecolors="k")

    for i in range(loadings.shape[0]):
        x_end = loadings[i, 0] * loading_scale
        y_end = loadings[i, 1] * loading_scale
        ax.arrow(
            0,
            0,
            x_end,
            y_end,
            color="r",
            alpha=0.7,
            head_width=0.03,
            head_length=0.03,
            linewidth=1.5,
            length_includes_head=True,
        )
        if labels is not None:
            ax.text(
                x_end * 1.15,
                y_end * 1.15,
                labels[i],
                color="darkred",
                ha="center",
                va="center",
                fontweight="bold",
            )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    return fig, ax


def save_biplot(name, scores, loadings, x_label, y_label, path):
    fig, ax = biplot(scores, loadings, FEATURES, y, [0, 1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{name} Biplot")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


save_biplot(
    "PCA",
    scores_pca,
    loadings_pca,
    f"PC1 ({evr_pca[0] * 100:.1f}% variance)",
    f"PC2 ({evr_pca[1] * 100:.1f}% variance)",
    os.path.join(OUT_DIR, "06_pca_biplot.png"),
)
save_biplot(
    "SVD",
    scores_svd,
    loadings_svd,
    f"SVD1 ({evr_svd[0] * 100:.1f}% variance)",
    f"SVD2 ({evr_svd[1] * 100:.1f}% variance)",
    os.path.join(OUT_DIR, "07_svd_biplot.png"),
)
save_biplot(
    "ICA",
    scores_ica,
    loadings_ica,
    f"IC1 (|kurtosis| = {significance_ica[0]:.2f})",
    f"IC2 (|kurtosis| = {significance_ica[1]:.2f})",
    os.path.join(OUT_DIR, "08_ica_biplot.png"),
)

print("\nSaved figures:")
for file_name in sorted(os.listdir(OUT_DIR)):
    print(os.path.join(OUT_DIR, file_name))
