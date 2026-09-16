---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: ml
    language: python
    name: python3
---


# 34. Niveau 1 (Guidé)

Reprenez le jeu de données de 15 clients et réalisez les étapes suivantes :

1. Entraînez un modèle K-means avec $K = 2$.
2. Visualisez le résultat sous forme de scatter plot en colorant les points selon leur cluster.
3. Affichez les coordonnées des centroïdes.
4. Entraînez ensuite un modèle avec $K = 3$.
5. Calculez et comparez les inerties (`model.inertia_`) pour $K = 2$ et $K = 3$.

### Question

Pourquoi l'inertie baisse-t-elle nécessairement quand on passe de $K = 2$ à $K = 3$ ?

# 35. Niveau 2 (Réflexion)

Utilisez `make_blobs` de `sklearn.datasets` pour générer un dataset synthétique en 2D avec 300 clients, 4 centres distincts et une variance standard de 1.2 (`cluster_std=1.2`).

```python
from sklearn.datasets import make_blobs

X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=1.2,
    random_state=42
)

```

1. Tracez les données générées.
2. Calculez l'inertie pour $K$ allant de 1 à 10 sur ce dataset.
3. Tracez la courbe du coude.
4. Retrouve-t-on visuellement le nombre réel de clusters ($K=4$) ?
5. Que se passe-t-il si vous augmentez la variance (`cluster_std=3.0`) ? Le coude reste-t-il aussi facile à repérer ?


# 36. Niveau 3 (Challenge)

Créez (ou générez avec `numpy`/`pandas`) un jeu de données de segmentation e-commerce comportant 4 variables :

* `age` (de 18 à 70 ans)
* `revenu_annuel` (de 15 à 150 k€)
* `score_depense` (de 1 à 100)
* `frequence_achats` (nombre d'achats par an, de 1 à 50)

### Travail demandé

1. Générer le jeu de données (au moins 200 lignes).
2. Vérifier l'échelle des variables à l'aide d'un `.describe()`.
3. Construire un `Pipeline` intégrant `StandardScaler` et `KMeans`.
4. Évaluer plusieurs valeurs de $K$ (de 2 à 8) avec la courbe du coude ET le score de silhouette (`from sklearn.metrics import silhouette_score`).
5. Choisir la meilleure valeur de $K$ et entraîner le modèle final.
6. Réaliser une analyse `.groupby('cluster').mean()` des variables initiales non standardisées.
7. Rédiger une synthèse métier décrivant chaque profil client trouvé (ex: « Jeunes impulseurs », « Seniors VIP », etc.).

### Question finale

> Pourquoi est-il indispensable de réinterpréter les moyennes des clusters sur les données *non standardisées* lors de la restitution métier ?


# 37. Pour aller plus loin

Dans la suite du cours, nous comparerons K-means avec :

* **DBSCAN** : idéal pour identifier des clusters de formes arbitraires et détecter les valeurs aberrantes (outliers) ;
* **Clustering hiérarchique (AHC)** : utile pour construire des dendrogrammes et comprendre les relations d'emboîtement entre groupes.
