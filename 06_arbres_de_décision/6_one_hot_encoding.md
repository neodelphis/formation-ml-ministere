# Pourquoi utiliser le One-Hot Encoding ?

La raison principale d'utiliser le **One-Hot Encoding** est simple : représenter les catégories sans introduire artificiellement un ordre ou une distance entre elles.


## Le problème avec l'encodage numérique

Imaginons une variable catégorielle représentant la ville d'un appartement :

```text
Paris = 1
Versailles = 2
Marseille = 3
```

Les valeurs 1, 2 et 3 ne sont que des codes, et non des quantités numériques, pourtant, un modèle peut être amené à les traiter comme telles.

### 1. Un ordre artificiel

L'encodage suggère implicitement :

```text
Paris < Versailles < Marseille
```

Mais il n'existe évidemment pas de hiérarchie naturelle entre ces trois villes.

### 2. Des écarts artificiels

L'encodage suggère également que :

```text
distance(Paris, Versailles) = 1
distance(Versailles, Marseille) = 1
```

alors que cette notion de distance entre les catégories n'a aucun sens ici.

Plus généralement, si l'on avait :

```text
Paris = 1
Versailles = 2
Marseille = 10
```

on aurait artificiellement créé une différence beaucoup plus importante entre Versailles et Marseille qu'entre Paris et Versailles, alors que les nombres utilisés ne sont que des étiquettes.

### 3. Une relation numérique artificielle dans certains modèles

Par exemple, dans une régression linéaire, le modèle pourrait apprendre quelque chose de la forme :

```text
Prix = a × Ville + b
```

Il supposerait alors implicitement que passer de Paris (1) à Versailles (2) a un effet comparable au passage de Versailles (2) à Marseille (3).

Ce n'est généralement pas ce que l'on souhaite pour une variable nominale comme la ville.

## La solution : le One-Hot Encoding

Le One-Hot Encoding consiste à créer une colonne binaire pour chaque catégorie.

Le **One-Hot Encoding** consiste à créer une colonne binaire pour chaque catégorie.

Avec nos trois villes :

| Ville      | Est_Paris | Est_Versailles | Est_Marseille |
| ---------- | --------: | -------------: | ------------: |
| Paris      |         1 |              0 |             0 |
| Versailles |         0 |              1 |             0 |
| Marseille  |         0 |              0 |             1 |

Chaque ligne indique simplement à quelle catégorie appartient l'observation.

### Exemple avec des prix d'appartements

Imaginons un petit jeu de données contenant le prix de six appartements situés
à Paris, Versailles ou Marseille. Le prix est exprimé en euros.

#### Données initiales

| Appartement | Ville      | Prix (€) |
| ----------- | ---------- | -------: |
| A           | Paris      |  320 000 |
| B           | Paris      |  450 000 |
| C           | Versailles |  280 000 |
| D           | Versailles |  390 000 |
| E           | Marseille  |  180 000 |
| F           | Marseille  |  240 000 |

La colonne `Ville` est une variable catégorielle. Pour la rendre exploitable
par un modèle qui attend des valeurs numériques, on la remplace par trois
colonnes binaires : `Est_Paris`, `Est_Versailles` et `Est_Marseille`.

#### Données après One-Hot Encoding

| Appartement | Prix (€) | Est_Paris | Est_Versailles | Est_Marseille |
| ----------- | -------: | --------: | -------------: | ------------: |
| A           |  320 000 |         1 |              0 |             0 |
| B           |  450 000 |         1 |              0 |             0 |
| C           |  280 000 |         0 |              1 |             0 |
| D           |  390 000 |         0 |              1 |             0 |
| E           |  180 000 |         0 |              0 |             1 |
| F           |  240 000 |         0 |              0 |             1 |

Par exemple, l'appartement C est situé à Versailles. Sa ville est donc
représentée par `[0, 1, 0]`, tandis que son prix reste une valeur numérique :

```text
Appartement C → Prix = 280 000, [Est_Paris, Est_Versailles, Est_Marseille] = [0, 1, 0]
```

Le modèle peut ainsi utiliser le prix comme une variable numérique et la ville
comme un ensemble de variables binaires, sans supposer que Paris, Versailles
ou Marseille sont naturellement classées les unes par rapport aux autres.

Par exemple :

- Paris → `[1, 0, 0]`
- Versailles → `[0, 1, 0]`
- Marseille → `[0, 0, 1]`

Les nombres 0 et 1 ne représentent donc plus une quantité ou un classement. Ils indiquent simplement la présence ou l'absence d'une catégorie.

## Pourquoi est-ce intéressant ?

Le One-Hot Encoding évite d'imposer une relation artificielle entre les catégories.

Avec les représentations suivantes :

- Paris → `[1, 0, 0]`
- Versailles → `[0, 1, 0]`
- Marseille → `[0, 0, 1]`

aucune ville n'est « plus grande » ou « plus petite » qu'une autre.

On peut également considérer que chaque catégorie correspond à une dimension différente de l'espace des données. Les catégories sont ainsi séparées plutôt que placées arbitrairement sur une même échelle numérique.

## Quand peut-on utiliser un encodage numérique ?

Le One-Hot Encoding n'est pas toujours nécessaire. Tout dépend de la nature de la variable et du modèle utilisé.

### 1. Lorsqu'il existe un ordre naturel

Si les catégories sont ordinales, leur ordre a une signification.

Par exemple, pour une taille de vêtement :

```text
S = 1
M = 2
L = 3
XL = 4
```

Ici, l'ordre est réel :

```text
S < M < L < XL
```

Un encodage numérique peut donc être pertinent.

Attention toutefois : même dans ce cas, il faut réfléchir à la signification des écarts. La différence entre S et M n'est pas nécessairement exactement équivalente à celle entre L et XL.

### 2. Lorsque le modèle gère les variables catégorielles nativement

Certains algorithmes et bibliothèques peuvent traiter directement les variables catégorielles sans qu'il soit nécessaire de les transformer manuellement en One-Hot Encoding.

Dans ce cas, il faut suivre la méthode recommandée par le modèle utilisé.

### 3. Avec certains modèles basés sur des arbres

Les arbres de décision fonctionnent différemment des modèles qui utilisent directement des distances ou des produits matriciels.

Cependant, il est faux de dire qu'un simple Label Encoding est toujours adapté aux arbres.

Si on donne :

```text
Paris = 1
Versailles = 2
Marseille = 3
```

un arbre peut par exemple effectuer une séparation du type :

```text
Ville < 1,5 ?
```

Il regroupe alors Paris d'un côté et Versailles + Marseille de l'autre.

Le problème est que ce regroupement dépend de l'ordre arbitraire que nous avons choisi pour encoder les villes.

## Les arbre de décision avec scikit-learn

En théorie un arbre de décision n'a pas besoin de One-Hot Encoding. Il est capable de séparer les données en se posant des questions du type : *"La ville est-elle Paris OU Marseille OU Versailles?"*.

### La limitation technique de scikit-learn

Les modèles standards de `scikit-learn` comme `DecisionTreeClassifier` ou `RandomForestClassifier` utilisent une version spécifique de l'algorithme des arbres (appelée CART) qui a été **codée pour ne traiter que des variables numériques continues ou ordonnées**.

Si vous utilisez de simples chiffres (Paris=1, Versailles=2, Marseille=3), l'arbre de `scikit-learn` ne peut pas faire de groupes arbitraires. Il ne sait faire que des coupures mathématiques basées sur **"plus petit que"** ou **"plus grand que"**.

### Ce qui se passe avec de simples chiffres (Label Encoding)

Si l'algorithme veut isoler les appartements de Versailles (2), il ne peut pas dire *"Est-ce que Ville == 2 ?"*. Il est obligé de faire deux coupes successives :

1. *"Est-ce que Ville > 1.5 ?"* (Il élimine Paris)
2. *"Est-ce que Ville < 2.5 ?"* (Il élimine Marseille)

Non seulement cela force l'arbre à faire des coupes supplémentaires (ce qui rend l'arbre plus profond et complexe), mais cela force aussi l'algorithme à regrouper temporairement des villes qui n'ont rien à voir géographiquement juste parce que leurs numéros se suivent.

### Pourquoi le One-Hot Encoding aide (dans ce cas précis)

En passant au One-Hot Encoding (`Est_Paris`, `Est_Versailles`, `Est_Marseille` avec des 0 et des 1), vous redonnez à `scikit-learn` la possibilité de faire une coupe propre et directe.

L'algorithme peut simplement demander : *"Est-ce que Est_Versailles > 0.5 ?"*. S'il répond oui, il a isolé Versailles en une seule étape, sans aucune fausse hiérarchie.

---

### ⚠️ Le revers de la médaille (et les solutions modernes)

Bien que le One-Hot Encoding règle ce problème technique pour les arbres de `scikit-learn`, il crée un autre défaut : si vous avez 100 villes, vous créez 100 colonnes remplies de zéros. Les arbres classiques deviennent très mauvais lorsqu'ils doivent gérer des données aussi "creuses" (sparse).

**Comment faire aujourd'hui ?**
La communauté du Machine Learning a bien compris cette limite technique.

1. **Les algorithmes externes :** Des bibliothèques extrêmement performantes basées sur les arbres, comme **LightGBM** ou **CatBoost**, ont été créées spécifiquement pour gérer les catégories nativement sans One-Hot Encoding.
2. **La mise à jour de scikit-learn :** Depuis la version 0.24, `scikit-learn` a introduit un nouveau modèle, le `HistGradientBoostingClassifier`, qui est enfin capable de gérer les variables catégorielles sans que vous n'ayez besoin de faire de One-Hot Encoding au préalable !