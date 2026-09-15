---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: ml
    language: python
    name: ml
---

# Module 1 — Explorer un jeu de données avec Pandas

> **Manipulation des données avec Pandas — Devenir Data Scientist**


## 🎯 Objectifs du module

À la fin de ce module, vous serez capables de :

* charger un jeu de données avec Pandas ;
* comprendre rapidement sa structure ;
* identifier les variables et leurs types ;
* sélectionner des colonnes ;
* sélectionner des lignes ;
* filtrer des observations ;
* combiner plusieurs conditions ;
* analyser des variables catégorielles ;
* produire des statistiques descriptives ;
* identifier les premiers problèmes de qualité des données ;
* adopter une démarche systématique face à un dataset inconnu.


# 1. Le travail commence rarement par un modèle

En Data Science, on imagine parfois le workflow suivant :

```text
Données
   ↓
Machine Learning
   ↓
Prédiction
```

Dans la réalité, le processus ressemble davantage à :

```text
                 Dataset
                    │
                    ↓
              Exploration
                    │
                    ↓
             Compréhension
                    │
                    ↓
              Nettoyage
                    │
                    ↓
           Transformation
                    │
                    ↓
        Feature Engineering
                    │
                    ↓
              Modélisation
```

Avant de construire un modèle, il faut donc répondre à des questions simples :

> Combien avons-nous d'observations ?

> Quelles sont les variables disponibles ?

> Quels sont leurs types ?

> Y a-t-il des valeurs manquantes ?

> Les valeurs semblent-elles cohérentes ?

> Quelles sont les catégories présentes ?

> Existe-t-il des valeurs aberrantes ?

> Quelles relations semblent intéressantes ?


# 2. Un réflexe de Data Scientist

Lorsqu'on reçoit un nouveau dataset, **ne commencez pas immédiatement à le transformer**.

Commencez par l'observer.

Une bonne séquence initiale est :

```text
1. Charger
      ↓
2. Regarder quelques lignes
      ↓
3. Comprendre les dimensions
      ↓
4. Examiner les colonnes
      ↓
5. Examiner les types
      ↓
6. Examiner les statistiques
      ↓
7. Chercher les valeurs manquantes
      ↓
8. Explorer les catégories
      ↓
9. Commencer à poser des questions
```

Cette séquence va devenir un réflexe.


# 3. Notre dataset

Nous allons travailler avec un jeu de données représentant des **commandes d'une boutique en ligne**.

Chaque ligne représente une commande.

Les variables disponibles sont :

| Colonne           | Description                |
| ----------------- | -------------------------- |
| `order_id`        | Identifiant de la commande |
| `customer_id`     | Identifiant du client      |
| `order_date`      | Date de la commande        |
| `country`         | Pays du client             |
| `category`        | Catégorie du produit       |
| `quantity`        | Quantité commandée         |
| `unit_price`      | Prix unitaire              |
| `discount`        | Remise appliquée           |
| `payment_method`  | Moyen de paiement          |
| `customer_rating` | Note donnée par le client  |


# 4. Importer Pandas

```python
import pandas as pd
```

Vérifions la version utilisée :

```python
pd.__version__
```


# 5. Charger les données

Dans un projet réel, les données peuvent provenir :

* d'un CSV ;
* d'un fichier Excel ;
* d'une base SQL ;
* d'une API ;
* d'un fichier JSON ;
* d'un data lake ;
* d'un autre DataFrame.

Ici, nous utiliserons un fichier CSV.

```python
df = pd.read_csv("data/orders.csv")
```


# 6. Première observation : `head()`

La première chose à faire après avoir chargé un dataset :

```python
df.head()
```

Par défaut, Pandas affiche les cinq premières lignes.

Nous pouvons demander davantage :

```python
df.head(10)
```


## Pourquoi `head()` est-il important ?

Il permet de détecter immédiatement des problèmes évidents :

* noms de colonnes inattendus ;
* valeurs mal formatées ;
* séparateur incorrect ;
* données décalées ;
* valeurs étranges ;
* dates qui semblent incorrectes ;
* colonnes qui ne contiennent pas ce que l'on attend.


# 7. Observer la fin du dataset

```python
df.tail()
```

Pourquoi regarder aussi les dernières lignes ?

Parce que certains problèmes peuvent apparaître uniquement à la fin du fichier.


# 8. Combien avons-nous de données ?

Utilisons `shape`.

```python
df.shape
```

Le résultat est un tuple :

```text
(nombre_de_lignes, nombre_de_colonnes)
```

Par exemple :

```text
(10000, 10)
```

signifie :

> 10 000 observations et 10 variables.


## Extraire séparément les dimensions

```python
rows, columns = df.shape

print("Nombre de lignes :", rows)
print("Nombre de colonnes :", columns)
```


# 9. Les noms des colonnes

```python
df.columns
```

On peut également les transformer en liste :

```python
list(df.columns)
```


## 💡 Question

Pourquoi est-il important de vérifier les noms des colonnes ?

Imaginez que vous pensiez avoir :

```text
customer_id
```

mais que le dataset contienne réellement :

```text
Customer ID
```

ou :

```text
customerId
```

ou :

```text
customer_id
```

Pour Pandas, ce sont trois colonnes différentes.


# 10. Les types de données

```python
df.dtypes
```

On peut par exemple obtenir :

```text
order_id              object
customer_id            int64
order_date            object
country               object
category              object
quantity               int64
unit_price           float64
discount             float64
payment_method        object
customer_rating      float64
```


# 11. Pourquoi les types sont-ils importants ?

Le type détermine en partie les opérations que nous pouvons effectuer.

Par exemple :

```text
int
    ↓
calculs numériques

float
    ↓
calculs numériques avec décimales

object / string
    ↓
texte

datetime
    ↓
dates et temps
```

Prenons une date :

```python
df["order_date"].head()
```

Si elle est de type `object`, Pandas la considère essentiellement comme du texte.

Nous verrons dans un prochain module comment convertir correctement les dates.


# 12. `info()` : notre première commande de diagnostic

```python
df.info()
```

Cette commande est particulièrement utile.

Elle donne notamment :

* le nombre de lignes ;
* les noms des colonnes ;
* le nombre de valeurs non nulles ;
* les types ;
* une estimation de la mémoire utilisée.


## 🧠 Réflexe Data Scientist

Après avoir chargé un nouveau dataset :

```python
df.head()
df.shape
df.info()
```

Ces trois commandes donnent déjà énormément d'informations.


# 13. Sélectionner une colonne

Pour récupérer une colonne :

```python
df["country"]
```

Le résultat est une `Series`.

Vérifions :

```python
type(df["country"])
```


# 14. Sélectionner plusieurs colonnes

Pour sélectionner plusieurs colonnes, on utilise une liste :

```python
df[["country", "category", "quantity"]]
```

Attention à la différence :

```python
df["country"]
```

et :

```python
df[["country"]]
```

Le premier retourne une `Series`.

Le second retourne un `DataFrame`.

```python
type(df["country"])
```

```python
type(df[["country"]])
```


# 15. Une question métier

Imaginons que notre responsable marketing nous demande :

> "Je veux uniquement connaître le pays, la catégorie et le montant de chaque commande."

Nous devons d'abord créer le montant :

```python
df["amount"] = df["quantity"] * df["unit_price"]
```

Puis sélectionner les colonnes utiles :

```python
df[
    ["country", "category", "quantity", "amount"]
].head()
```


# 16. Sélectionner une ligne avec `iloc`

`iloc` permet de sélectionner par **position**.

Première ligne :

```python
df.iloc[0]
```

Deuxième ligne :

```python
df.iloc[1]
```

Dixième ligne :

```python
df.iloc[9]
```


# 17. Sélectionner plusieurs lignes

Les slices fonctionnent comme avec les listes Python.

```python
df.iloc[0:5]
```

Les cinq premières lignes.

```python
df.iloc[10:20]
```

Les lignes 10 à 19.


## Sélectionner lignes et colonnes

```python
df.iloc[0:5, 0:3]
```

Lecture :

> lignes 0 à 4, colonnes 0 à 2.


# 18. `loc` : sélectionner avec les labels

`loc` travaille avec les labels de l'index et des colonnes.

```python
df.loc[0]
```

Sélectionner certaines colonnes :

```python
df.loc[:, ["country", "category"]]
```

Lecture :

```text
:
↓
toutes les lignes

["country", "category"]
↓
ces deux colonnes
```


# 19. `iloc` vs `loc`

Retenez pour l'instant :

```text
iloc
 ↓
position

loc
 ↓
label
```

Exemple :

```python
df.iloc[0:10]
```

signifie :

> les 10 premières lignes.

Alors que :

```python
df.loc[:, ["country", "category"]]
```

signifie :

> toutes les lignes et les colonnes dont le nom est `country` et `category`.


# 20. Filtrer avec une condition

Supposons que nous voulions uniquement les commandes dont la quantité est supérieure à 5.

Commençons par construire la condition :

```python
df["quantity"] > 5
```

Le résultat est une série de booléens :

```text
True
False
False
True
...
```


# 21. Utiliser la condition comme filtre

```python
df[df["quantity"] > 5]
```

Nous avons créé un **masque booléen**.

Conceptuellement :

```text
DataFrame
    │
    ↓
condition
    │
    ↓
True / False
    │
    ↓
lignes conservées
```

C'est l'un des mécanismes fondamentaux de Pandas.


# 22. Filtrer sur une variable numérique

Commandes supérieures à 100 € :

```python
df[df["amount"] > 100]
```

Commandes dont la quantité est égale à 1 :

```python
df[df["quantity"] == 1]
```

Commandes dont la quantité est différente de 1 :

```python
df[df["quantity"] != 1]
```


# 23. Filtrer sur du texte

Toutes les commandes provenant de France :

```python
df[df["country"] == "France"]
```

Toutes les commandes provenant d'Allemagne :

```python
df[df["country"] == "Germany"]
```


# 24. Plusieurs conditions

Supposons que nous voulions :

> les commandes françaises de plus de 100 €.

```python
df[
    (df["country"] == "France")
    & (df["amount"] > 100)
]
```


## Les opérateurs importants

Avec Pandas, on utilise :

| Python logique | Pandas |
| -------------- | ------ |
| `and`          | `&`    |
| `or`           | `\|`   |
| `not`          | `~`    |

Et il faut généralement entourer chaque condition de parenthèses.


# 25. `OR`

Commandes provenant de France **ou** d'Allemagne :

```python
df[
    (df["country"] == "France")
    | (df["country"] == "Germany")
]
```


# 26. `isin()`

Lorsque nous voulons tester plusieurs valeurs, `isin()` est souvent plus lisible.

```python
df[
    df["country"].isin(
        ["France", "Germany", "Spain"]
    )
]
```

Cela signifie :

> conserver les lignes dont le pays appartient à cette liste.


# 27. `between()`

Pour sélectionner une plage numérique :

```python
df[
    df["amount"].between(50, 100)
]
```

Cela sélectionne les commandes dont le montant est compris entre 50 et 100.


# 28. `query()`

Pandas fournit également une syntaxe plus proche d'une requête.

```python
df.query("amount > 100")
```

Plusieurs conditions :

```python
df.query(
    "amount > 100 and quantity >= 2"
)
```


# 29. Comparaison des approches

Les trois approches suivantes peuvent être utilisées :

```python
df[df["amount"] > 100]
```

```python
df[df["amount"].between(100, 1000)]
```

```python
df.query("amount > 100")
```

Il ne s'agit pas de choisir une syntaxe "magique".

L'objectif est de savoir **quelle expression est la plus claire dans le contexte**.


# 30. Explorer une variable catégorielle

Imaginons que nous voulions connaître les pays présents dans notre dataset.

```python
df["country"].unique()
```


## Combien de pays différents ?

```python
df["country"].nunique()
```


## Combien de commandes par pays ?

```python
df["country"].value_counts()
```


# 31. `value_counts()` : une commande essentielle

Pour une variable catégorielle :

```python
df["category"].value_counts()
```

Nous obtenons par exemple :

```text
Electronics    3210
Clothing       2870
Home           2340
Books          1580
```

Cela permet immédiatement de voir la distribution des catégories.


# 32. Obtenir des proportions

On peut demander les fréquences relatives :

```python
df["category"].value_counts(
    normalize=True
)
```

Par exemple :

```text
Electronics    0.321
Clothing       0.287
Home           0.234
Books          0.158
```

Soit :

```text
32.1 %
28.7 %
23.4 %
15.8 %
```


# 33. Explorer une variable numérique

Pour une variable numérique :

```python
df["amount"].describe()
```

On obtient notamment :

```text
count
mean
std
min
25%
50%
75%
max
```


# 34. Comprendre les quartiles

Supposons :

```text
25%    = 32 €
50%    = 61 €
75%    = 120 €
```

La médiane est :

```text
50% → 61 €
```

Cela signifie que 50 % des observations sont inférieures ou égales à 61 €.

Et 75 % sont inférieures ou égales à 120 €.


# 35. `describe()` sur tout le DataFrame

```python
df.describe()
```

Pandas sélectionne automatiquement les variables numériques.


# 36. Statistiques catégorielles

Pour obtenir également des informations sur les colonnes non numériques :

```python
df.describe(include="all")
```

Cette commande peut être très utile lors d'une première exploration.


# 37. Compter les valeurs manquantes

Une question essentielle :

> **Combien de valeurs sont absentes ?**

```python
df.isna().sum()
```


# 38. Calculer le taux de valeurs manquantes

```python
df.isna().mean() * 100
```

Nous obtenons alors un pourcentage par colonne.

Par exemple :

```text
order_id            0.0 %
customer_id         0.0 %
order_date          0.0 %
country             0.2 %
category            1.4 %
customer_rating     8.7 %
```


# 39. Pourquoi les valeurs manquantes sont-elles importantes ?

Supposons :

```text
customer_rating
----------------
5
4
3
NaN
5
NaN
```

Les `NaN` ne signifient pas nécessairement :

> "Le client a donné une note de 0."

Ils signifient :

> "Nous n'avons pas de valeur."

Cette distinction est fondamentale en Data Science.


# 40. Vérifier les doublons

Nous pouvons également rechercher les lignes dupliquées :

```python
df.duplicated().sum()
```


# 41. Première exploration statistique

Nous pouvons maintenant commencer à poser des questions métier.

### Quel est le montant moyen d'une commande ?

```python
df["amount"].mean()
```

### Quel est le montant médian ?

```python
df["amount"].median()
```

### Quel est le montant maximum ?

```python
df["amount"].max()
```

### Quelle est la quantité moyenne ?

```python
df["quantity"].mean()
```


# 42. Comparer moyenne et médiane

Calculons :

```python
mean_amount = df["amount"].mean()
median_amount = df["amount"].median()

print("Moyenne :", mean_amount)
print("Médiane :", median_amount)
```


## 🧠 Question

Que peut-on déduire si :

```text
moyenne  = 142 €
médiane  = 72 €
```

La moyenne est fortement supérieure à la médiane.

Cela peut être un indice de la présence de **valeurs élevées** qui tirent la moyenne vers le haut.


# 43. Explorer les valeurs extrêmes

Minimum :

```python
df["amount"].min()
```

Maximum :

```python
df["amount"].max()
```

Les quantiles :

```python
df["amount"].quantile(
    [0.01, 0.25, 0.50, 0.75, 0.99]
)
```


# 44. Identifier les commandes les plus importantes

Les 10 commandes les plus élevées :

```python
df.nlargest(
    10,
    "amount"
)
```

Les 10 plus petites :

```python
df.nsmallest(
    10,
    "amount"
)
```


# 45. Une première question métier

Imaginons que le directeur commercial demande :

> "Quelles sont les 10 commandes qui représentent les plus gros montants ?"

Nous pouvons répondre avec :

```python
df.nlargest(10, "amount")[
    [
        "order_id",
        "country",
        "category",
        "quantity",
        "amount"
    ]
]
```

Nous commençons déjà à transformer une question métier en opération Pandas.


# 46. Explorer les méthodes de paiement

```python
df["payment_method"].value_counts()
```

Puis :

```python
df["payment_method"].value_counts(
    normalize=True
)
```


# 47. Explorer les notes clients

```python
df["customer_rating"].value_counts()
```

Attention :

`customer_rating` est une variable numérique, mais elle peut également être considérée comme une variable **ordinale**.

Une note de :

```text
1 < 2 < 3 < 4 < 5
```

a un ordre.

Mais la différence entre 1 et 2 n'est pas nécessairement équivalente à celle entre 4 et 5.


# 48. Une première analyse croisée

Nous pouvons demander :

> Combien de commandes avons-nous par pays et par catégorie ?

```python
df[
    ["country", "category"]
].value_counts()
```

Nous commençons ici à croiser plusieurs variables.

Nous verrons plus tard comment effectuer des analyses beaucoup plus puissantes avec `groupby()`.


# 49. Un autre exemple

Combien de commandes françaises utilisent chaque moyen de paiement ?

```python
df[
    df["country"] == "France"
]["payment_method"].value_counts()
```

Décomposons mentalement :

```text
df
 │
 ↓
filtrer France
 │
 ↓
sélectionner payment_method
 │
 ↓
compter les valeurs
```

C'est une manière très importante de penser Pandas :

> **enchaîner des opérations simples pour répondre à une question métier.**


# 50. Le chaînage d'opérations

Par exemple :

```python
df[
    df["country"] == "France"
]["amount"].mean()
```

Lecture :

```text
DataFrame
   ↓
commandes françaises
   ↓
colonne amount
   ↓
moyenne
```


# 51. Un exemple plus complexe

Question :

> Quel est le montant moyen des commandes françaises contenant au moins 2 produits ?

```python
df[
    (df["country"] == "France")
    & (df["quantity"] >= 2)
]["amount"].mean()
```

Nous combinons :

* filtrage ;
* conditions multiples ;
* sélection de colonne ;
* agrégation.


# 52. Construire une checklist d'exploration

Face à un nouveau dataset, vous pouvez utiliser cette checklist.

### Structure

```python
df.shape
df.columns
df.head()
df.tail()
```

### Types

```python
df.dtypes
df.info()
```

### Statistiques

```python
df.describe()
```

### Variables catégorielles

```python
df["country"].unique()

```

```python
df["country"].nunique()
```

```python
df["country"].value_counts()
```

### Valeurs manquantes

```python
df.isna().sum()
```

### Doublons

```python
df.duplicated().sum()
```

### Valeurs extrêmes

```python
df["unit_price"].min()
```


```python
df["unit_price"].max()
```

