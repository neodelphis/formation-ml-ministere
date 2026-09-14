Voici une note explicative pas à pas, conçue spécialement pour vous accompagner dans la mise en place de votre environnement de travail pour le Machine Learning.


# 📝 Guide d'installation et de configuration de Conda (Environnement Linux)

Bienvenue dans cette formation en Machine Learning !

Pour travailler efficacement et éviter les conflits entre les différentes versions des bibliothèques (comme Scikit-Learn, Pandas ou XGBoost), nous allons utiliser **Conda**. Conda est un gestionnaire d'environnements virtuels : il permet de créer une "bulle" isolée sur votre ordinateur, contenant exactement les bons outils et les bonnes versions dont nous aurons besoin.

Voici les étapes à suivre pour installer Conda, configurer votre environnement, et le relier à vos outils de travail (VS Code et Jupyter Notebook).


## Étape 1 : Installer Miniconda sur Linux

Miniconda est une version légère de Conda, parfaite pour commencer sans surcharger votre machine.

1. **Ouvrez votre terminal Linux** (le raccourci par défaut est souvent `Ctrl + Alt + T`).
2. **Téléchargez le script d'installation** en tapant cette commande :
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```


3. **Lancez l'installation** avec la commande suivante :
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```


4. **Suivez les instructions à l'écran** :
* Appuyez sur `Entrée` pour lire la licence.
* Tapez `yes` pour accepter les conditions.
* Appuyez sur `Entrée` pour confirmer l'emplacement d'installation par défaut.
* Lorsqu'on vous demande si vous souhaitez initialiser Conda ("*Do you wish the installer to initialize Miniconda3 by running conda init?*"), tapez **`yes`**.


5. **Redémarrez votre terminal** (fermez-le puis rouvrez-le) pour que les changements prennent effet. Vous devriez voir un petit `(base)` apparaître au début de votre ligne de commande.


## Étape 2 : Créer l'environnement de Machine Learning

Nous vous avons fourni un fichier nommé exactement **`environment.yml`**. Ce fichier est une "recette" qui indique à Conda toutes les bibliothèques à installer.
Par exemple, ce fichier créera un environnement nommé `ml` et téléchargera des outils essentiels via Conda et `pip`, comme `scikit-learn`, `pandas`...

1. Dans votre terminal, déplacez-vous dans le dossier où vous avez enregistré le fichier **`environment.yml`**. (Utilisez la commande `cd chemin/vers/le/dossier`).
2. Lancez la création de l'environnement avec cette commande :```bash
conda env create -f environment.yml
```


*Note : Cette étape peut prendre quelques minutes, car Conda va télécharger et installer de nombreux paquets mathématiques et de Machine Learning.*
3. Une fois l'installation terminée, **activez votre nouvel environnement** :```bash
conda activate ml
```


Votre terminal devrait maintenant afficher `(ml)` au lieu de `(base)`. Félicitations, votre espace de travail est prêt !


## Étape 3 : Utiliser votre environnement dans VS Code

Visual Studio Code (VS Code) sera votre outil principal. Voici comment lui dire d'utiliser l'environnement que vous venez de créer :

1. Ouvrez VS Code.
2. Ouvrez le dossier contenant vos scripts ou vos projets de Machine Learning.
3. Créez un fichier Python (terminant par `.py`) nommé `hello_ml.py` puis un notebook (terminant par `.ipynb`) nommé `hello_jupyter.ipynb` (cf plus bas pour le contenu)
4. Ouvrez la palette de commandes en appuyant sur `Ctrl + Maj + P`.
5. Tapez et sélectionnez **`Python: Select Interpreter`** (Sélectionner l'interpréteur Python).
6. Dans la liste qui apparaît, cherchez et cliquez sur celui qui mentionne **`Python 3.13.1 ('ml': conda)`** (car Python 3.13.1 est installé dans cet environnement).



Désormais, lorsque vous exécuterez du code dans VS Code, il utilisera toutes les bibliothèques de Machine Learning de notre environnement `ml`.


### Exemple de script python
Ce script d'initiation utilise une régression linéaire basique pour illustrer la mécanique fondamentale du Machine Learning : fournir des données d'exemple, entraîner l'algorithme, puis lui demander une nouvelle prédiction. Vos apprenants peuvent le sauvegarder dans un fichier nommé `hello_ml.py` pour valider de manière ludique que leur environnement fonctionne parfaitement.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

print("⏳ Chargement des données et création du modèle...")
surfaces = np.array([[20], [30], [40], [50], [60]])
prix = np.array([40, 60, 80, 100, 120]) 
modele = LinearRegression()
modele.fit(surfaces, prix)
nouvelle_surface = np.array([[75]])
prix_predit = modele.predict(nouvelle_surface)

# 4. Affichage du résultat
print("\n🌍 Hello World du Machine Learning !")
print(f"Pour un appartement de {nouvelle_surface[0][0]} m², le modèle a déduit un prix de {prix_predit[0]:.2f} k€.")
```

**Instructions d'exécution pour les apprenants**

* Enregistrez le code ci-dessus dans un fichier `hello_ml.py`.
* Ouvrez le terminal dans le même dossier que ce fichier.
* Vérifiez que l'environnement Conda est bien activé (le préfixe `(ml)` doit être visible sur la ligne de commande).
* Lancez le script en tapant : `python hello_ml.py`

Le modèle déduira logiquement la règle mathématique sous-jacente par lui-même et affichera la bonne réponse (150.00 k€).

## Étape 4 : Utiliser Jupyter Notebook dans le navigateur

Si vous préférez travailler ponctuellement de manière interactive directement dans votre navigateur web, vous pouvez utiliser Jupyter Notebook (qui est bien inclus dans votre fichier **`environment.yml`**).

1. Ouvrez votre terminal.
2. Assurez-vous que votre environnement est activé (le préfixe `(ml)` doit être visible). Si ce n'est pas le cas, tapez `conda activate ml`.
3. Naviguez vers le dossier de votre projet.
4. Lancez Jupyter avec la commande :

```bash
jupyter notebook
```


5. Votre navigateur web va s'ouvrir automatiquement sur l'interface de Jupyter.
6. Lorsque vous créez un nouveau Notebook, vérifiez en haut à droite que le "Kernel" (le noyau) sélectionné est bien lié à votre environnement Python (`Python 3 (ipykernel)`).


### 💡 Quelques commandes utiles pour le quotidien :

* **Désactiver l'environnement** (pour revenir à l'état normal de votre terminal) :`
 
``bash
conda deactivate
```


* **Lister tous vos environnements Conda** :

```bash
conda env list
```


### Exemple de notebook jupyter

Comment alterner entre du texte formaté (Markdown), du code Python, et la visualisation d'un graphique.

#### Comment préparer ce Notebook pour vos apprenants ?

Ouvrir le fichier nommé **`hello_jupyter.ipynb`**, puis de créer les cellules suivantes une par une.


#### 📝 Cellule 1 : Markdown (Texte explicatif)

*Créer une cellule de type **Markdown**, et coller le texte suivant, puis de l'exécuter avec `Maj + Entrée` (Shift + Enter) pour voir le rendu visuel.*

```markdown
# 🌍 Hello World : Mon Premier Notebook

Bienvenue dans Jupyter Notebook ! 

Cet outil est le meilleur ami du Data Scientist. Il permet de mélanger dans un même document :
1. Du **texte explicatif** (comme cette cellule, écrite en Markdown)
2. Du **code Python exécutable**
3. Des **résultats visuels** (tableaux, graphiques)

Pour exécuter une cellule et passer à la suivante, sélectionnez-la et appuyez sur **`Maj + Entrée`** (Shift + Enter).
```

#### 💻 Cellule 2 : Code (Python basique)

*Ajouter une cellule de type **Code**, coller ce script et l'exécuter.*

```python
# Ceci est une cellule de code. Exécutez-la !
prenom = input("Quel est votre prénom ? ")
print(f"Bonjour {prenom}, et bienvenue dans le monde du Machine Learning ! 🚀")

```

#### 📝 Cellule 3 : Markdown (Introduction aux graphiques)

*Une nouvelle cellule **Markdown** pour introduire la suite.*

```markdown
## 📊 Générer un graphique

En Machine Learning, on a constamment besoin de visualiser nos données pour les comprendre. 
Grâce aux bibliothèques `numpy` et `matplotlib` (qui sont bien installées dans notre environnement Conda), nous pouvons tracer des graphiques très facilement.

Exécutez la cellule ci-dessous pour voir le résultat.

```

### 💻 Cellule 4 : Code (Génération du graphique)

*La dernière cellule de **Code** pour générer le graphique.*

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Création de quelques données d'exemple
# Imaginons la progression de vos connaissances au fil des jours de formation
jours = np.array([1, 2, 3, 4, 5])
connaissances = np.array([10, 30, 60, 85, 100])

# 2. Création du graphique
plt.figure(figsize=(8, 4))
plt.plot(jours, connaissances, marker='o', color='teal', linestyle='--', linewidth=2, markersize=8)

# 3. Personnalisation (titre, axes, grille)
plt.title("Évolution de mes connaissances en Machine Learning 🧠", fontsize=14)
plt.xlabel("Jours de formation", fontsize=12)
plt.ylabel("Niveau de connaissance (%)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)

# 4. Affichage
plt.show()

```
