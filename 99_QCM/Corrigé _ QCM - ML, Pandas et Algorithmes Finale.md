# **Corrigé du QCM : Machine Learning, Algorithmes et Pandas**

## **Partie 1 : Introduction et Manipulation avec Pandas**

> 1. **Qu'est-ce que le Machine Learning ?**  
   * **Réponse : A. Une méthode permettant à un algorithme d'apprendre des modèles à partir de données.**  
> 2. **Quelle est la différence entre l'apprentissage supervisé et non supervisé ?**  
   * **Réponse : B. L'apprentissage supervisé utilise des données étiquetées (avec la réponse), le non supervisé n'en a pas.**  
> 3. **À quoi sert le jeu d'entraînement (train set) ?**  
   * **Réponse : C. À fournir les exemples sur lesquels l'algorithme va apprendre.**  
> 4. **Qu'est-ce que le surapprentissage (overfitting) ?**  
   * **Réponse : B. Quand le modèle apprend par cœur les données d'entraînement et échoue sur de nouvelles données.**  
> 5. **Dans un DataFrame Pandas, que représente généralement une ligne ?**  
   * **Réponse : A. Une observation ou un individu unique dans le jeu de données.**  
> 6. **Comment gère-t-on le plus souvent les valeurs manquantes (NaN) avec Pandas ?**  
   * **Réponse : B. On supprime les lignes concernées ou on remplace la valeur manquante (par exemple par la moyenne).**  
> 7. **Que fait la méthode head() lorsqu'elle est appliquée à un DataFrame Pandas ?**  
   * **Réponse : A. Elle affiche un aperçu des premières lignes du tableau.**  
> 8. **Lequel de ces problèmes est un cas de classification ?**  
   * **Réponse : B. Prédire si un email reçu est un "Spam" ou "Non Spam".**  
> 9. **Lequel de ces problèmes est un cas de régression ?**  
   * **Réponse : A. Prédire la température de demain en degrés.**  
> 10. **À quoi sert la méthode "groupby" dans Pandas ?**  
    * **Réponse : C. À regrouper les données selon une catégorie pour calculer des statistiques agrégées.**

## **Partie 2 : Distances, kNN et K-Means**

> 11. **Sur quoi l'algorithme kNN (k-Nearest Neighbors) base-t-il sa prédiction ?**  
    * **Réponse : B. La classe majoritaire parmi ses 'k' voisins les plus proches.**  
> 12. **Dans l'algorithme kNN, que représente le paramètre "k" ?**  
    * **Réponse : A. Le nombre de voisins à prendre en compte pour prendre une décision.**  
> 13. **Que se passe-t-il dans kNN si l'on choisit k=1 ?**  
    * **Réponse : A. Le modèle prend simplement la classe de l'unique point d'entraînement le plus proche.**  
> 14. **Lequel de ces algorithmes est un modèle d'apprentissage NON supervisé ?**  
    * **Réponse : B. K-Means (les K-Moyennes).**  
> 15. **Quel est l'objectif principal de l'algorithme K-Means ?**  
    * **Réponse : C. Regrouper les données en 'K' clusters (groupes) homogènes en fonction de leurs similarités.**  
> 16. **Que représente un "centroïde" dans K-Means ?**  
    * **Réponse : B. Le point central virtuel (la moyenne) d'un cluster.**  
> 17. **Comment l'algorithme K-Means assigne-t-il un point à un cluster ?**  
    * **Réponse : A. En l'assignant au centroïde dont la distance est la plus courte.**  
> 18. **Est-il important de mettre les données à la même échelle (scaling) avant d'utiliser kNN ou K-Means ?**  
    * **Réponse : C. Oui, car ces algorithmes basent leurs décisions sur des calculs de distances.**

## **Partie 3 : Arbres de Décision et Forêts Aléatoires (Random Forests)**

> 19. **Comment fonctionne le principe de base d'un arbre de décision ?**  
    * **Réponse : B. Il pose une série de questions (règles si/alors) pour séparer progressivement les données.**  
> 20. **Qu'est-ce qu'une "feuille" dans le contexte d'un arbre de décision ?**  
    * **Réponse : C. Le nœud final qui donne la décision ou prédiction.**  
> 21. **Quel est le principal risque d'un arbre de décision très profond (beaucoup de conditions) ?**  
    * **Réponse : C. Le surapprentissage (overfitting) : il apprend par cœur les données d'entraînement.**  
> 22. **Qu'est-ce qu'une "Random Forest" (Forêt Aléatoire) ?**  
    * **Réponse : A. Un ensemble (une forêt) de plusieurs arbres de décision qui votent ensemble.**  
> 23. **Pourquoi utiliser une Random Forest plutôt qu'un seul arbre de décision ?**  
    * **Réponse : C. Pour obtenir un modèle plus robuste et réduire les risques de surapprentissage.**  
> 24. **Comment la Random Forest fait-elle sa prédiction finale en classification ?**  
    * **Réponse : B. Par un vote majoritaire de tous les arbres qui la composent.**  
> 25. **Que signifie le terme "Random" (aléatoire) dans Random Forest ?**  
    * **Réponse : C. Chaque arbre est entraîné sur un échantillon aléatoire de données et avec un sous-ensemble aléatoire de caractéristiques.**  
> 26. **Pour quels types de tâches peut-on utiliser une Random Forest ?**  
    * **Réponse : B. Aussi bien pour des tâches de classification que de régression.**

## **Partie 4 : Réseaux Neuronaux et Évaluation des Modèles**

> 27. **De quoi s'inspirent originellement les réseaux neuronaux artificiels ?**  
    * **Réponse : B. Du fonctionnement interconnecté des neurones du cerveau humain.**  
> 28. **Qu'est-ce qu'un "perceptron" ?**  
    * **Réponse : C. L'unité de base, ou neurone artificiel, constituant un réseau.**  
> 29. **À quoi servent les "couches cachées" (hidden layers) dans un réseau de neurones ?**  
    * **Réponse : B. À extraire des motifs complexes et non linéaires dans les données.**  
> 30. **Qu'est-ce qu'une "fonction d'activation" dans un réseau de neurones ?**  
    * **Réponse : A. Une fonction mathématique qui détermine si le neurone doit transmettre son signal.**  
> 31. **Qu'est-ce que le "Deep Learning" (Apprentissage profond) ?**  
    * **Réponse : C. L'utilisation de réseaux de neurones possédant un grand nombre de couches cachées.**  
> 32. **Quel algorithme est classiquement utilisé pour ajuster les poids d'un réseau de neurones et réduire son erreur ?**  
    * **Réponse : C. La rétropropagation du gradient (Backpropagation).**  
> 33. **Pour quels domaines les réseaux neuronaux profonds excellent-ils particulièrement ?**  
    * **Réponse : C. La reconnaissance d'images, le traitement du langage naturel et l'audio.**  
> 34. **Qu'est-ce qu'une matrice de confusion ?**  
    * **Réponse : A. Un tableau qui permet d'évaluer les performances d'un modèle en croisant prédictions et réalités.**  
> 35. **Dans l'équation de régression Y \= aX \+ b, que représente "a" ?**  
    * **Réponse : B. Le poids ou coefficient donné à la variable X.**  
> 36. **Qu'est-ce qu'un Faux Positif (False Positive) ?**  
    * **Réponse : A. Le modèle a prédit que l'événement allait arriver, mais il ne s'est rien passé.**  
> 37. **Pourquoi la métrique de l'accuracy (précision globale) peut-elle être mauvaise ?**  
    * **Réponse : B. Si les classes sont très déséquilibrées (ex: 99% d'exemples normaux et 1% d'anomalies).**  
> 38. **Pourquoi est-il indispensable de faire une analyse exploratoire (EDA) avant de modéliser ?**  
    * **Réponse : A. Pour comprendre la distribution des données, repérer les erreurs et choisir les bonnes variables.**  
> 39. **Si votre algorithme est anormalement lent à s'entraîner, quelle peut être la raison ?**  
    * **Réponse : C. Le jeu de données est très volumineux ou le modèle est trop complexe.**  
> 40. **En Data Science, que signifie "Garbage In, Garbage Out" ?**  
    * **Réponse : B. Que des données de mauvaise qualité produiront forcément un mauvais modèle prédictif.**