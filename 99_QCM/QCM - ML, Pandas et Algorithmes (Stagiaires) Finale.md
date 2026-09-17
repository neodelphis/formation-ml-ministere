# **QCM : Machine Learning, Algorithmes et Pandas**

Ce questionnaire de 40 questions évalue vos connaissances sur la préparation des données, les concepts fondamentaux du Machine Learning et les différents algorithmes étudiés.

## **Partie 1 : Introduction et Manipulation avec Pandas**

> 1. **Qu'est-ce que le Machine Learning ?**  
   * A. Une méthode permettant à un algorithme d'apprendre des modèles à partir de données.  
   * B. Un mixeur haute technologie pour la cuisine.  
   * C. Une série de règles conditionnelles écrites manuellement par un développeur.  
> 2. **Quelle est la différence entre l'apprentissage supervisé et non supervisé ?**  
   * A. L'apprentissage supervisé a un chef de projet, le non supervisé est en roue libre.  
   * B. L'apprentissage supervisé utilise des données étiquetées (avec la réponse), le non supervisé n'en a pas.  
   * C. L'apprentissage supervisé est toujours beaucoup plus rapide.  
> 3. **À quoi sert le jeu d'entraînement (train set) ?**  
   * A. À muscler le processeur de l'ordinateur.  
   * B. À évaluer la performance finale du modèle.  
   * C. À fournir les exemples sur lesquels l'algorithme va apprendre.  
> 4. **Qu'est-ce que le surapprentissage (overfitting) ?**  
   * A. Le fait que l'ordinateur chauffe trop pendant les calculs.  
   * B. Quand le modèle apprend par cœur les données d'entraînement et échoue sur de nouvelles données.  
   * C. Quand le modèle est beaucoup trop simple pour comprendre la logique des données.  
> 5. **Dans un DataFrame Pandas, que représente généralement une ligne ?**  
   * A. Une observation ou un individu unique dans le jeu de données.  
   * B. Une caractéristique ou une variable.  
   * C. Une ligne de code Python très compliquée.  
> 6. **Comment gère-t-on le plus souvent les valeurs manquantes (NaN) avec Pandas ?**  
   * A. On met du blanc correcteur sur l'écran pour ne plus les voir.  
   * B. On supprime les lignes concernées ou on remplace la valeur manquante (par exemple par la moyenne).  
   * C. On laisse l'algorithme deviner au hasard.  
> 7. **Que fait la méthode head() lorsqu'elle est appliquée à un DataFrame Pandas ?**  
   * A. Elle affiche un aperçu des premières lignes du tableau.  
   * B. Elle met un chapeau virtuel sur la première colonne du fichier.  
   * C. Elle supprime toutes les données pour ne garder que le titre.  
> 8. **Lequel de ces problèmes est un cas de classification ?**  
   * A. Prédire le prix d'une maison en euros.  
   * B. Prédire si un email reçu est un "Spam" ou "Non Spam".  
   * C. Ranger les câbles de l'ordinateur par ordre de taille.  
> 9. **Lequel de ces problèmes est un cas de régression ?**  
   * A. Prédire la température de demain en degrés.  
   * B. Identifier si une image contient un chat ou un chien.  
   * C. Prédire son avenir amoureux en lisant dans les cartes.  
> 10. **À quoi sert la méthode "groupby" dans Pandas ?**  
    * A. À créer un groupe WhatsApp avec les membres de l'équipe de données.  
    * B. À mélanger aléatoirement toutes les lignes du tableau.  
    * C. À regrouper les données selon une catégorie pour calculer des statistiques agrégées.

## **Partie 2 : Distances, kNN et K-Means**

> 11. **Sur quoi l'algorithme kNN (k-Nearest Neighbors) base-t-il sa prédiction ?**  
    * A. Les prévisions météo du jour.  
    * B. La classe majoritaire parmi ses 'k' voisins les plus proches.  
    * C. Une équation mathématique créant une ligne droite.  
> 12. **Dans l'algorithme kNN, que représente le paramètre "k" ?**  
    * A. Le nombre de voisins à prendre en compte pour prendre une décision.  
    * B. Le poids du fichier de données en kilo-octets.  
    * C. La lettre initiale de son créateur.  
> 13. **Que se passe-t-il dans kNN si l'on choisit k=1 ?**  
    * A. Le modèle prend simplement la classe de l'unique point d'entraînement le plus proche.  
    * B. L'algorithme plante irrémédiablement.  
    * C. Il demande l'avis du public.  
> 14. **Lequel de ces algorithmes est un modèle d'apprentissage NON supervisé ?**  
    * A. La régression linéaire.  
    * B. K-Means (les K-Moyennes).  
    * C. La règle de trois.  
> 15. **Quel est l'objectif principal de l'algorithme K-Means ?**  
    * A. Prédire le prix d'un article.  
    * B. Trouver le centre géométrique de la Terre.  
    * C. Regrouper les données en 'K' clusters (groupes) homogènes en fonction de leurs similarités.  
> 16. **Que représente un "centroïde" dans K-Means ?**  
    * A. Un droïde astromech dans Star Wars.  
    * B. Le point central virtuel (la moyenne) d'un cluster.  
    * C. Le point le plus éloigné du jeu de données.  
> 17. **Comment l'algorithme K-Means assigne-t-il un point à un cluster ?**  
    * A. En l'assignant au centroïde dont la distance est la plus courte.  
    * B. En tirant à pile ou face.  
    * C. En demandant à l'utilisateur de cliquer dessus.  
> 18. **Est-il important de mettre les données à la même échelle (scaling) avant d'utiliser kNN ou K-Means ?**  
    * A. Seulement les jours de pleine lune.  
    * B. Non, cela abîme les données d'origine.  
    * C. Oui, car ces algorithmes basent leurs décisions sur des calculs de distances.

## **Partie 3 : Arbres de Décision et Forêts Aléatoires (Random Forests)**

> 19. **Comment fonctionne le principe de base d'un arbre de décision ?**  
    * A. Il fait pousser des branches virtuelles en 3D sur l'écran.  
    * B. Il pose une série de questions (règles si/alors) pour séparer progressivement les données.  
    * C. Il trace une droite qui coupe l'écran en deux.  
> 20. **Qu'est-ce qu'une "feuille" dans le contexte d'un arbre de décision ?**  
    * A. Un élément décoratif pour rendre l'algorithme plus vert.  
    * B. Le point de départ de l'algorithme.  
    * C. Le nœud final qui donne la décision ou prédiction.  
> 21. **Quel est le principal risque d'un arbre de décision très profond (beaucoup de conditions) ?**  
    * A. Le sous-apprentissage.  
    * B. Qu'il perde ses feuilles en automne.  
    * C. Le surapprentissage (overfitting) : il apprend par cœur les données d'entraînement.  
> 22. **Qu'est-ce qu'une "Random Forest" (Forêt Aléatoire) ?**  
    * A. Un ensemble (une forêt) de plusieurs arbres de décision qui votent ensemble.  
    * B. Un fond d'écran Windows populaire dans les années 2000\.  
    * C. Un seul arbre géant et très complexe.  
> 23. **Pourquoi utiliser une Random Forest plutôt qu'un seul arbre de décision ?**  
    * A. Pour économiser de l'électricité.  
    * B. Parce que cela sonne plus écologique lors des réunions.  
    * C. Pour obtenir un modèle plus robuste et réduire les risques de surapprentissage.  
> 24. **Comment la Random Forest fait-elle sa prédiction finale en classification ?**  
    * A. Elle choisit l'arbre le plus grand du modèle.  
    * B. Par un vote majoritaire de tous les arbres qui la composent.  
    * C. Elle demande conseil à un garde forestier.  
> 25. **Que signifie le terme "Random" (aléatoire) dans Random Forest ?**  
    * A. Les prédictions finales sont tirées au sort.  
    * B. Le programme s'exécute à des heures aléatoires de la journée.  
    * C. Chaque arbre est entraîné sur un échantillon aléatoire de données et avec un sous-ensemble aléatoire de caractéristiques.  
> 26. **Pour quels types de tâches peut-on utiliser une Random Forest ?**  
    * A. Uniquement pour la classification d'images d'arbres.  
    * B. Aussi bien pour des tâches de classification que de régression.  
    * C. Uniquement si les données sont au format texte.

## **Partie 4 : Réseaux Neuronaux et Évaluation des Modèles**

> 27. **De quoi s'inspirent originellement les réseaux neuronaux artificiels ?**  
    * A. Du réseau routier national.  
    * B. Du fonctionnement interconnecté des neurones du cerveau humain.  
    * C. Des toiles d'araignées dans la salle des serveurs.  
> 28. **Qu'est-ce qu'un "perceptron" ?**  
    * A. Un super-héros issu de l'univers Marvel.  
    * B. Un nouvel outil de nettoyage de base de données.  
    * C. L'unité de base, ou neurone artificiel, constituant un réseau.  
> 29. **À quoi servent les "couches cachées" (hidden layers) dans un réseau de neurones ?**  
    * A. À cacher les données sensibles pour respecter le RGPD.  
    * B. À extraire des motifs complexes et non linéaires dans les données.  
    * C. À jouer à cache-cache avec le processeur.  
> 30. **Qu'est-ce qu'une "fonction d'activation" dans un réseau de neurones ?**  
    * A. Une fonction mathématique qui détermine si le neurone doit transmettre son signal.  
    * B. Un bouton rouge sur le clavier pour lancer l'entraînement.  
    * C. Un script qui active l'antivirus de l'ordinateur.  
> 31. **Qu'est-ce que le "Deep Learning" (Apprentissage profond) ?**  
    * A. Un modèle qui explore les abysses des océans.  
    * B. Un algorithme qui réfléchit très profondément avant de répondre.  
    * C. L'utilisation de réseaux de neurones possédant un grand nombre de couches cachées.  
> 32. **Quel algorithme est classiquement utilisé pour ajuster les poids d'un réseau de neurones et réduire son erreur ?**  
    * A. Le retour vers le futur.  
    * B. Le grand nettoyage de printemps.  
    * C. La rétropropagation du gradient (Backpropagation).  
> 33. **Pour quels domaines les réseaux neuronaux profonds excellent-ils particulièrement ?**  
    * A. Trier un petit fichier Excel de 10 lignes.  
    * B. La préparation des commandes dans un fast-food.  
    * C. La reconnaissance d'images, le traitement du langage naturel et l'audio.  
> 34. **Qu'est-ce qu'une matrice de confusion ?**  
    * A. Un tableau qui permet d'évaluer les performances d'un modèle en croisant prédictions et réalités.  
    * B. Un labyrinthe mathématique dont personne ne sort.  
    * C. L'expression d'un développeur face à un bug.  
> 35. **Dans l'équation de régression Y \= aX \+ b, que représente "a" ?**  
    * A. La longueur du fichier.  
    * B. Le poids ou coefficient donné à la variable X.  
    * C. Le mot de passe de l'ordinateur.  
> 36. **Qu'est-ce qu'un Faux Positif (False Positive) ?**  
    * A. Le modèle a prédit que l'événement allait arriver, mais il ne s'est rien passé.  
    * B. Un sourire hypocrite sur une photo.  
    * C. Le modèle a correctement prédit que l'événement ne se produirait pas.  
> 37. **Pourquoi la métrique de l'accuracy (précision globale) peut-elle être mauvaise ?**  
    * A. Elle change selon la météo.  
    * B. Si les classes sont très déséquilibrées (ex: 99% d'exemples normaux et 1% d'anomalies).  
    * C. Parce qu'elle est calculée par un dé à six faces.  
> 38. **Pourquoi est-il indispensable de faire une analyse exploratoire (EDA) avant de modéliser ?**  
    * A. Pour comprendre la distribution des données, repérer les erreurs et choisir les bonnes variables.  
    * B. Pour faire du tourisme virtuel dans les serveurs de l'entreprise.  
    * C. Pour inventer des données qui donneront un meilleur résultat final.  
> 39. **Si votre algorithme est anormalement lent à s'entraîner, quelle peut être la raison ?**  
    * A. Les données sont écrites avec une police de caractères trop petite.  
    * B. Il a besoin qu'on lui serve un café serré.  
    * C. Le jeu de données est très volumineux ou le modèle est trop complexe.  
> 40. **En Data Science, que signifie "Garbage In, Garbage Out" ?**  
    * A. Que les IA font le tri sélectif automatiquement.  
    * B. Que des données de mauvaise qualité produiront forcément un mauvais modèle prédictif.  
    * C. Qu'il faut vider la corbeille de l'ordinateur avant de lancer Python.
