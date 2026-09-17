# Fixer la graine aléatoire pour la reproductibilité
np.random.seed(42)

n_samples = 2000

# Génération des caractéristiques environnementales
pH = np.random.uniform(5.5, 9.0, n_samples)
oxygen = np.random.uniform(1.0, 12.0, n_samples)
turbidity = np.random.uniform(0, 100, n_samples)
temperature = np.random.uniform(5, 30, n_samples)
nitrates = np.random.uniform(0, 50, n_samples)
conductivity = np.random.uniform(100, 1500, n_samples)

# Construction de la matrice X
X = np.column_stack([
    pH,
    oxygen,
    turbidity,
    temperature,
    nitrates,
    conductivity
])

# ---------------------------------------------------------
# Construction de règles environnementales non linéaires
# ---------------------------------------------------------

# Condition 1 :
# faible oxygène ET forte turbidité
condition_1 = (
    (oxygen < 5)
    & (turbidity > 60)
)

# Condition 2 :
# forte concentration en nitrates ET forte conductivité
condition_2 = (
    (nitrates > 30)
    & (conductivity > 900)
)

# Condition 3 :
# température élevée ET oxygène insuffisant
condition_3 = (
    (temperature > 23)
    & (oxygen < 6)
)

# Interaction non linéaire entre pH et température
condition_4 = (
    (pH < 6.5)
    & (temperature > 20)
)

# Combinaison des différentes situations
pollution_score = (
    condition_1.astype(int)
    + condition_2.astype(int)
    + condition_3.astype(int)
    + condition_4.astype(int)
)

# Ajout d'une logique non linéaire :
# certaines combinaisons de facteurs peuvent compenser
# partiellement le risque.
interaction = (
    (oxygen > 8)
    & (turbidity < 30)
    & (nitrates < 15)
)

pollution_score[interaction] -= 1

# Classification finale
y = (pollution_score >= 1).astype(int)

# Ajout d'un faible bruit de mesure
noise_probability = 0.05
noise_mask = np.random.random(n_samples) < noise_probability
y[noise_mask] = 1 - y[noise_mask]

print(f"Nombre d'observations : {len(X)}")
print(f"Nombre de cas conformes : {(y == 1).sum()}")
print(f"Nombre de cas à risque : {(y == 0).sum()}")