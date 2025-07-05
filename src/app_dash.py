import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Configuration de base pour les graphiques
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# === Chargement des données ===
DATA_CSV_PATH = "/Users/NOKHO/Desktop/Meteo/historique_meteo_uemoa_80villes_clean.csv"

try:
    df = pd.read_csv(DATA_CSV_PATH, parse_dates=['datetime'])
    print("✅ Données chargées avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement des données: {e}")
    exit()

# Renommage des colonnes pour simplifier
df.rename(columns={
    'datetime': 'Date',
    'temp': 'Température',
    'tempmax': 'Température_max',
    'tempmin': 'Température_min',
    'precip': 'Précipitations',
    'latitude': 'Lat',
    'longitude': 'Lon',
    'Pays': 'Pays',
    'Ville': 'Ville'
}, inplace=True)

# Nettoyage des données
df_clean = df.dropna(subset=['Température', 'Précipitations']).copy()

# === Fonction pour afficher et sauvegarder ===
def show_and_save(plt, filename):
    """Affiche et sauvegarde un graphique"""
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé sous: {filename}")
    plt.show()
    plt.close()

# === Analyse Exploratoire ===
print("\nAperçu des données :")
print(df_clean.head())

print("\nStatistiques descriptives :")
print(df_clean[['Température', 'Précipitations']].describe())

# 1. Distribution des températures
plt.figure()
sns.histplot(df_clean['Température'], bins=30, kde=True, color='royalblue')
plt.title("Distribution des Températures dans l'UEMOA")
plt.xlabel("Température (°C)")
plt.ylabel("Fréquence")
show_and_save(plt, "distribution_temperature.png")

# 2. Températures moyennes par pays
temp_pays = df_clean.groupby('Pays')['Température'].mean().sort_values()

plt.figure()
temp_pays.plot(kind='bar', color=sns.color_palette("coolwarm", len(temp_pays)))
plt.title("Température Moyenne par Pays")
plt.ylabel("Température moyenne (°C)")
plt.xticks(rotation=45)
show_and_save(plt, "temp_moyenne_par_pays.png")

# 3. Matrice de corrélation
corr_matrix = df_clean[['Température', 'Précipitations']].corr()

plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            annot_kws={"size": 12}, fmt=".2f", linewidths=.5)
plt.title("Corrélation entre Température et Précipitations")
show_and_save(plt, "correlation_matrix.png")

# === Analyse Statistique ===
pearson_corr, p_value = stats.pearsonr(df_clean['Température'], df_clean['Précipitations'])
print(f"\nTest de corrélation de Pearson:")
print(f"Corrélation: {pearson_corr:.3f}, p-value: {p_value:.4f}")

# === Machine Learning ===
# 1. Clustering KMeans
X_cluster = df_clean[['Température', 'Précipitations']].values

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_cluster)

plt.figure()
sns.scatterplot(x=df_clean['Température'], y=df_clean['Précipitations'], 
                hue=clusters, palette='viridis', s=100, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='red', marker='X', label='Centroïdes')
plt.title("Clustering des Conditions Météorologiques (KMeans k=3)")
plt.xlabel("Température (°C)")
plt.ylabel("Précipitations (mm)")
plt.legend(title='Cluster')
show_and_save(plt, "kmeans_clusters.png")

# 2. Régression Linéaire
X_reg = df_clean[['Précipitations']].values
y_reg = df_clean['Température_max'].values

reg = LinearRegression()
reg.fit(X_reg, y_reg)
y_pred = reg.predict(X_reg)

plt.figure()
plt.scatter(X_reg, y_reg, color='royalblue', alpha=0.5, label='Données réelles')
plt.plot(X_reg, y_pred, color='red', linewidth=2, label='Modèle de régression')
plt.title("Régression: Température Max vs Précipitations")
plt.xlabel("Précipitations (mm)")
plt.ylabel("Température Max (°C)")
plt.legend()
show_and_save(plt, "regression_temp_precip.png")

print("\n✅ Analyse terminée avec succès!")
