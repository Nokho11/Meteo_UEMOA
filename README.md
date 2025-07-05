# Éditez le README.md
cat > README.md << 'EOF'
# 📊 Dashboard Météo UEMOA

## Pipeline de traitement

1. **Collecte des données** (`etl/collect.py`)
2. **Transformation** (`etl/transform.py`)
3. **Chargement** (`etl/load.py`)
4. **Automatisation** (`airflow/dags/pipeline_meteo_uemoa.py`)

## Analyse de données

- **EDA** (`analysis/analyse.py`)
- **Statistiques avancées**
- **Machine Learning** (Clustering KMeans, Régression linéaire)

## Dashboard interactif

Fonctionnalités :
- Visualisation températures/précipitations/humidité
- Filtres par pays/ville/dates
- Export CSV

Code : `visualization/app.py`

## Technologies

- Python (Pandas, NumPy, Seaborn, Scikit-learn)
- Dash/Plotly
- PostgreSQL
- Airflow
- VS Code/Jupyter/Git

## Licence

Licence MIT - Conserver mention copyright

## Auteur

**Nokho** - 2025  
Contact : `nokhondeyesokhna11@gmail.com`
EOF