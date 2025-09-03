# ML-Housing-Project

# 🏠 Housing Price Prediction - MLOps Project

Ce projet implémente un pipeline MLOps complet pour la prédiction des prix immobiliers, avec déploiement multi-cloud (AWS, GCP, Azure).

## 🎯 Objectifs

- Entraînement de modèles ML avec tracking MLflow
- Déploiement automatisé sur plusieurs clouds
- Pipeline CI/CD complet
- Monitoring et observabilité

## 🚀 Quickstart

### 1. Setup de l'environnement

```bash
# Cloner le repository
git clone https://github.com/votre-username/housing-mlops-project.git
cd housing-mlops-project

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
pip install -e .
