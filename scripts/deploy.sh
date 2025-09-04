#!/bin/bash

# Script de déploiement MLOps

set -e  # Arrêter en cas d'erreur

echo "🚀 Déploiement du modèle Housing Price Prediction"

# Variables
ENV=${1:-staging}
MODEL_NAME="housing-model"
VERSION="1.0.0"

deploy_to_mlflow() {
    echo "📦 Déploiement vers MLflow..."
    
    # Enregistrer le modèle
    mlflow models register-model \
        --model-uri "runs:/${RUN_ID}/model" \
        --name "${MODEL_NAME}" \
        --await-registration-for 300
    
    # Transition vers staging/production
    if [ "$ENV" = "production" ]; then
        mlflow models transition-stage \
            --name "${MODEL_NAME}" \
            --version "${VERSION}" \
            --stage "Production" \
            --archive-existing-versions
    else
        mlflow models transition-stage \
            --name "${MODEL_NAME}" \
            --version "${VERSION}" \
            --stage "Staging" \
            --archive-existing-versions
    fi
}

# Exécution principale
main() {
    echo "📍 Environnement: $ENV"
    
    # Déployer selon l'environnement
    case $ENV in
        "staging")
            deploy_to_mlflow
            ;;
        "production")
            deploy_to_mlflow
            ;;
        *)
            echo "❌ Environnement non reconnu: $ENV"
            exit 1
            ;;
    esac
    
    echo "✅ Déploiement terminé avec succès!"
}

main
