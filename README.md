# Système de Prédiction de Qualité de Vin

Ce projet démontre l'implémentation d'un pipeline de machine learning complet pour prédire la qualité du vin. Il utilise MLflow pour la gestion des expériences, Prefect pour l'orchestration des workflows, FastAPI pour fournir une API de prédiction, et Streamlit pour une interface utilisateur interactive.

## Vue d'Ensemble

Le projet est structuré autour de plusieurs services principaux, chacun étant conteneurisé à l'aide de Docker pour faciliter le déploiement et la gestion des dépendances :

- **MLflow :** Pour le suivi des expériences, l'enregistrement des modèles et de leurs métriques.
- **Prefect :** Pour orchestrer le workflow de préparation des données, d'entraînement et d'évaluation du modèle.
- **FastAPI :** Pour exposer une API REST permettant de réaliser des prédictions avec le modèle entraîné.
- **Streamlit :** Pour offrir une interface utilisateur permettant de visualiser et interagir avec les résultats des prédictions.

## Structure du Projet

- `/flows` : Scripts Prefect pour l'orchestration des workflows d'entraînement et d'évaluation du modèle.
- `/api` : Application FastAPI pour l'API de prédiction.
- `/client` : Application Streamlit pour l'interface utilisateur.
- `/data` : Contient le dataset `winequality.csv` pour l'entraînement et l'évaluation du modèle.
- `/model` : Stocke le modèle entraîné et d'autres artefacts liés au modèle.
- `Dockerfile` : Fichiers Dockerfile pour chaque service, définissant les étapes de construction des images Docker.
- `docker-compose.yml` : Configuration Docker Compose pour orchestrer le déploiement de tous les services.

## Prérequis

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Sous Windows, [WSL 2](https://docs.microsoft.com/fr-fr/windows/wsl/install) et [Docker Desktop](https://www.docker.com/products/docker-desktop) avec le backend WSL 2 activé.

## Installation et Démarrage

1. **Cloner le dépôt :**

   Clonez le dépôt sur votre machine locale et naviguez dans le répertoire du projet :

    ```bash
    git clone <URL_DU_REPO>
    cd <NOM_DU_REPO>
    ```

2. **Construire les images Docker :**

   Construisez les images Docker pour tous les services à l'aide de Docker Compose :

    ```bash
    docker-compose build
    ```

3. **Démarrer les services :**

   Lancez les services en arrière-plan :

    ```bash
    docker-compose up -d
    ```

## Accès aux Services

- **MLflow UI :** `http://localhost:5000` - Interface utilisateur de MLflow pour suivre les expériences.
- **API FastAPI :** `http://localhost:8000` - API de prédiction. Utilisez `http://localhost:8000/docs` pour accéder à la documentation Swagger de l'API.
- **Application Streamlit :** `http://localhost:8501` - Interface utilisateur Streamlit pour visualiser les prédictions.

## Nettoyage

Pour arrêter et supprimer les conteneurs, ainsi que les réseaux et volumes créés par Docker Compose :

```bash
docker-compose down -v


# Env
    conda env create -f environment.yml
    conda activate mlde
    # conda env remove --name mlde
    # Remove-Item -Recurse C:\Users\Kynes\.conda\envs\mlde

# Prefect
## Infos
    Prefect 2.x utilise un serveur "éphémère" et une base de données SQLite par défaut,
    ce qui simplifie la configuration et l'utilisation pour les développements locaux.
    Il n'est pas obligatoire d'utiliser la version cloud.
    attention la commande 'orion' n'est plus utilisée (utiliser 'server').
    Erreur "utf-8" dans mon environnement / windows 
        => passage à la version cloud.
## Server (NON)
    ### Lancer
        prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api
        prefect server start --host 0.0.0.0
    ### Accéder
        http://127.0.0.1:4200/docs
        http://127.0.0.1:4200/

## Cloud (OUI)
    créer un compte sur : https://app.prefect.cloud
    prefect cloud login

## Puis lancer le fichier
    python .\5MLDE_proj_wine_light.py
    avec .serve => ajout du deployment : le script doit rester actif pour que le schedul fonctionne.
    avec .deploy => il faut relancer via CLI ou UI
    besoin d'un workpool :
    prefect work-pool create my-managed-pool --type prefect:managed 
    vérifier les workpool :
    prefect work-pool ls
    et ajout du ciblage vers le github
    lancement par :
    prefect deployment run 'Wine quality prediction flow/WineDeployGithubDeployment'
    => Déploiment mis de côté pour le moment, les fonctions sont en commentaire dans le "if __name__ == "__main__":"

# Great_expectations
## installer
    conda install conda-forge::great-expectations
## intialiser
    great_expectations init
    great_expectations datasource new
    => ouverture du notebook, choisir un nom et lancer les cellules : 'winequality_datasource'
    great_expectations suite new
    => automatique / choix du nom : 'wine_quality_expectation_suite'
    great_expectations checkpoint new wine_quality_checkpoint
    => ouverture du notebook : tout executer
## lancer 
    context.run_checkpoint(checkpoint_name="wine_quality_checkpoint")
    context.open_data_docs()