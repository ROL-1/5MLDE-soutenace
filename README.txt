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
## Server
    ### Lancer
        prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api
        prefect server start --host 0.0.0.0
    ### Accéder
        http://127.0.0.1:4200/docs
        http://127.0.0.1:4200/

## cloud
    créer un compte sur : https://app.prefect.cloud
    prefect cloud login

## Puis lancer le fichier
    python .\5MLDE_proj_wine_light.py