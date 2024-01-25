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