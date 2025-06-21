
-----        Cadrage 'Compagnon Immmobilier'    -----

Cas d'usages auquels nous souhaitons répondre en l'état de nos connaissances :

Cas 1: (achat résidence principale)
"En fonction du type de bien, de la localisation, de son état, et d'autres variables à définir (le minimum possible), recommmander un prix de vente / d'achat du bien, ainsi que l'évolution de sa valeur sur les 5 années à venir"

Cas 2: (investissement locatif)
"Pour les investisseurs locatifs, être en mesure de rapprocher le prix d'achat de la rentabilité qu'il pourra procurer, pour ensuite donner une recommandation "GO" ou "NO GO". Tenter de mettre en place un indicateur de "facilité de gestion" (définition à affiner)



-----        MÉTHODOLOGIE        -----

- Exploration des datasets initiaux  : Synthèse dans le fichier excel sur le repo
    - ech_annonces_locations_68.csv (locations département 68)
    - ech_annonces_ventes_68.csv (ventes département 68)
    - DVF (année par année)


- Première formalisation des cas d'usages auxquels nous souhaitons répondre (en tête de ce document)


- Création / concaténation de fichiers
    - ventes DVF de 2020 à 2025
    - création d'un fichier national de ventes à partir des fichiers "ech_annonces_ventes_XX.csv => merged_sales_data.csv
    - création d'un fichier national de locations à partir des fichiers "ech_annonces_location_XX.csv


- Seconde exploration après merge des fichiers:
    - notebooks associés: 
            "Second_exploration_and_merge_DVF" 
            "Second_exploration_and_merge_VENTE_et_LOC


- Première régression avant enrichissement:
    - notebook associé:
        "Première régression avant enrichissement"
    - premieres problématiques de modélisation soulevées :
            - Compte tenu de la nature du projet, des performances observées, et du fait que le modèle HistGradientBoostingRegressor gère nativement les valeurs manquantes, contrairement au SGDRegressor qui impose une imputation souvent arbitraire, pensez-vous qu’il soit pertinent de privilégier un modèle plus complexe mais robuste, comme le HGBR, même si son interprétation directe est moins intuitive qu’un modèle linéaire ?
            => Ou recommanderiez-vous de garder un modèle linéaire moins performant, mais plus transparent, quitte à devoir “fabriquer” une partie des données via l’imputation ?


- Enrichissement
    exploration d'API :
        - API de l'Ademe : https://data.ademe.fr/datasets/dpe03existant
        - Doc : https://data.ademe.fr/datasets/dpe03existant/api-doc

        Description des étapes:
        - exploration de l'API, de sa documentation
        - choix d'enrichissement sur les Nan des colonnes de nos datasets "ech_annonces_ventes_XX.csv"
        - création d'un script d'enrichissement par chunk (notebook sur le repo), en rapprochant les données géo des fichiers "ech_annonces_ventes_XX.csv" et des données API.
        !! nous rencontrons des problématiques de temps de calcul et de disponibilité de l'API !! (limité à 600 calls par minute)

