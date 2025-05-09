#  Extraction semi-automatique de courbes TAC sur images TEP dynamiques

## Objectif

Ce projet vise à extraire de manière semi-automatique les **courbes temps-activité (TAC)** des **artères carotides** à partir d’images TEP dynamiques (4D), pour estimer la fonction d’entrée (IDIF) sans recours à des prélèvements sanguins.

---

## Contexte

Dans les études TEP dynamiques, la modélisation pharmacocinétique nécessite une fonction d’entrée artérielle. Cette extraction, généralement invasive, est ici remplacée par une méthode image-dérivée utilisant :

- Segmentation du cerveau et du bas cerveau
- Seuillage automatique
- Parcellisation (Ward clustering)
- Identification des clusters gauche/droit
- Calcul de la moyenne d’intensité temporelle
- Ajustement par modèle gamma-variate (Feng 1994)

---

## Structure des fichiers

```bash
📁 Projet/
├──Bibliographie
│   ├── Articles_et_résumé_des_articles
│   ├── Énoncé_du_projet
│   ├── Rapport_final
│   ├── Rapports_avancement_au_cours_du_projet
├── Comparaison_entre_nos_TACs_et_celle_de_l_encadrant
├── Data/
│   ├── nii/             # Images TEP 4D au format NIfTI (.nii)
│   └── crv/             # Fichiers de TAC de référence (.crv)
├── Graphs
├── Pistes_et_Tests
├── Resultats_finaux_de_notre_fonction/
│   └── Resultats_de_la_fonction_p01.png
TacReferences
├── download_data.py  # Pour télécharger les datas
├── idif_pipeline_pet.py
├── tests_unitaires.py   # Début d'intégration de Tests Unitaires 
├── README.md            # Documentation du projet
