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
├── Bibliographie/
│   ├── Articles_et_résumé_des_articles/
│   ├── Énoncé_du_projet/
│   ├── Rapport_final/
│   ├── Rapports_avancement_au_cours_du_projet/
│
├── Comparaison_entre_nos_TACs_et_celle_de_l_encadrant/
│
├── Data/
│   ├── nii/         # Images TEP 4D au format NIfTI (.nii)
│   └── crv/         # Fichiers de TAC de référence (.crv)
│
├── Graphs/          # Graphiques générés pour les masques et rapports
│
├── Pistes_et_Tests/
│   ├── autres_pistes/
│   ├── piste_kmeans/
│   ├── piste_meanshiftetbrainmask/
│   ├── piste_ward/
│
├── Resultats_finaux_de_notre_fonction/       # Figures .png produites par la fonction principale
├── TacReferences/                            # TAC manuelles utilisées comme référence
│
├── download_data.py              # Script de téléchargement initial des données
├── idif_pipeline_pet.py          # Script principal du pipeline final
├── tests_unitaires.py            # Début d’intégration de tests unitaires
├── README.md                     # Ce fichier de documentation


---

###  Lancement du projet

### 1. Télécharger les données

Exécute d’abord :

```bash
python download_data.py
```

Ce script télécharge les fichiers `.nii` et `.crv` depuis un Google Drive et crée automatiquement le dossier `Data/` avec les sous-dossiers nécessaires.

### 2. Lancer le pipeline principal

Ensuite, exécute :

```bash
python idif_pipeline_pet.py
```

Tu seras invité à entrer un identifiant patient : `p01` à `p10`  
⚠️ `p08` n’est pas disponible (pas de données).

Les fichiers résultats (visualisations + TACs) sont enregistrés dans :

```bash
Resultats_finaux_de_notre_fonction/
```

### 3. Explorer les autres pistes

Le dossier `Pistes_et_Tests/` contient toutes les expérimentations testées (KMeans, MeanShift, etc.) avant d’aboutir à la méthode finale retenue.

---

##  Auteurs

Projet réalisé dans le cadre du **L3 Informatique à la Faculté des Sciences de Montpellier**  par Alia CHEBBI, Camille GUIGUEN, Colin DIEVART, Imene LAAZIZI et Kanzy YOUSSEF
Encadré par **Dr Florentin Kucharczak** *(CHU Montpellier – LIRMM)*
