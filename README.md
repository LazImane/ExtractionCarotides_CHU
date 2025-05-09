# 🧪 Extraction semi-automatique de courbes TAC sur images TEP dynamiques

## 🎯 Objectif

Ce projet vise à extraire de manière semi-automatique les **courbes temps-activité (TAC)** des **artères carotides** à partir d’images TEP dynamiques (4D), pour estimer la fonction d’entrée (IDIF) sans recours à des prélèvements sanguins.

---

## 🧠 Contexte

Dans les études TEP dynamiques, la modélisation pharmacocinétique nécessite une fonction d’entrée artérielle. Cette extraction, généralement invasive, est ici remplacée par une méthode image-dérivée utilisant :

- Segmentation du cerveau et du bas cerveau
- Seuillage automatique
- Parcellisation (Ward clustering)
- Identification des clusters gauche/droit
- Calcul de la moyenne d’intensité temporelle
- Ajustement par modèle gamma-variate (Feng 1994)

---

## 🗂 Structure des fichiers

```bash
📁 Projet/
├──
├── Data/
│   ├── nii/             # Images TEP 4D au format NIfTI (.nii)
│   └── crv/             # Fichiers de TAC de référence (.crv)
├── Resultats_finaux_de_notre_fonction/
│   └── Resultats_de_la_fonction_p01.png
├── Fichier_fonctions_projet_L3.py  # Script principal
├── README.md            # Documentation du projet
