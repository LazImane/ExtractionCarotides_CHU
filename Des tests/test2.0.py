import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter 

# Chemin dynamique pour charger le fichier NIfTI
data_path = os.path.dirname(os.path.abspath(__file__))
filename_img_4D = os.path.join(data_path, "w_PET_4D.nii")

########## CHARGER UN FICHIER .NII
nifti_img = nib.load(filename_img_4D)

########## ACCEDER A LA MATRICE 4D

img_4D = nifti_img.get_fdata() # Shape Matrice 4D  : (91, 109, 91, 29) : x = 91, y = 109, z = 91, t = 29

data = gaussian_filter(img_4D, sigma=1) # Filtre Gaussien pour réduire le bruit 

########## Commencer par isoler la ROI ( Region d'interet)
########## Etant donné que la carotide se trouvera dans une zone précise du cerveau ( peu importe le patient ), on peut isoler cette zone pour réduire le temps de calcul

def create_roi(matrix):
    shape = matrix.shape
    pos1 = [int(np.floor(0.27 * shape[0])), int(np.floor(0.20 * shape[1])), int(np.floor(0.0 * shape[2]))]
    pos2 = [int(np.ceil(0.35 * shape[0])), int(np.ceil(0.90 * shape[1])), int(np.ceil(0.20 * shape[2]))]
    return pos1, pos2

pos1, pos2 = create_roi(data) # On récupère les coordonnées de la ROI approximative

def get_stronger_point(matrix):
    """
    Retourne le point le plus fort de la matrice 4D

    Parameters:
        matrix (numpy.ndarray): Matrice 4D.

    Returns:
        tuple: Coordonnées du point le plus fort (1-indexed).
    """
    point = np.unravel_index(np.argmax(matrix), matrix.shape)
    point_1_based = tuple(p + 1 for p in point)

    return point_1_based

def extract_submatrix_4d(data, pos1, pos2):
    """
    Extrait une sous-matrice 4D à partir des coordonnées données pos1 et pos2.

    Parameters:
        data (numpy.ndarray): Matrice 4D originale (x, y, z, t).
        pos1 (tuple): Coordonnées (x_min, y_min, z_min) du coin supérieur gauche de la région.
        pos2 (tuple): Coordonnées (x_max, y_max, z_max) du coin inférieur droit de la région.

    Returns:
        numpy.ndarray: Sous-matrice 4D correspondant à la région définie par pos1 et pos2.
    """
    x_min, x_max = min(pos1[0], pos2[0]), max(pos1[0], pos2[0])
    y_min, y_max = min(pos1[1], pos2[1]), max(pos1[1], pos2[1])
    z_min, z_max = min(pos1[2], pos2[2]), max(pos1[2], pos2[2])

    # Extraire la sous-matrice 4D
    submatrix = data[x_min:x_max, y_min:y_max, z_min:z_max, :]

    return submatrix

def get_Tac_point():
    matrix = extract_submatrix_4d(img_4D, pos1, pos2)
    coords = get_stronger_point(matrix)
    TAC_input = []
    print(coords)
    for i in range(29):
        TAC_input.append(matrix[coords[0]-1, coords[1]-1, coords[2]-1, i]) # On affiche les valeurs de la TAC pour chaque temps
    return TAC_input

def gammavariate_model_dec(t, A1, A2, A3, lambda1, lambda2, lambda3):
    """
    Modèle gamma-varié corrigé pour éviter les débordements.
    """
    t = np.clip(t, a_min=0, a_max=None)  # Limiter t à des valeurs positives
    exp1 = np.exp(-np.clip(lambda1 * t, a_min=None, a_max=700))  # Limiter l'exponentielle
    exp2 = np.exp(-np.clip(lambda2 * t, a_min=None, a_max=700))
    exp3 = np.exp(-np.clip(lambda3 * t, a_min=None, a_max=700))
    
    Cpmodel = (A1 - A2 - A3) * exp1 + A2 * exp2 + A3 * exp3
    return Cpmodel

# Définition des points temporels
t = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 180, 210, 240, 270, 300, 360, 420, 480, 540, 600, 900, 1200, 1500, 1800, 2100])
t_min = t / 60  # Convertir en minutes

# Extraction des données TAC
TAC_input = get_Tac_point()  # Obtenir les données TAC pour le point le plus fort
TAC_input.insert(0, 0)  # Ajouter une valeur nulle au début de la TAC

# Ajuster t_min pour inclure un point supplémentaire
t_min = np.insert(t_min, 0, 0)  # Ajouter une valeur temporelle nulle au début

t_data = t_min  # Tableau contenant toutes les valeurs de temps des acquisitions en minutes
y_data = TAC_input  # Contient les valeurs de la TAC

# Vérifier si les dimensions sont correctes
if len(t_data) != len(y_data):
    print(f"Erreur : Les dimensions des données ne correspondent pas. t_data: {len(t_data)}, y_data: {len(y_data)}")
    exit()

# Trouver l'index de la valeur maximale
idx_max = np.argmax(y_data)

# Extraire les parties décroissantes
xx_dec = t_data[idx_max:]  # Temps après le maximum
yy_dec = y_data[idx_max:]  # TAC après le maximum

# Normalisation des données pour l'ajustement
yy_dec_norm = yy_dec / np.max(yy_dec)

# Définir des valeurs initiales raisonnables
p0 = [1.0, 0.5, 0.1, 0.01, 0.005, 0.001]  # Initialisation des paramètres

# Ajustement avec curve_fit
try:
    params_dec, covariance_dec = curve_fit(
        gammavariate_model_dec, xx_dec, yy_dec_norm, p0=p0, maxfev=100000
    )
    # Dénormaliser les paramètres si nécessaire
    params_dec[:3] *= np.max(yy_dec)
except RuntimeError as e:
    print(f"Erreur d'ajustement : {e}")
    exit()

# Générer les valeurs ajustées
IDIF_fit = gammavariate_model_dec(t_min, *params_dec)

# Visualisation
plt.figure(figsize=(10, 6))
plt.plot(t_min, TAC_input, 'bo-', label="Données expérimentales", markersize=4)
plt.plot(t_min, IDIF_fit, 'r--', label="Modèle ajusté")
plt.title("Ajustement de la TAC par le modèle gamma-varié (corrigé)")
plt.xlabel("Temps (minutes)")
plt.ylabel("Concentration (kBq/cc)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()