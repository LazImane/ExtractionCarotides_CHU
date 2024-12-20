import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter


data_path = r"ProjectFiles/"
filename_img_4D =  os.path.join(data_path, "w_PET_4D.nii")

########## CHARGER UN FICHIER .NII
nifti_img = nib.load(filename_img_4D)

########## ACCEDER A LA MATRICE 4D

try:
    first_choice = int(input("Tapez 1 pour extraire la TAC à partir du voxel le plus fort ou 2 pour voir la moyenne des voxels de la carotide : "))
    if first_choice == 1:
        print("Vous avez choisi d'extraire la TAC à partir du voxel le plus fort")
    elif first_choice == 2:
        print("Vous avez choisi de voir la moyenne des voxels de la carotide")
        group_size = int(input("Entrez la taille du groupe (rayon en voxels) : "))
    else:
        print("Veuillez choisir 1 ou 2")
        exit()
except ValueError:
    print("Veuillez entrer une valeure valide")
    exit()

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


def collect_voxel_values_4d(big_increase, roi_data, t):
    """
    Crée une matrice 4D avec les valeurs des voxels correspondant aux positions True dans big_increase.

    Parameters:
        big_increase (numpy.ndarray): Tableau 3D contenant des booléens (True pour une augmentation significative).
        roi_data (numpy.ndarray): Région d'intérêt 4D (x, y, z, t).
        t (int): Index temporel pour extraire les valeurs des voxels.

    Returns:
        numpy.ndarray: Matrice 4D contenant les valeurs détectées, zéro sinon.
    """
    # Créer une matrice 4D de même forme que roi_data
    detected_voxels = np.zeros_like(roi_data)
    
    # Ajouter les valeurs détectées à la matrice
    detected_voxels[:, :, :, t] = np.where(big_increase, roi_data[:, :, :, t], 0)
    
    return detected_voxels
    
########## Fonction de Test pour identifier la carotide

# def identify_carotid(pos1,pos2,t_min=4,t_max=8):
#     x_min, x_max = min(pos1[0], pos2[0]), max(pos1[0], pos2[0])
#     y_min, y_max = min(pos1[1], pos2[1]), max(pos1[1], pos2[1])
#     z_min, z_max = min(pos1[2], pos2[2]), max(pos1[2], pos2[2])
#     roi_data = data[x_min:x_max, y_min:y_max, z_min:z_max, :]
#     baseline = roi_data[:,:,:,t_min]

#     carotid_mask = np.zeros(roi_data.shape[:3])

#     for t in range(t_min, t_max):
#         current_frame = roi_data[:,:,:,t]
#         big_increase = current_frame > (5*baseline)
#         tabvoxels = collect_voxel_values_4d(big_increase, roi_data, t)
#     return tabvoxels
 

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

# Permet de calculer la moyenne des voxels autour d'un point donné si besoin

def get_average_voxels(str_point, group_size, matrix, t):
    i, j, k = str_point[:3]
    i_min, i_max = max(0, i - group_size), min(matrix.shape[0], i + group_size + 1)
    j_min, j_max = max(0, j - group_size), min(matrix.shape[1], j + group_size + 1)
    k_min, k_max = max(0, k - group_size), min(matrix.shape[2], k + group_size + 1)
    group_tac = matrix[i_min:i_max, j_min:j_max, k_min:k_max, t]
    return np.mean(group_tac)

# Permet de récupérer la TAC à partir du voxel le plus fort ou de la moyenne des voxels de la carotide

def get_Tac_point():
    matrix = extract_submatrix_4d(img_4D, pos1, pos2)
    coords = get_stronger_point(matrix)
    TAC_input = []
    if first_choice == 1:
        for t in range(29):
            TAC_input.append(matrix[coords[0]-1, coords[1]-1, coords[2]-1, t]) # On affiche les valeurs de la TAC pour chaque temps
        return TAC_input
    
    elif first_choice == 2:
        for t in range(29):
            TAC_input.append(get_average_voxels(coords, group_size, matrix, t)) # On affiche les valeurs de la TAC pour chaque temps
        return TAC_input
    
    

def gammavariate_model_dec(t, A1, A2, A3, lambda1, lambda2, lambda3):
    """
    Compute a gamma variate model for the given parameters over time.

    This function calculates the concentration of a tracer using a gamma variate model
    based on the provided amplitudes and decay rates. It is particularly used to model 
    the exponential decay observed in tracer kinetics.

    Parameters:
    t : array-like
        Time points at which the model is evaluated.
    A1, A2, A3 : float
        Amplitudes of the exponential components.
    lambda1, lambda2, lambda3 : float
        Decay rates of the exponential components.

    Returns:
    Cpmodel : array-like
        The computed concentration values at each time point.
    """
    Cpmodel = (A1 - A2 - A3) * np.exp(-lambda1 * t) + A2 * np.exp(-lambda2 * t) + A3 * np.exp(-lambda3 * t)
    return Cpmodel

t = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,150,180,210,240,270,300,360,420,480,540,600, 900, 1200, 1500, 1800, 2100, 2400])
t_min = t/60
delta_i = np.diff(t_min) # Durée de chaque frame en min

TAC_input = get_Tac_point()
TAC_input.insert(0, 0) # On ajoute une valeur nulle au début de la TAC  
t_data = t_min # Tableau contenant toutes les valeurs de temps des acquisitions en minutes
y_data = TAC_input # Contient les valeurs de la TAC

# Trouver l'index de la valeur maximale de yy
idx_max = np.argmax(y_data) # Index de la valeur maximale de la TAC

# Extraire la partie décroissante des données
xx_dec = t_data[idx_max:] 
yy_dec = y_data[idx_max:]

# Extraire les valeurs de t qui ont été supprimées
xx_removed = t_data[:idx_max]
yy_removed = [0] * len(xx_removed)

# Ajouter ces valeurs au début de xx_dec et yy_dec
xx_full = np.concatenate((xx_removed, xx_dec))
yy_full = np.concatenate((yy_removed, yy_dec))

params_dec, covariance_dec = curve_fit(gammavariate_model_dec, xx_dec, yy_dec, maxfev=100000000)

IDIF_fit = np.concatenate((yy_removed, gammavariate_model_dec(xx_dec, *params_dec)))

########## PLOT DES RESULTATS
plt.figure(figsize=(10, 6))

# Tracé des données expérimentales
plt.plot(t_min, TAC_input, 'bo-', label="Données expérimentales", markersize=4)

# Tracé du modèle ajusté
plt.plot(t_min, IDIF_fit, 'r--', label="Modèle ajusté")

plt.xlabel("Temps (minutes)")
plt.ylabel("Concentration (kBq/cc)")
plt.title("Ajustement de la TAC par le modèle gamma-varié")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# img_4D = nifti_img.get_fdata() # Shape Matrice 4D  : (91, 109, 91, 29) : x = 91, y = 109, z = 91, t = 29

# nifti_img = nib.Nifti1Image(img_4dtest, nifti_img.affine)
# filename_img_4D_2 = D =  os.path.join(data_path, "w_PET_4D_newtest.nii")
# nib.save(nifti_img, filename_img_4D_2)
