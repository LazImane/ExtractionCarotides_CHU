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

img_4D = nifti_img.get_fdata() # Shape Matrice 4D  : (91, 109, 91, 29) : x = 91, y = 109, z = 91, t = 29

########## ECHANTILLONAGE DES SERIES TEMPORELLES 

t = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,150,180,210,240,270,300,360,420,480,540,600, 900, 1200, 1500, 1800, 2100, 2400])
t_min = t/60
delta_i = np.diff(t_min) # Durée de chaque frame en min
########## CHARGER LE FICHIER .TXT de la TAC MANUELLE
tac_input_filename = os.path.join(data_path,f"P01_carot.crv")
        
if os.path.isfile(tac_input_filename):
    df = pd.read_csv(tac_input_filename, sep="\t", skiprows=0)
    
    # Récupère les données numériques de la deuxième colonne
    TAC_input = df["value[kBq/cc]"].to_numpy()
    TAC_input = np.insert(TAC_input, 0, 0) 
    print(TAC_input)
else : 
    raise FileNotFoundError("Erreur de chargement du fichier!")


########## FAIRE LE FIT A LA PARTIE DECROISSANTE AVEC LE MODELE DE FENG 
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

# with open("matrice4dsans0.txt", "w") as f:
#     for x in range(img_4D.shape[0]):
#         for y in range(img_4D.shape[1]):
#             for z in range(img_4D.shape[2]):
#                 for t in range(img_4D.shape[3]):
#                     value = img_4D[x, y, z, t]
#                     if value > 5:
#                         f.write(f"x={x+1}, y={y+1}, z={z+1}, t={t+1}, value={value:.2f}\n")

# f = open("matrice4d.txt", "w")
# f.write(str(img_4D))
# f.close()

########## APRES MODIFICATIONS, SAUVEGARDER A LA NOUVELLE MATRICE 4D APRES MODIFICATIONS AU FORMAT NII

# img_4D_2 = img_4D**2 # Exemple d'opération sur la matrice 4D
img_4D_2 = gaussian_filter(img_4D, sigma=1)

nifti_img = nib.Nifti1Image(img_4D_2, nifti_img.affine)
filename_img_4D_2 = D =  os.path.join(data_path, "w_PET_4D_new.nii")
nib.save(nifti_img, filename_img_4D_2)

