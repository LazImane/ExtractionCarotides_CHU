import os
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import nibabel as nib 
from scipy.optimize import curve_fit 

data_path = "C:\\Users\\camil\\OneDrive - umontpellier.fr\\PROJET_INFO_S6" ## A CHANGER
filename_img_4D =  os.path.join(data_path, "w_PET_4D.nii")

# Charger le fichier NIfTI 4D
nifti_img = nib.load(filename_img_4D)

# Obtenir les données sous forme de tableau NumPy
img_4D = nifti_img.get_fdata()  # Dimensions typiques : (X, Y, Z, T)


########## SÉLECTION MANUELLE D'UN VOXEL EN PARTICULIER
voxel_coords = input("Entrez les coordonnées du voxel (i, j, k) séparées par des espaces : ")
i, j, k = map(int, voxel_coords.split())
tac_voxel = img_4D[i, j, k, :]
print(f"TAC pour le voxel ({i}, {j}, {k}): {tac_voxel}")

# Extraire la courbe temporelle pour ce voxel
tac = img_4D[i, j, k, :]  # Données sur la dimension temporelle

# Visualiser la TAC
time_points = range(len(tac))  # Index temporels (ou utilisez une échelle réelle si disponible)
plt.plot(time_points, tac, marker='o')
plt.title(f'TAC pour le voxel ({i}, {j}, {k})')
plt.xlabel('Temps')
plt.ylabel('Intensité')
plt.grid(True)
plt.show()
