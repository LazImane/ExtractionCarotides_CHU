import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Détecter le chemin actuel du script
data_path = os.path.dirname(os.path.abspath(__file__))
filename_img_4D = os.path.join(data_path, "../patients_cam/p09.nii")

# Charger le fichier NIfTI 4D
if not os.path.exists(filename_img_4D):
    print(f"Erreur : Le fichier {filename_img_4D} est introuvable.")
    exit()

nifti_img = nib.load(filename_img_4D)
img_4D = nifti_img.get_fdata()  # Charger les données en tableau NumPy
print(f"Image chargée avec succès. Dimensions : {img_4D.shape}")

# Demander les coordonnées et la taille du groupe
try:
    voxel_coords = input("Entrez les coordonnées du voxel central (i, j, k) séparées par des espaces : ")
    i, j, k = map(int, voxel_coords.split())
    group_size = int(input("Entrez la taille de ROI (rayon en voxels) : "))

    # Vérification des limites
    if not (0 <= i < img_4D.shape[0] and 0 <= j < img_4D.shape[1] and 0 <= k < img_4D.shape[2]):
        raise ValueError("Les coordonnées du voxel sont hors limites.")
    if group_size < 0:
        raise ValueError("La taille du ROI doit être un entier positif.")
except ValueError as e:
    print(f"Erreur : {e}")
    exit()

# Calculer les limites du groupe
i_min = max(0, i - group_size)
j_min = max(0, j - group_size)
k_min = max(0, k - group_size)

i_max = min(img_4D.shape[0], i + group_size + 1)
j_max = min(img_4D.shape[1], j + group_size + 1)
k_max = min(img_4D.shape[2], k + group_size + 1)

# Extraire les voxels du groupe
group_tac = img_4D[i_min:i_max, j_min:j_max, k_min:k_max, :]

# Calculer la TAC moyenne sur le groupe
tac_mean = np.mean(group_tac, axis=(0, 1, 2))

# Afficher des informations utiles
print("Limites de la ROI :")
print(f"i : [{i_min}, {i_max}], j : [{j_min}, {j_max}], k : [{k_min}, {k_max}]")
print(f"TAC moyenne : {tac_mean}")

# Visualiser les courbes
time_points = np.arange(len(tac_mean))

plt.plot(time_points, img_4D[i, j, k, :], marker='o', linestyle='-', color='blue', label=f'Voxel central ({i}, {j}, {k})')
plt.plot(time_points, tac_mean, marker='s', linestyle='--', color='green', label=f'TAC moyenne (ROI={group_size})')

plt.title('TAC pour un voxel et sa moyenne régionale')
plt.xlabel('Temps (Index des frames)')
plt.ylabel('Intensité')
plt.legend()
plt.grid(True)
plt.show()
