import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Chemin vers le fichier NIfTI
data_path = "C:\\Users\\camil\\OneDrive - umontpellier.fr\\PROJET_INFO_S6"  # À CHANGER 
filename_img_4D = os.path.join(data_path, "w_PET_4D.nii")

# Charger le fichier NIfTI 4D
nifti_img = nib.load(filename_img_4D)
# Obtenir les données sous forme de tableau NumPy
img_4D = nifti_img.get_fdata()  # Dimensions typiques : (X, Y, Z, T)


# Demander les coordonnées du voxel central et la taille du groupe
try:
    voxel_coords = input("Entrez les coordonnées du voxel central (i, j, k) séparées par des espaces : ")
    i, j, k = map(int, voxel_coords.split())
    group_size = int(input("Entrez la taille du groupe (rayon en voxels) : "))
except ValueError:
    print("Veuillez entrer des valeurs valides.")
    exit()

# Calculer les limites du groupe
i_min, i_max = max(0, i - group_size), min(img_4D.shape[0], i + group_size + 1)
j_min, j_max = max(0, j - group_size), min(img_4D.shape[1], j + group_size + 1)
k_min, k_max = max(0, k - group_size), min(img_4D.shape[2], k + group_size + 1)



# Extraire les voxels du groupe
group_tac = img_4D[i_min:i_max, j_min:j_max, k_min:k_max, :]

# Calculer la TAC moyenne sur le groupe
tac_mean = np.mean(group_tac, axis=(0, 1, 2))


# Affichage infos
print("i_min:", i_min, "i_max",i_max, "j_min", j_min, "j_max", j_max, "k_min", k_min, "k_max", k_max)
#print("group_tac\n", group_tac)
print("tac_mean", tac_mean)

# Visualiser la TAC moyenne
time_points = range(len(tac_mean))  # Index temporels (ou utilisez une échelle réelle si disponible)
plt.plot(time_points, tac_mean, marker='o')
plt.title(f'TAC moyenne pour le groupe centré sur ({i}, {j}, {k}) avec taille {group_size}')
plt.xlabel('Temps')
plt.ylabel('Intensité moyenne')
plt.grid(True)
plt.show()
