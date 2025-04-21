import os
import numpy as np
import nibabel as nib
from nilearn.image import mean_img
import matplotlib.pyplot as plt

# === Chargement du fichier ===
data_path = os.path.dirname(os.path.abspath(__file__))
filename_img_4D = os.path.join(data_path, "./patients_cam/p09.nii")

if not os.path.exists(filename_img_4D):
    print(f"Erreur : Le fichier {filename_img_4D} est introuvable.")
    exit()

# Charger l'image 4D
nifti_img = nib.load(filename_img_4D)
img_4D = nifti_img.get_fdata()
print(f"Image 4D chargée. Dimensions : {img_4D.shape}")

# === Calcul de l'image moyenne dans le temps ===
img_mean = mean_img(filename_img_4D)  # image 3D
mean_data = img_mean.get_fdata()

# === Création d’un masque cérébral ===
# On garde les voxels ayant une intensité > à un seuil relatif (ex: 80e percentile)
threshold = np.percentile(mean_data, 80)
brain_mask = mean_data > threshold
print(f"Seuil utilisé pour le masque : {threshold:.2f}")
print(f"Nombre de voxels dans le masque cerveau : {np.sum(brain_mask)}")

# === Application du masque au 4D ===
masked_img_4D = img_4D * brain_mask[..., np.newaxis]

# === Trouver le voxel max après masque ===
max_voxel_idx = np.unravel_index(np.argmax(masked_img_4D), masked_img_4D.shape)
print(f"Coordonnées du voxel max dans le cerveau : {max_voxel_idx}")
print(f"Valeur max après masque : {masked_img_4D[max_voxel_idx]}")

# === Visualisation rapide du masque sur une coupe centrale ===
center_slice = mean_data.shape[2] // 2

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(mean_data[:, :, center_slice].T, origin="lower", cmap="gray")
plt.title("Image moyenne TEP (coupe axiale)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(brain_mask[:, :, center_slice].T, origin="lower", cmap="hot")
plt.title("Masque cerveau estimé")
plt.axis('off')

plt.tight_layout()
plt.show()
