import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.cluster import KMeans

# Charger le fichier NIfTI
file_path = "ProjectFiles/p01.nii"  # Modifier ce chemin si besoin
nifti_img = nib.load(file_path)

# Accéder à la matrice 4D
img_4D = nifti_img.get_fdata()  # Shape (91, 109, 91, 29)

# Filtrer les voxels ayant une intensité significative (évite le bruit)
threshold = np.percentile(img_4D, 90)  # Garde les 10% de valeurs les plus élevées
mask = img_4D.max(axis=-1) > threshold

# Extraire les courbes temporelles des voxels sélectionnés
x, y, z = np.where(mask)
voxel_time_series = img_4D[x, y, z, :]

# Appliquer K-Means sur les courbes temporelles
n_clusters = 3  # Ajustable
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(voxel_time_series)
labels = kmeans.labels_

# Visualisation des courbes les plus typées carotide
plt.figure(figsize=(10, 6))

carotid_like_voxels = []
best_coords = []

for i in range(n_clusters):
    cluster_mask = labels == i
    cluster_voxels = voxel_time_series[cluster_mask]

    # Calcul du temps du pic et de l'aire sous la courbe
    time_to_peak = np.argmax(cluster_voxels, axis=1)
    auc = cluster_voxels.sum(axis=1)

    # Score pondéré : favorise pic précoce et AUC élevée
    score = auc - 10 * time_to_peak  # Pondération ajustable

    # Sélection du voxel le plus représentatif
    best_idx_in_cluster = np.argmax(score)
    global_idx = np.where(cluster_mask)[0][best_idx_in_cluster]
    best_curve = cluster_voxels[best_idx_in_cluster]
    coord = (x[global_idx], y[global_idx], z[global_idx])

    # Sauvegarde
    carotid_like_voxels.append(best_curve)
    best_coords.append(coord)

    # Tracé de la courbe
    plt.plot(best_curve, label=f'Cluster {i} @ {coord}')

# Affichage du graphique
plt.xlabel("Temps (t)")
plt.ylabel("Intensité du voxel")
plt.title("Courbes des voxels les plus typés carotide par cluster")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Affichage des coordonnées exactes
for i, (xi, yi, zi) in enumerate(best_coords):
    print(f"Cluster {i} - Voxel représentatif : Voxel indices (x, y, z) = ({xi}, {yi}, {zi})")
