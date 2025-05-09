import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# Charger le fichier NIfTI
file_path = "../data/nii/p11. ii"
nifti_img = nib.load(file_path)

# Accéder à la matrice 4D
img_4D = nifti_img.get_fdata()

# Filtrer les voxels ayant une intensité significative
threshold = np.percentile(img_4D, 90)
mask = img_4D.max(axis=-1) > threshold

# Extraire les courbes temporelles des voxels sélectionnés
x, y, z = np.where(mask)
voxel_time_series = img_4D[x, y, z, :]

# Appliquer K-Means sur les courbes temporelles
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(voxel_time_series)
labels = kmeans.labels_

# Définir les temps d'acquisition (t en secondes, t_min en minutes)
t = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,150,180,210,240,270,300,360,420,480,540,600,900,1200,1500,1800,2100,2400])
t_min = t / 60

# Adapter t_min à la longueur réelle des courbes
t_min = t_min[:voxel_time_series.shape[1]]

# Définir le modèle gamma-varié
def gammavariate_model_dec(t, A1, A2, A3, lambda1, lambda2, lambda3):
    Cpmodel = (A1 - A2 - A3) * np.exp(-lambda1 * t) + A2 * np.exp(-lambda2 * t) + A3 * np.exp(-lambda3 * t)
    return Cpmodel

# Visualisation des courbes les plus typées carotide + Fit modèle de Feng
plt.figure(figsize=(10, 6))
carotid_like_voxels = []
best_coords = []

for i in range(n_clusters):
    cluster_mask = labels == i
    cluster_voxels = voxel_time_series[cluster_mask]

    time_to_peak = np.argmax(cluster_voxels, axis=1)
    auc = cluster_voxels.sum(axis=1)
    score = auc - 10 * time_to_peak

    best_idx_in_cluster = np.argmax(score)
    global_idx = np.where(cluster_mask)[0][best_idx_in_cluster]
    best_curve = cluster_voxels[best_idx_in_cluster]
    coord = (x[global_idx], y[global_idx], z[global_idx])

    carotid_like_voxels.append(best_curve)
    best_coords.append(coord)

    # Fit de la partie décroissante avec le modèle de Feng
    y_data = best_curve
    t_trimmed = t_min[:len(y_data)]
    idx_max = np.argmax(y_data)
    xx_dec = t_trimmed[idx_max:]
    yy_dec = y_data[idx_max:]

    try:
        # Définir les bornes pour éviter l'overflow
        bounds = (
            [0, 0, 0, 0, 0, 0],       # lower bounds
            [1e4, 1e4, 1e4, 10, 10, 10]  # upper bounds
        )
        p0 = [500, 200, 100, 0.1, 0.05, 0.01]

        params_dec, _ = curve_fit(
            gammavariate_model_dec, xx_dec, yy_dec,
            p0=p0, bounds=bounds, maxfev=1000000
        )

        IDIF_fit = gammavariate_model_dec(xx_dec, *params_dec)

        # Affichage
        plt.plot(t_trimmed, y_data, 'o-', label=f'Cluster {i} @ {coord}')
        plt.plot(xx_dec, IDIF_fit, '--', label=f'Modèle Feng - Cluster {i}')

    except (RuntimeError, ValueError) as e:
        plt.plot(t_trimmed, y_data, 'o-', label=f'Cluster {i} @ {coord} (fit échoué)')
        print(f"Cluster {i} - Fit échoué : {e}")

plt.xlabel("Temps (minutes)")
plt.ylabel("Intensité du voxel")
plt.title("Courbes des voxels typés carotide + ajustement modèle Feng")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Affichage des coordonnées exactes
for i, (xi, yi, zi) in enumerate(best_coords):
    print(f"Cluster {i} - Voxel représentatif : Voxel indices (x, y, z) = ({xi}, {yi}, {zi})")