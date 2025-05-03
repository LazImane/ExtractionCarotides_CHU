import os
import numpy as np
import nibabel as nib
from nilearn import plotting, image, masking
from nilearn.masking import compute_background_mask, compute_brain_mask
from scipy.ndimage import binary_erosion, binary_closing, label, binary_dilation
from skimage import filters, measure, morphology
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans, MeanShift
from nilearn.regions import Parcellations
from nilearn.image import index_img

# Chargement des données

def load_patient_data(path, patient_id):
    filename = os.path.join(path, patient_id + ".nii")
    nifti = nib.load(filename)
    return nifti, nifti.get_fdata(), nifti.affine

# Préparation des masques

def get_brain_mask_optimized(data, affine, threshold=0.5, opening=2, connected=True, post_process=True):
    mean_data = np.mean(data, axis=-1)
    mask = compute_brain_mask(nib.Nifti1Image(mean_data, affine), threshold=threshold, opening=opening, connected=connected).get_fdata().astype(bool)
    if post_process:
        mask = binary_closing(mask, structure=np.ones((3, 3, 3)))
        mask = binary_erosion(mask, iterations=1)
    return mask

def get_lower_brain_mask(brain_mask):
    lower_mask = np.zeros_like(brain_mask, dtype=bool)
    X, Y, Z = brain_mask.shape
    for x in range(X):
        for y in range(Y):
            brain_voxels = np.where(brain_mask[x, y, :])[0]
            if brain_voxels.size == 0:
                continue
            z_bottom = brain_voxels.min()
            z_depth = brain_voxels.size
            offset = int(0.25 * z_depth)
            z_start = max(z_bottom - z_depth, 0)
            z_end = z_bottom
            lower_mask[x, y, z_start:z_end] = True
    return lower_mask

def prepare_masks(data, affine):
    brain_mask = get_brain_mask_optimized(data, affine).astype(np.int8)
    lower_mask = get_lower_brain_mask(brain_mask).astype(np.int8)
    return brain_mask, lower_mask

# Calcul des intensités moyennes

def calculate_mean_intensities_per_frame(nft, mask):
    intensities_per_volume = []
    num_volumes = nft.shape[-1]
    for i in range(num_volumes):
        volume = image.index_img(nft, i).get_fdata()
        masked_data = volume[mask > 0]
        mean_intensity = masked_data.mean() if masked_data.size > 0 else 0
        intensities_per_volume.append(mean_intensity)
    return intensities_per_volume

def find_best_volume(intensities_per_volume):
    max_int = np.max(intensities_per_volume)
    entry_seuil = 0.7 * max_int
    return next((i for i, val in enumerate(intensities_per_volume) if val >= entry_seuil), np.argmax(intensities_per_volume))

# Segmentation par parcellation

def thresholdmask(nft_img, frame_index):
    nifti_img_3D = index_img(nft_img, frame_index)
    flatdata = nifti_img_3D.get_fdata().flatten()
    flatdata = flatdata[flatdata > 5]
    threshold = np.percentile(flatdata, 97)
    return image.math_img(f"img > {threshold}", img=nifti_img_3D)

def Parcellationclusters(nifti_img1, roi_mask, n_parcels=30):
    ward = Parcellations(method="ward", n_parcels=n_parcels, standardize=True, mask=roi_mask, smoothing_fwhm=2.0)
    ward.fit(nifti_img1)
    return ward.labels_img_

def trouver_cluster_intense(ward_labels_img, nifti_img_3D):
    parcellations_data = ward_labels_img.get_fdata()
    intensite_clusters = []
    for cluster in np.unique(parcellations_data):
        if cluster == 0:
            continue
        condition = (parcellations_data == cluster)
        intensity = np.mean(nifti_img_3D.get_fdata()[condition])
        intensite_clusters.append((cluster, intensity))
    intensite_clusters.sort(key=lambda x: x[1], reverse=True)
    return intensite_clusters

def cluster1belongstobrain(voxel_coordinates, brain_mask_coords):
    min_x = min(brain_mask_coords[0])
    max_x = max(brain_mask_coords[0])
    long = max_x - min_x
    max_gauche = long // 2
    for coord in voxel_coordinates:
        if coord[0] < min_x - 1 or coord[0] > max_gauche - 1:
            return False
    return True

def cluster2belongstobrain(voxel_coordinates, brain_mask_coords):
    min_x = min(brain_mask_coords[0])
    max_x = max(brain_mask_coords[0])
    long = max_x - min_x
    min_droite = long // 2
    for coord in voxel_coordinates:
        if coord[0] < min_droite or coord[0] > max_x:
            return False
    return True

def trouver_clusters_candidats(intensite_clusters, ward_labels_img, nifti_img_3D, brain_mask_coords):
    i = 0
    left_voxel_coordinates = None
    right_voxel_coordinates = None
    while True:
        cluster_id = intensite_clusters[i][0]
        cluster_mask = image.math_img(f"img == {cluster_id}", img=ward_labels_img)
        datatotreat = cluster_mask.get_fdata()
        voxel_coordinates = np.argwhere(datatotreat)
        if left_voxel_coordinates is None and cluster1belongstobrain(voxel_coordinates, brain_mask_coords):
            left_voxel_coordinates = voxel_coordinates
        elif right_voxel_coordinates is None and cluster2belongstobrain(voxel_coordinates, brain_mask_coords):
            right_voxel_coordinates = voxel_coordinates
        if left_voxel_coordinates is not None and right_voxel_coordinates is not None:
            break
        i += 1
    return left_voxel_coordinates, right_voxel_coordinates

def segment_carotids_by_parcellation(nft_img, lower_mask, best_frame_idx):
    threshold_mask = thresholdmask(nft_img, best_frame_idx - 1)
    ward_images = Parcellationclusters(nft_img, threshold_mask)
    nifti_3D = index_img(nft_img, best_frame_idx - 1)
    intensites = trouver_cluster_intense(ward_images, nifti_3D)
    coords = np.where(lower_mask)
    left, right = trouver_clusters_candidats(intensites, ward_images, nifti_3D, coords)
    return left, right, coords

# Extraction des TAC

def findingpointsforplot(voxel_coordinates, dataNIFTI):
    listofpoints = [dataNIFTI[x, y, z, :] for x, y, z in voxel_coordinates]
    return np.mean(np.array(listofpoints), axis=0)

def extract_TACs(data, left_coords, right_coords):
    combined_coords = np.vstack((left_coords, right_coords))
    left_TAC = [0] + list(findingpointsforplot(left_coords, data))
    right_TAC = [0] + list(findingpointsforplot(right_coords, data))
    combined_TAC = [0] + list(findingpointsforplot(combined_coords, data))
    return left_TAC, right_TAC, combined_TAC

# Fit de la courbe

def get_IDIF_FIT(nft_img, volume_data_list):
    img_4D = nft_img.get_fdata()
    t = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,150,180,210,240,270,300,360,420,480,540,600,900,1200,1500,1800,2100,2400])
    t_min = t / 60
    def safe_exp(x, max_exp=700):
        return np.exp(np.clip(x, -max_exp, max_exp))
    def model(t, A1, A2, A3, l1, l2, l3):
        return (A1 - A2 - A3) * safe_exp(-l1 * t) + A2 * safe_exp(-l2 * t) + A3 * safe_exp(-l3 * t)
    idx_max = np.argmax(volume_data_list)
    xx_dec, yy_dec = t_min[idx_max:], volume_data_list[idx_max:]
    if len(xx_dec) != len(yy_dec):
        min_len = min(len(xx_dec), len(yy_dec))
        xx_dec, yy_dec = xx_dec[:min_len], yy_dec[:min_len]
    xx_removed = t_min[:idx_max]
    yy_removed = [0] * len(xx_removed)
    params, _ = curve_fit(model, xx_dec, yy_dec, maxfev=1000000)
    IDIF_fit = np.concatenate((yy_removed, model(xx_dec, *params)))
    return t_min, np.concatenate((xx_removed, xx_dec)), IDIF_fit

# Affichage final

def fit_and_plot_all_TACs(nft_img, patient_id, left, right, combined, reference):
    courbes = [reference, combined, left, right]
    noms = ["prof", "carotide", "carotide_gauche", "carotide_droite"]
    fitted, axes = [], []
    for courbe in courbes:
        t_min, x_vals, idif = get_IDIF_FIT(nft_img, courbe)
        axes.append(x_vals)
        fitted.append(idif)
    plt.figure(figsize=(16, 10))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(axes[i], courbes[i], 'bo-', label="Données expérimentales", markersize=4)
        plt.plot(axes[i], fitted[i], 'r--', label="Modèle ajusté")
        plt.title(f"Ajustement TAC - {noms[i]}")
        plt.xlabel("Temps (min)")
        plt.ylabel("Concentration (kBq/cc)")
        plt.grid()
        plt.legend()
    plt.suptitle(f"Ajustement des TAC - {patient_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"TAC_fit_{patient_id}.png", dpi=300)
    plt.close()

# Lecture fichier CRV

def open_crv(filename):
    values = []
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            _, val = line.strip().split('\t')
            val = float(val)
            if val > 0:
                values.append(val)
    return np.array(values)

def calcul_moy_intensite(data, mask):
    num_volumes = data.shape[-1]
    moy = []
    mask_indices = np.where(mask)
    for i in range(num_volumes):
        masked_values = data[mask_indices[0], mask_indices[1], mask_indices[2], i]
        m = np.mean(masked_values) if masked_values.size > 0 else 0
        moy.append(m)
    return moy

# Ajout d'affichage interactif dans fit_and_plot_all_TACs

def fit_and_plot_all_TACs(nft_img, patient_id, left, right, combined, reference):
    courbes = [reference, combined, left, right]
    noms = ["prof", "carotide", "carotide_gauche", "carotide_droite"]
    fitted, axes = [], []
    for courbe in courbes:
        t_min, x_vals, idif = get_IDIF_FIT(nft_img, courbe)
        axes.append(x_vals)
        fitted.append(idif)
    plt.figure(figsize=(16, 10))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(axes[i], courbes[i], 'bo-', label="Données expérimentales", markersize=4)
        plt.plot(axes[i], fitted[i], 'r--', label="Modèle ajusté")
        plt.title(f"Ajustement TAC - {noms[i]}")
        plt.xlabel("Temps (min)")
        plt.ylabel("Concentration (kBq/cc)")
        plt.grid()
        plt.legend()
    plt.suptitle(f"Ajustement des TAC - {patient_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"TAC_fit_{patient_id}.png", dpi=300)
    plt.show()
    plt.close()

# Fonction principale

def mainFunction(patient_id):
    path = "./Données_Patients"
    nft_img, data, affine = load_patient_data(path, patient_id)
    brain_mask, lower_mask = prepare_masks(data, affine)
    intensities = calculate_mean_intensities_per_frame(nft_img, lower_mask)
    best_frame_idx = find_best_volume(intensities)
    left_coords, right_coords, brain_coords = segment_carotids_by_parcellation(nft_img, lower_mask, best_frame_idx)
    left_TAC, right_TAC, combined_TAC = extract_TACs(data, left_coords, right_coords)
    prof_crv = open_crv(f"crv/crv/{patient_id}_carot.crv")
    prof_crv = [0] + list(prof_crv)
    fit_and_plot_all_TACs(nft_img, patient_id, left_TAC, right_TAC, combined_TAC, prof_crv)
    print(f"TAC_fit_{patient_id}.png saved")

# Exécution principale

def main():
    p = input("Entrez l'id du patient (e.g : P04) : ")
    mainFunction(p)

if __name__ == "__main__":
    main()
