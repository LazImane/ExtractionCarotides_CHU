import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.image import index_img, math_img
from nilearn.masking import compute_brain_mask
from nilearn import plotting
from nilearn.regions import Parcellations
from process_idif_cluster import process_cluster

# === CONFIGURATION ===
data_path = "Données_Patients"
result_dir = "resultats_fonctionalia"
os.makedirs(result_dir, exist_ok=True)

n_clusters = 17
frame_for_seg = 19
frame_for_clustering = 2

# === Lister tous les fichiers .nii
nii_files = [f for f in os.listdir(data_path) if f.endswith(".nii")]

for filename in nii_files:
    try:
        filepath = os.path.join(data_path, filename)
        base_name = os.path.splitext(filename)[0]
        prefix = f"resultat_{base_name}"

        print(f"\n Traitement de : {filename}")

        # === Charger l'image 4D
        nifti_img4D = nib.load(filepath)

        # === Frame pour le brain_mask
        nifti_img_3D_seg = index_img(nifti_img4D, frame_for_seg)
        brain_mask = compute_brain_mask(nifti_img_3D_seg, threshold=0.0)
        brain_mask_coords = np.argwhere(brain_mask.get_fdata())
        print(" Brain mask OK")

        # === Frame pour clustering carotides
        nifti_img_3D = index_img(nifti_img4D, frame_for_clustering)
        roi_mask = math_img("img > 23", img=nifti_img_3D)
        if np.sum(roi_mask.get_fdata()) == 0:
            raise ValueError("Le masque ROI est vide — signal insuffisant pour clustering.")

        display = plotting.plot_roi(roi_mask, bg_img=nifti_img_3D, title=f"{base_name} – ROI clustering")
        display.savefig(os.path.join(result_dir, f"{prefix}_ROI_clustering.png"))
        display.close()


        # === Ward clustering
        ward = Parcellations(method="ward", n_parcels=n_clusters, standardize=True,
                             mask=roi_mask, smoothing_fwhm=2.0,
                             memory="nilearn_cache", memory_level=1, verbose=0)
        n_voxels = int(np.sum(roi_mask.get_fdata()))
        if n_voxels < n_clusters:
            raise ValueError(f"Trop peu de voxels pour {n_clusters} clusters : {n_voxels} voxels détectés.")

        ward.fit(nifti_img4D)
        ward_labels_img = ward.labels_img_

        # === Trouver le cluster avec intensité maximale
        parcellations_data = ward_labels_img.get_fdata()
        intensities = []
        for cluster_id in np.unique(parcellations_data):
            if cluster_id == 0:
                continue
            mask = parcellations_data == cluster_id
            mean_intensity = np.mean(nifti_img_3D.get_fdata()[mask])
            intensities.append((cluster_id, mean_intensity))

        intensities.sort(key=lambda x: x[1], reverse=True)
        cluster_max_id = intensities[0][0]
        print(f" Cluster le plus intense : {cluster_max_id}")

        # === Récupérer les voxels du cluster max
        best_cluster_mask = math_img(f"img == {cluster_max_id}", img=ward_labels_img)
        display = plotting.plot_roi(best_cluster_mask, bg_img=nifti_img_3D, title=f"{base_name} – Cluster sélectionné")
        display.savefig(os.path.join(result_dir, f"{prefix}_cluster_selection.png"))
        display.close()


        voxel_coordinates = np.argwhere(best_cluster_mask.get_fdata())

        # === Séparation gauche/droite, masks, TACs
        output_prefix = os.path.join(result_dir, f"{prefix}_carotide")
        TAC_gauche, TAC_droite = process_cluster(
            voxel_coordinates,
            brain_mask_coords,
            nifti_img_3D,
            nifti_img4D,
            output_prefix=output_prefix
        )

        # === Tracer les courbes TAC
        t = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,150,180,210,240,270,300,360,420,480,540,600,900,1200,1500,1800,2100])
        t_min = t / 60

        plt.figure(figsize=(10, 5))
        if len(TAC_gauche) != len(t_min) or len(TAC_droite) != len(t_min):
            raise ValueError("Taille invalide de la courbe TAC : nombre de points incompatible.")
        plt.plot(t_min, TAC_gauche, 'r-', label="TAC Carotide Gauche")
        plt.plot(t_min, TAC_droite, 'b-', label="TAC Carotide Droite")
        plt.xlabel("Temps (min)")
        plt.ylabel("Intensité (kBq/cc)")
        plt.title(f"{base_name} – TACs Carotide Gauche/Droite")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"{prefix}_TAC_plot.png"))
        plt.close()

        print(f" Résultats sauvegardés pour {filename}.")

    except Exception as e:
        print(f" Erreur lors du traitement de {filename} : {e}")
