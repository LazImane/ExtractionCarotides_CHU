import os
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import nibabel as nib 
from scipy.optimize import curve_fit 
import time

from matplotlib import patches, ticker

from nilearn import datasets, plotting
from nilearn.image import get_data, index_img, mean_img
from nilearn.regions import Parcellations
from nilearn import plotting
from nilearn import image
from nilearn.plotting import plot_roi
from nilearn.image import index_img
from scipy.cluster.hierarchy import dendrogram,linkage
import random
from nilearn.masking import compute_brain_mask

def cluster1belongstobrain(voxel_coordinates, brain_mask_coords):
    """
    entree :
        voxel_coordinates : coordonnées des voxels du cluster à vérifier
        brain_mask_coords : coordonnées du cerveau extraites du masque
    sortie :
        bool : True si tous les voxels sont dans la moitié GAUCHE du cerveau, False sinon
    """
    x_min, x_max = brain_mask_coords[:, 0].min(), brain_mask_coords[:, 0].max()
    long = x_max - x_min
    max_gauche = long // 2
    for coord in voxel_coordinates:
        if tuple(coord)[0] < x_min or tuple(coord)[0] > max_gauche:
            print("this tuple is outside", coord)
            return False
    return True 

def cluster2belongstobrain(voxel_coordinates, brain_mask_coords):
    """
    entree :
        voxel_coordinates  : coordonnées des voxels du cluster à vérifier
        brain_mask_coords  : coordonnées du cerveau extraites du masque
    sortie :
        bool : True si tous les voxels sont dans la moitié DROITE du cerveau, False sinon
    """
    x_min, x_max = brain_mask_coords[:, 0].min(), brain_mask_coords[:, 0].max()
    long = x_max - x_min
    min_droite = long // 2
    for coord in voxel_coordinates:
        if tuple(coord)[0] < min_droite or tuple(coord)[0] > x_max:
            print("this tuple is outside", coord)
            return False
    return True 


def charger_image_et_mask(data_path, patient_filename, frame_index):
    """
    entree :
        data_path (str) : chemin vers le dossier contenant les données patient
        patient_filename (str) : nom fichier NIfTI du patient (ex: "p04.nii")
        frame_index (int) : index de l'image 3D à extraire du fichier 4D, A MODIF POUR CHAQUE PATIENT!!
    sortie :
        tuple (nifti_img1, nifti_img_3D, brain_mask, brain_mask_coords)
        - nifti_img1 :   image 4D chargée
        - nifti_img_3D : frame 3D extraite
        - brain_mask :   masque du cerveau calculé
        - brain_mask_coords : coordonnées des voxels du masque
    """
    filename_img_4D = os.path.join(data_path, patient_filename)
    nifti_img1 = nib.load(filename_img_4D)
    nifti_img_3D = index_img(nifti_img1, frame_index) #ici j'utilise un frame différent qui peut être l'optimal pour detecter le cerveau 

    #Extraction données cerveau a partir de la bibliothéque nilearn: nilearn.masking.compute_brain_mask
    brain_mask = compute_brain_mask(nifti_img_3D, threshold=0.0)
    plot_roi(brain_mask, bg_img=nifti_img_3D, title='Brain Mask over Anatomical Image')
    plt.show()
    #Extraction des coordonnées du masque
    brain_mask_coords = np.argwhere(brain_mask.get_fdata())
    print("coordonnées du mask ", brain_mask_coords)

    return nifti_img1, nifti_img_3D, brain_mask, brain_mask_coords


def sauvegarder_brain_mask(brain_mask_coords, nifti_img_3D):
    """
    entree :
        brain_mask_coords (array)  : coordonnées des voxels du cerveau
        nifti_img_3D (Nifti image) : image anatomique de référence 
    sortie :
        None (sauvegarde le fichier Nifti contenant le masque)
    """
    # Get the shape and affine from your original image
    shape = nifti_img_3D.shape
    affine = nifti_img_3D.affine

    # Create an empty mask
    brain_mask_array = np.zeros(shape, dtype=np.uint8)

    # Fill in the coordinates from your list
    for coord in brain_mask_coords:
        x, y, z = coord
        brain_mask_array[x, y, z] = 1
    brain_mask_nifti = nib.Nifti1Image(brain_mask_array, affine)
    nib.save(brain_mask_nifti, "brain_mask_from_coords.nii")



def segmenter_image(nifti_img_3D):
    """
    entree :
        nifti_img_3D (Nifti1Image) : image anatomique 3D
    sortie :
        roi_mask (Nifti1Image) : masque des region of interest a partir d'un seuil (23)
    """
    roi_mask = image.math_img("img > 23", img=nifti_img_3D)
    plot_roi(roi_mask, title="Selected Brain Region Mask")
    plt.show()
    return roi_mask


def make_parcellation(nifti_img1, roi_mask, n_parcels=17):
    """
    entree :
        nifti_img1 (Nifti1Image) : image fonctionnelle 4D
        roi_mask (Nifti1Image) : masque des régions d'intérêt
        n_parcels (int) : nombre de clusters souhaités
    sortie :
        Parcellations : objet "Ward" contenant les clusters
    """
    start = time.time()
    ward = Parcellations(
        method="ward",
        n_parcels=n_parcels,
        standardize=True,
        mask=roi_mask,
        smoothing_fwhm=2.0,
        memory="nilearn_cache",
        memory_level=1,
        verbose=1,
    )
    ward.fit(nifti_img1)
    print(f"Ward agglomeration {n_parcels} clusters: {time.time() - start:.2f}s")
    return ward


def save_parcellation(ward_labels_img):
    """
    entree :
        ward_labels_img (Nifti1Image) : image contenant les labels de notre parcellation
    sortie :
        None (sauvegarde le fichier NIfTI des labels)
    """
    from pathlib import Path
    output_dir = Path.cwd() / "results" / "plot_data_driven_parcellations"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output will be saved to: {output_dir}")
    ward_labels_img.to_filename(output_dir / "ward_parcellation.nii.gz")


def plot_parcellation(ward_labels_img):
    """
    action : affiche la parcellation et retourne les coordonnées de la coupe (yxz)
    """
    first_plot = plotting.plot_roi(ward_labels_img, title="parcellation:", display_mode="xz")
    plotting.show()# Grab cut coordinates from this plot to use as a common for all plots !!!!
    return first_plot.cut_coords


def trouver_cluster_intense(ward_labels_img, nifti_img_3D):
    """
    entree :
        ward_labels_img (Nifti1Image) : image contenant les labels de parcellation
        nifti_img_3D (Nifti1Image) : image anatomique 3D
    sortie :
        list of tuple : liste des clusters et de leur intensité moyenne, triée par intensité décroissante
    """
    parcellations_data = ward_labels_img.get_fdata()
    
    #on calcule l'intensité moyenne de chaque cluster
    intensite_clusters = []
    for cluster in np.unique(parcellations_data):
        if cluster == 0:
            continue
        condition = (parcellations_data == cluster)
        cluster_intensity = np.mean(nifti_img_3D.get_fdata()[condition])
        intensite_clusters.append((cluster, cluster_intensity))
    intensite_clusters.sort(key=lambda x: x[1], reverse=True)
    print(intensite_clusters)
    return intensite_clusters

def afficher_cluster(ward_labels_img, clustertrouve, nifti_img_3D):
    """
    entree :
        ward_labels_img (Nifti1Image) : image contenant les labels de parcellation
        clustertrouve (int) : numéro du cluster à afficher
        nifti_img_3D (Nifti1Image) : image anatomique 3D
    sortie :
        Nifti1Image : masque binaire du cluster sélectionné
    """
    # Create a binary mask for the cluster chosen
    cluster_mask = image.math_img(f"img == {clustertrouve}", img=ward_labels_img)

    # Plot the cluster on a brain slice
    plotting.plot_roi(
        cluster_mask,
        bg_img=nifti_img_3D,
        title="Cluster Visualization",
        display_mode="ortho",
        cut_coords=None,
        cmap="autumn",
    )
    plotting.show()
    return cluster_mask

def trouver_clusters_candidats(intensite_clusters, ward_labels_img, nifti_img_3D, brain_mask_coords):
    """
    entree :
        intensite_clusters (list of tuple) : clusters triés par intensité
        ward_labels_img (Nifti1Image) : image contenant les labels de parcellation
        nifti_img_3D (Nifti1Image) : image anatomique 3D
        brain_mask_coords (np.ndarray) : coordonnées des voxels du cerveau
    sortie :
        tuple (np.ndarray, np.ndarray) : coordonnées des voxels des deux clusters candidats (gauche et droite)
    """
    i = 0
    left_voxel_coordinates = None
    right_voxel_coordinates = None
    
    while True:
        clustertrouve = intensite_clusters[i][0]
        print(f"Cluster with the highest intensity: {clustertrouve}, with an intensity of : {intensite_clusters[i][1]}")
        cluster_mask = afficher_cluster(ward_labels_img, clustertrouve, nifti_img_3D)
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


### ----------------------------CODE PRINCIPAL------------------------------###
def main():
    data_path = "../PROJET_INFO_S6/patients_cam/"  # A CHANGER
    patient_filename = "p04.nii"
    
    nifti_img1, nifti_img_3D, brain_mask, brain_mask_coords = charger_image_et_mask(
        data_path, 
        patient_filename, 
        frame_index=19 # >>> A MODIFIER POUR CHAQUE PATIENT
    )


    
    sauvegarder_brain_mask(brain_mask_coords, nifti_img_3D)
    # Attention : ici changement de frame
    nifti_img_3D = index_img(nifti_img1, 2)  # A MODIFIER pour chaque patient
    
    roi_mask = segmenter_image(nifti_img_3D)
    ward = make_parcellation(nifti_img1, roi_mask, n_parcels=17)
    ward_labels_img = ward.labels_img_ # transformer ward en un object image nifti 
    
    save_parcellation(ward_labels_img)
    cut_coords = plot_parcellation(ward_labels_img)
    
    intensite_clusters = trouver_cluster_intense(ward_labels_img, nifti_img_3D)
    left_voxels, right_voxels = trouver_clusters_candidats(
        intensite_clusters, ward_labels_img, nifti_img_3D, brain_mask_coords
    )
    
    print("Left voxels coordinates:", left_voxels)
    print("Right voxels coordinates:", right_voxels)

    # cam -> je fini ca demain, je pusherais quand ca sera joli
        #afficher_left_right_TAC(nifti_img1, brain_mask_coords)
        #afficher_TAC_moyenne(nifti_img1, brain_mask_coords)
        #afficher_all_TAC(nifti_img1, brain_mask_coords)

#lancement
if __name__ == "__main__":
    main()
