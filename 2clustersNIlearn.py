import os
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import nibabel as nib 
from scipy.optimize import curve_fit 
import time

import numpy as np
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
#from nilearn.masking import compute_background_mask
####strat :  on a les coordonnées du brain mask maintenant on extrait x max et min ( extremités de l'axe coronal pour extraire tout ce qui à l'éxtérieur du cerveau)
def cluster1belongstobrain(voxel_coordinates, brain_mask_coords):
    x_min, x_max = brain_mask_coords[:, 0].min(), brain_mask_coords[:, 0].max()
    long = x_max - x_min
    max_gauche = long // 2
    for coord in voxel_coordinates:
        if tuple(coord)[0] < x_min or tuple(coord)[0]> max_gauche:
            print("this tuple is outside", coord)
            return False
    return True 
def cluster2belongstobrain(voxel_coordinates, brain_mask_coords):
    x_min, x_max = brain_mask_coords[:, 0].min(), brain_mask_coords[:, 0].max()
    long = x_max - x_min
    min_droite = long //2
    for coord in voxel_coordinates:
        if tuple(coord)[0] < min_droite or tuple(coord)[0]> x_max:
            print("this tuple is outside", coord)
            return False
    return True 
###pour trouver les 2 carotides
##strat : (x_max - x_min)/ 2 
data_path = "FICHIERSDONNEES/" ## A CHANGER
filename_img_4D =  os.path.join(data_path, "p04.nii")

# Charger le fichier NIfTI 4D
nifti_img1 = nib.load(filename_img_4D)
nifti_img_3D = index_img(nifti_img1,19) #on choisit le temps !!A MODIFIER POUR CHAQUE PATIENT
#ici j'utilise un frame différent qui peut être l'optimal pour detecter le cerveau 

#extraire données cerveau a partir de la bibliothéque nilearn: nilearn.masking.compute_brain_mask
brain_mask = compute_brain_mask(nifti_img_3D, threshold=0.0)
plot_roi(brain_mask, bg_img=nifti_img_3D, title='Brain Mask over Anatomical Image')
plt.show()
mask_data = brain_mask.get_fdata()
#extraire les données du masque
brain_mask_coords= np.argwhere(mask_data)
print("coordonnées du mask ",brain_mask_coords)
# Obtenir les données sous forme de tableau NumPy
#img_4D = nifti_img.get_fdata()[12][12][4][3] #on recupere le frame interessant 
#!!! on met le frame le plus intéressant sur itk-snap -1 pour décalage de 0
nifti_img_3D = index_img(nifti_img1,2) #on choisit le temps !!A MODIFIER POUR CHAQUE PATIENT
####partie temporaire pour visualiser le mask cerveau sur itk-snap
import numpy as np
import nibabel as nib

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

#####end 
roi_mask = image.math_img("img > 23", img=nifti_img_3D) #seuiller img 
plot_roi(roi_mask, title="Selected Brain Region Mask") #visualiser le mask 
start = time.time()
# un mask est un des paramétres de Parcellation and it could be useful to choose specific voxels to apply clustering to 
ward = Parcellations(
    method="ward",
    n_parcels=17, #creating n regions of interest ROI
    standardize=True,
    mask = roi_mask, 
    smoothing_fwhm=2.0, #smoothing data to reduce noise
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
ward.fit(nifti_img1) #fit la parcellation sur l'image NIFTI 
#X = nifti_img1.get_fdata().reshape(-1, 1)  

print(f"Ward agglomeration 10 clusters: {time.time() - start:.2f}s")
ward_labels_img = ward.labels_img_ #Class for single file NIfTI1 format image (????) 
# transformer ward en un object image nifti 
from pathlib import Path
# (maintenant qu'on a transformé ward_labels_img en NIFTIIMAGE) :  save the results in a .nii file #optionnel 
output_dir = Path.cwd() / "results" / "plot_data_driven_parcellations"
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Output will be saved to: {output_dir}")
ward_labels_img.to_filename(output_dir / "ward_parcellation.nii.gz")

#plotting the cluster data 
first_plot = plotting.plot_roi(
    ward_labels_img, title="parcellation:", display_mode="xz"
)
plotting.show()
# Grab cut coordinates from this plot to use as a common for all plots !!!!
cut_coords = first_plot.cut_coords
#print("so this is what cut coords does ? ",cut_coords)
# EXTRA DETAILS SUR LES CLUSTERS TROUVEES 
data_path = "results/plot_data_driven_parcellations/" ## A CHANGER
filename_img_4D =  os.path.join(data_path, "ward_parcellation.nii.gz"); 
#on prend la data des clusters pour la traiter
parcellations_data = ward_labels_img.get_fdata()

#on calcule l'intensité moyenne de chaque cluster
intensite_clusters=[]
for cluster in np.unique(parcellations_data):
    if cluster == 0 : #on ignore le 'bruit'
        continue
    condition = (parcellations_data == cluster)
    cluster_intensity = np.mean(nifti_img_3D.get_fdata()[condition])
    intensite_clusters.append((cluster,cluster_intensity))
intensite_clusters.sort(key=lambda x: x[1], reverse=True)
print(intensite_clusters)
# Identify the cluster with the highest mean intensity
clustertrouve = intensite_clusters[0][0]
print(f"Cluster with the highest intensity: {clustertrouve}, with an intensity of : {intensite_clusters[0][1]}")

cluster_7_mask = image.math_img(f"img == {clustertrouve}", img=ward_labels_img)
# Create a binary mask for the cluster chosen

# Plot the cluster on a brain slice
plotting.plot_roi(
    cluster_7_mask,
    bg_img=nifti_img_3D,  # Use the 3D image as the background
    title="Cluster Visualization",
    display_mode="ortho",  # Orthogonal view (axial, sagittal, coronal)
    cut_coords=None,  # Use the cut coordinates from earlier
    cmap="autumn",  # Color map for the cluster
)
plotting.show()
datatotreat= cluster_7_mask.get_fdata()
#premiers voxels à choisir 
voxel_coordinates = np.argwhere(datatotreat)
print("this is the data to treat:",voxel_coordinates)
left_voxel_coordinates = voxel_coordinates
right_voxel_coordinates = voxel_coordinates
#brain_mask_set = set(map(tuple, brain_mask_coords))
i=0
while(not (cluster1belongstobrain(left_voxel_coordinates, brain_mask_coords))):
    clustertrouve = intensite_clusters[i][0]
    print(f"Cluster with the highest intensity: {clustertrouve}, with an intensity of : {intensite_clusters[i][1]}")

    cluster_7_mask = image.math_img(f"img == {clustertrouve}", img=ward_labels_img)
    # Create a binary mask for cluster 7

    # Plot the cluster on a brain slice
    plotting.plot_roi(
        cluster_7_mask,
        bg_img=nifti_img_3D,  # Use the 3D image as the background
        title="Cluster Visualization",
        display_mode="ortho",  # Orthogonal view (axial, sagittal, coronal)
        cut_coords=None,  # Use the cut coordinates from earlier
        cmap="autumn",  # Color map for the cluster
    )
    plotting.show()
    datatotreat= cluster_7_mask.get_fdata()

    left_voxel_coordinates = np.argwhere(datatotreat)
    print("this is the data to treat for left side:",left_voxel_coordinates)
    i+=1
while(not (cluster2belongstobrain(right_voxel_coordinates, brain_mask_coords))):
    clustertrouve = intensite_clusters[i][0]
    print(f"Cluster with the highest intensity: {clustertrouve}, with an intensity of : {intensite_clusters[i][1]}")

    cluster_7_mask = image.math_img(f"img == {clustertrouve}", img=ward_labels_img)
    # Create a binary mask for cluster 7

    # Plot the cluster on a brain slice
    plotting.plot_roi(
        cluster_7_mask,
        bg_img=nifti_img_3D,  # Use the 3D image as the background
        title="Cluster Visualization",
        display_mode="ortho",  # Orthogonal view (axial, sagittal, coronal)
        cut_coords=None,  # Use the cut coordinates from earlier
        cmap="autumn",  # Color map for the cluster
    )
    plotting.show()
    datatotreat= cluster_7_mask.get_fdata()

    right_voxel_coordinates = np.argwhere(datatotreat)
    print("this is the data to treat for right side:",right_voxel_coordinates)
    i+=1

"""while not(clusterbelongstobrain(voxel_coordinates,brain_mask_set)):
    clustertrouve = intensite_clusters[i][0]
    print(f"Cluster with the highest intensity: {clustertrouve}, with an intensity of : {intensite_clusters[i][1]}")

    cluster_7_mask = image.math_img(f"img == {clustertrouve}", img=ward_labels_img)
    # Create a binary mask for cluster 7

    # Plot the cluster on a brain slice
    plotting.plot_roi(
        cluster_7_mask,
        bg_img=nifti_img_3D,  # Use the 3D image as the background
        title="Cluster Visualization",
        display_mode="ortho",  # Orthogonal view (axial, sagittal, coronal)
        cut_coords=None,  # Use the cut coordinates from earlier
        cmap="autumn",  # Color map for the cluster
    )
    plotting.show()
    datatotreat= cluster_7_mask.get_fdata()

    voxel_coordinates = np.argwhere(datatotreat)
    print("this is the data to treat:",voxel_coordinates)
    i+=1
"""





    
    
print('if you didnt recieve the past message its all good !')
nifti_img = nib.load(filename_img_4D)
img_4D = nifti_img.get_fdata()
print(img_4D.shape) 
print("max value: ", img_4D.max())
print("min value:", img_4D.min())
unique_clusters = np.unique(img_4D)
print("unique clusters", unique_clusters) #nous montre les clusters trouvé d'aprés parcellations 
cluster_label = unique_clusters[1]  # Example: second cluster
cluster_data = (img_4D == cluster_label)  # Boolean mask for the cluster

# Optionally get voxel indices for the cluster
voxel_indices = np.argwhere(cluster_data)
print("Number of voxels in cluster:", len(voxel_indices))

plotting.plot_roi("results\plot_data_driven_parcellations\ward_parcellation.nii.gz", title="Clustered Data")
plotting.show()

# Save data to a CSV
np.savetxt("cluster_data.csv", unique_clusters.flatten(), delimiter=",")


#on recupére le data de l'image principale 
dataNIFTI = nifti_img1.get_fdata()
#puis les données de l'image clustered 
data_path = "results/plot_data_driven_parcellations/" ## A CHANGER
filename_img_4D =  os.path.join(data_path, "ward_parcellation.nii.gz"); 

nifti_img = nib.load(filename_img_4D)
img_4D = nifti_img.get_fdata()
i,j,k=0,0,0

#on recupere un point d'interet #on peut aussi choisir le point avec la max intensité 
indice = np.random.randint(0,len(left_voxel_coordinates))
x,y,z = left_voxel_coordinates[indice]
"""voxel = dataNIFTI[x,y,z,:]
print("le point qui (most likely) represente la carothide:" , voxel)"""
listofpoints = []
for i in range(len(left_voxel_coordinates)):
    x,y,z = left_voxel_coordinates[i]
    listofpoints.append(dataNIFTI[x,y,z,:])
nplistofpoint= np.array(listofpoints)
voxel_left = nplistofpoint.mean(axis=0)
print(voxel_left)

indice = np.random.randint(0,len(right_voxel_coordinates))
x,y,z = right_voxel_coordinates[indice]
"""voxel = dataNIFTI[x,y,z,:]
print("le point qui (most likely) represente la carothide:" , voxel)"""
listofpoints = []
for i in range(len(right_voxel_coordinates)):
    x,y,z = right_voxel_coordinates[i]
    listofpoints.append(dataNIFTI[x,y,z,:])
nplistofpoint= np.array(listofpoints)
voxel_right = nplistofpoint.mean(axis=0)
print(voxel_right)


#tep = ward.transform(datasets.func)
# le temps en fontion de minutes 
t = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,150,180,210,240,270,300,360,420,480,540,600, 900, 1200, 1500, 1800, 2100])
t_min = t/60
delta_i = np.diff(t_min) # Durée de chaque frame en min*

# prendre un voxel et extraire les données voisins (intensité) par rapport au temps ajouter un 0 au début AND ITS DONE!!!!
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
y_data = voxel_left # Contient les valeurs de la TAC
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


#extraire les valeurs des voxels spécifiés 


#time_pointscube = range(len(cube))  # Index temporels (ou utilisez une échelle réelle si disponible)

#on choisit un voxel dans la carotide d'abord 

#plt.plot(time_pointscube, cube, marker='o')
plt.plot(t_min, voxel_left, 'go-', label="Données réelles du voxel", markersize=4)
plt.plot(t_min, IDIF_fit, 'm--', label="Modèle gamma-varié du voxel")
#plt.title(f'TAC pour le cube du voxel ({x}, {y}, {z})')
plt.xlabel('Temps')
plt.ylabel('Intensité')
plt.grid(True)
plt.show()






t_data = t_min # Tableau contenant toutes les valeurs de temps des acquisitions en minutes
y_data = voxel_right # Contient les valeurs de la TAC
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


#extraire les valeurs des voxels spécifiés 


#time_pointscube = range(len(cube))  # Index temporels (ou utilisez une échelle réelle si disponible)

#on choisit un voxel dans la carotide d'abord 

#plt.plot(time_pointscube, cube, marker='o')
plt.plot(t_min, voxel_right, 'go-', label="Données réelles du voxel", markersize=4)
plt.plot(t_min, IDIF_fit, 'm--', label="Modèle gamma-varié du voxel")
#plt.title(f'TAC pour le cube du voxel ({x}, {y}, {z})')
plt.xlabel('Temps')
plt.ylabel('Intensité')
plt.grid(True)
plt.show()