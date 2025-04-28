import os
import numpy as np
import nibabel as nib
from nilearn import plotting ,image,masking
from nilearn.masking import compute_background_mask,compute_brain_mask
from scipy.ndimage import binary_erosion, binary_closing, label,binary_dilation
from skimage import filters, measure,morphology
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans,MeanShift,estimate_bandwidth
from nilearn.regions import Parcellations
from nilearn.image import index_img
from nilearn.plotting import plot_roi


def get_data(path, patient_id):
    """path = where the patients are, patien_id = like "P04" """
    filename = os.path.join(path, patient_id+".nii")
    nifti = nib.load(filename)
    return nifti,nifti.get_fdata(), nifti.affine

def calculate_mean_intensities_per_frame(nft,mask):
    """Calculate mean intensities for each volume."""
    intensities_per_volume = []
    num_volumes = nft.shape[-1]
    nilearn = image.new_img_like(nft, nft.get_fdata(dtype=np.float32), affine=nft.affine)

    for i in range(num_volumes):
        volume = image.index_img(nilearn, i)
        volume_data = volume.get_fdata(dtype=np.float32)
        mask_data = mask * volume_data
        masked_data = volume_data[mask_data > 0]
        mean_intensity = masked_data.mean() if masked_data.size > 0 else 0
        intensities_per_volume.append(mean_intensity)
    return intensities_per_volume

def find_best_volume(intensities_per_volume):
    """Find the volume with the highest intensity."""
    max_int = np.max(intensities_per_volume)
    entry_seuil = 0.6 * max_int
    best_vol = next((i for i, val in enumerate(intensities_per_volume) if val >= entry_seuil), np.argmax(intensities_per_volume))
    return best_vol

def calcul_moy_intensite(data, mask):
    """Calculate mean intensity for each volume using a mask, memory-efficient version"""
    num_volumes = data.shape[-1]
    moy = []
    
    # Convert mask to indices once
    mask_indices = np.where(mask)
    
    for i in range(num_volumes):
        # Extract only the masked values for this volume
        masked_values = data[mask_indices[0], mask_indices[1], mask_indices[2], i]
        m = np.mean(masked_values) if masked_values.size > 0 else 0
        moy.append(m)
    return moy

def get_brain_mask(data, affine):
    """parameters is output of get_data we return a 3D binary mask"""
    mean_data = np.mean(data, axis=-1) if data.ndim == 4 else data
    mask = compute_background_mask(
        nib.Nifti1Image(mean_data, affine),
        connected=False,
        opening=1, 
    ).get_fdata().astype(bool)
    
    mask = binary_closing(mask, structure=np.ones((3, 3, 3)))
    mask = binary_erosion(mask, iterations=1)
    
    labeled_mask, num_labels = label(mask)
    if num_labels > 0:
        largest_component = (labeled_mask == np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1)
        mask = largest_component
    
    return mask

def get_brain_mask_optimized(
    data, 
    affine, 
    threshold=0.5, 
    opening=2, 
    connected=True,
    post_process=True
):
    """define parameters and return bro"""
    mean_data = np.mean(data, axis=-1)
    
    mask = compute_brain_mask(
        nib.Nifti1Image(mean_data, affine),
        threshold=threshold,
        opening=opening,
        connected=connected,
    ).get_fdata().astype(bool)

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
            #z_end = max(z_bottom - offset, 0) 
            z_end = z_bottom

            lower_mask[x, y, z_start:z_end] = True

    return lower_mask

def process_hemisphere(volume, mask, bandwidth=5.0, subsample_size=10000, min_cluster_size=50):
    region = volume * mask
    masked_values = region[region > 0]
    if len(masked_values) == 0:
        return None

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(masked_values.reshape(-1, 1))
    thresh = np.mean(kmeans.cluster_centers_)
    bright_mask = (volume > thresh) & mask
    
    coords = np.where(bright_mask)
    if len(coords[0]) == 0:
        return None
    
    features = np.column_stack((coords[0], coords[1], coords[2], volume[coords]))
    features = features.astype(np.float32)
    
    if features.shape[0] > subsample_size:
        idx = np.random.choice(features.shape[0], size=subsample_size, replace=False)
        coords_subsampled = (coords[0][idx], coords[1][idx], coords[2][idx])
        features = features[idx]
    else:
        coords_subsampled = coords
        
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=1)
    mean_shift.fit(features)
    labels = mean_shift.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)

    if len(unique_labels) == 0:
        return None

    cluster_means = [np.mean(features[labels == label, 3]) for label in unique_labels]
    sorted_indices = np.argsort(cluster_means)[::-1] 
    top_labels = unique_labels[sorted_indices[:2]]

    final_mask = np.zeros_like(mask, dtype=np.uint8)
    for label in top_labels:
        mask_indices = labels == label
        final_mask[coords_subsampled[0][mask_indices],
                  coords_subsampled[1][mask_indices],
                  coords_subsampled[2][mask_indices]] = 1

    if min_cluster_size > 0:
        from skimage.morphology import remove_small_objects
        final_mask = remove_small_objects(final_mask.astype(bool), min_size=min_cluster_size).astype(np.uint8)
    final_mask = binary_dilation(final_mask,iterations=1)  
    struct = morphology.ball(1)
    final_mask = binary_closing(final_mask,struct)
    return final_mask.astype(np.uint8)

def cluster1belongstobrain(voxel_coordinates, brain_mask_coords):
    """
    entree :
        voxel_coordinates : coordonnées des voxels du cluster à vérifier
        brain_mask_coords : coordonnées du cerveau extraites du masque
    sortie :
        bool : True si tous les voxels sont dans la moitié GAUCHE du cerveau, False sinon
    """
    min_x = min(brain_mask_coords[0])
    max_x = max(brain_mask_coords[0])
    print(min_x)
    print(max_x)
    min_y = min(brain_mask_coords[1])
    max_y = max(brain_mask_coords[1])
    min_z = min(brain_mask_coords[2])
    max_z = max(brain_mask_coords[2])
    long = max_x - min_x
    max_gauche = long // 2
    for coord in voxel_coordinates:
        if tuple(coord)[0] < min_x-1 or tuple(coord)[0] > max_gauche-1:
            print("this tuple is outside the left side", coord)
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
    min_x = min(brain_mask_coords[0])
    max_x = max(brain_mask_coords[0])
    print(min_x)
    print(max_x)
    min_y = min(brain_mask_coords[1])
    max_y = max(brain_mask_coords[1])
    min_z = min(brain_mask_coords[2])
    max_z = max(brain_mask_coords[2])
    long = max_x - min_x
    min_droite = long // 2
    for coord in voxel_coordinates:
        if tuple(coord)[0] < min_droite-1 or tuple(coord)[0] > max_x-1:
            print("this tuple is outside the right side", coord)
            return False
    return True 
    """x_min, x_max = brain_mask_coords[:, 0].min(), brain_mask_coords[:, 0].max()
    long = x_max - x_min
    for coord in voxel_coordinates:
        if tuple(coord)[0] < min_droite or tuple(coord)[0] > x_max:
            print("this tuple is outside", coord)
            return False
    return True """




def thresholdmask(nft_img, frame_index):
    #step 1 : finding best threshold: through np.percentile 
    #np.percentile()
    nifti_img_3D = index_img(nft_img, frame_index) 
    nifti_img_3D_2 = index_img(nft_img, frame_index+1) 

    flatdata1= nifti_img_3D_2.get_fdata().flatten()
    flatdata = flatdata1[flatdata1 > 5]
    #need to find something more efficient for threshold
    threshold = format(np.percentile(flatdata, 97)) #ill modify this to see what we got 
    print("the threshold found :", (threshold))
    roi_mask = image.math_img(f"img > {threshold}", img=nifti_img_3D)
    plot_roi(roi_mask, title="Selected Brain Region Mask")
    plt.show()
    return roi_mask

def Parcellationclusters(nifti_img1, roi_mask, n_parcels=30): 
    """
    entrée : image nifti , threshold mask
    sortie : version nifti des clusters 
    """
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
    ward_labels_img = ward.labels_img_ # transformer ward en un object image nifti 
    #montrer dans un plot le cluster(pas trop necessaire)
    first_plot = plotting.plot_roi(ward_labels_img, title="parcellation:", display_mode="xz")
    plotting.show()# Grab cut coordinates from this plot to use as a common for all plots !!!
    return ward_labels_img

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
        cluster_mask = image.math_img(f"img == {clustertrouve}", img=ward_labels_img)
        datatotreat = cluster_mask.get_fdata()
        voxel_coordinates = np.argwhere(datatotreat)
        """plotting.plot_roi(
        cluster_mask,
        bg_img=nifti_img_3D,
        title="Cluster Visualization",
        display_mode="ortho",
        cut_coords=None,
        cmap="autumn",
            )
        plotting.show()"""
        
        if left_voxel_coordinates is None and cluster1belongstobrain(voxel_coordinates, brain_mask_coords):
            left_voxel_coordinates = voxel_coordinates
        elif right_voxel_coordinates is None and cluster2belongstobrain(voxel_coordinates, brain_mask_coords):
            right_voxel_coordinates = voxel_coordinates
        
        if left_voxel_coordinates is not None and right_voxel_coordinates is not None:
            break
        i += 1

    return left_voxel_coordinates, right_voxel_coordinates

def plotcarotid(left_carotid, right_carotid, nft_img,affine,patient_id ): 
    combined_mask = np.zeros_like(nft_img.get_fdata(), dtype=bool)
    
    # Apply the carotid masks to the combined mask
    if left_carotid is not None:
        combined_mask = np.logical_or(combined_mask, left_carotid)
    if right_carotid is not None:
        combined_mask = np.logical_or(combined_mask, right_carotid)
    
    # Compute the mean volume for background image
    mean_volume = np.mean(nft_img.get_fdata(), axis=-1)
    
    # Plot the combined carotid mask
    roi_plot = plotting.plot_roi(
        nib.Nifti1Image(combined_mask.astype(np.uint8), affine),
        bg_img=nib.Nifti1Image(mean_volume, affine),
        title="Combined Carotid Mask",
        cmap="viridis",
        display_mode="ortho"
    )
    
    roi_plot.savefig(f"carotid_mask_{patient_id}.png", dpi=300)
    roi_plot.close()


def extract_carotids(nft_img, lower_brain_mask,affine,patient_id):
    """simple function to get ONLY the mask for carotids, parameters : brain_mean_volume(3d result),then pretty self-explanatory"""
    #well take this part and leave the rest
    intensities = calculate_mean_intensities_per_frame(nft_img,lower_brain_mask)
    best_frame_idx = find_best_volume(intensities)
    print(f"Using frame {best_frame_idx} for carotid segmentation")

    best_volume = image.index_img(nft_img, best_frame_idx)
    best_volume_data = best_volume.get_fdata(dtype=np.float32)

    x_mid = lower_brain_mask.shape[0] // 2
    
    left_mask = lower_brain_mask.copy()
    left_mask[x_mid:, :, :] = False 
    
    right_mask = lower_brain_mask.copy()
    right_mask[:x_mid, :, :] = False  
   
    left_bright = process_hemisphere(best_volume_data, left_mask)
    right_bright = process_hemisphere(best_volume_data, right_mask)
    
    bright_mask = np.zeros_like(lower_brain_mask, dtype=bool)
    if left_bright is not None:
        bright_mask = np.logical_or(bright_mask, left_bright)
    if right_bright is not None:
        bright_mask = np.logical_or(bright_mask, right_bright)
    mean_volume = np.mean(nft_img.get_fdata(), axis=-1)
    
    roi_plot = plotting.plot_roi(
        nib.Nifti1Image(bright_mask.astype(np.uint8), affine),
        bg_img=nib.Nifti1Image(mean_volume, affine),
        title="Combined Carotid Mask",
        cmap="viridis",
        display_mode="ortho"
    )
    roi_plot.savefig(f"carotid_mask_{patient_id}.png", dpi=300)
    roi_plot.close() 
    return left_bright,right_bright,bright_mask


def get_IDIF_FIT(nft_img,volume_data_list):
    img_4D = nft_img.get_fdata()
    t = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,150,180,210,240,270,300,360,420,480,540,600, 900, 1200, 1500, 1800, 2100, 2400])
    print(len(t))
    t_min = t/60
    delta_i = np.diff(t_min) # DurÃ©e de chaque frame en min
	# Extraire la courbe temporelle pour le voxel
    tac = img_4D[18, 52, 3, :]  # Données sur la dimension temporelle


########## FAIRE LE FIT A LA PARTIE DECROISSANTE AVEC LE MODELE DE FENG 
    def safe_exp(x, max_exp=700):
        return np.exp(np.clip(x, -max_exp, max_exp))#added because P11 is problematic

    def gammavariate_model_dec(t, A1, A2, A3, lambda1, lambda2, lambda3):
        Cpmodel = (A1 - A2 - A3) * safe_exp(-lambda1 * t) + A2 * safe_exp(-lambda2 * t) + A3 * safe_exp(-lambda3 * t)
        #Cpmodel = (A1 - A2 - A3) * np.exp(-lambda1 * t) + A2 * np.exp(-lambda2 * t) + A3 * np.exp(-lambda3 * t)
        return Cpmodel
    
    t_data = t_min
    y_data = volume_data_list
    
    
# Trouver l'index de la valeur maximale de yy
    idx_max = np.argmax(y_data)

# Extraire la partie dÃ©croissante des donnÃ©es
    xx_dec = t_data[idx_max:]
    yy_dec = y_data[idx_max:]
#added this line because there was a mismatch 
    if len(xx_dec) != len(yy_dec):
        min_len = min(len(xx_dec), len(yy_dec))
        xx_dec = xx_dec[:min_len]
        yy_dec = yy_dec[:min_len]
# Extraire les valeurs de t qui ont été supprimÃ©es
    xx_removed = t_data[:idx_max]
    yy_removed = [0] * len(xx_removed)

# Ajouter ces valeurs au début de xx_dec et yy_dec
    xx_full = np.concatenate((xx_removed, xx_dec))
    yy_full = np.concatenate((yy_removed, yy_dec))

    params_dec, covariance_dec = curve_fit(gammavariate_model_dec, xx_dec, yy_dec, maxfev=100000000)

    IDIF_fit = np.concatenate((yy_removed, gammavariate_model_dec(xx_dec, *params_dec)))
    
    return t_min,xx_full,IDIF_fit

def plot(t_min,volume_data_list,IDIF_fit,patient):
########## PLOT DES RESULTATS
    plt.figure(figsize=(10, 6))

# TracÃ© des donnÃ©es expÃ©rimentales
    plt.plot(t_min, volume_data_list, 'bo-', label="Données expérimentales", markersize=4)

# TracÃ© du modÃ¨le ajustÃ©
    plt.plot(t_min, IDIF_fit, 'r--', label="Modèle ajusté")

    plt.xlabel("Temps (minutes)")
    plt.ylabel("Concentration (kBq/cc)")
    plt.title("Ajustement de la TAC par le modèle gamma-variée pour le patient "+ patient)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    #plot_filename = os.path.join("./graphes_brainmask", f"{patient}_TAC_fit.png")
    #plt.savefig(plot_filename, dpi=300)
    #plt.close()
    plt.show()

def findingpointsforplot(voxel_coordinates,dataNIFTI ):
    listofpoints = []
    for i in range(len(voxel_coordinates)):
        x,y,z = voxel_coordinates[i]
        listofpoints.append(dataNIFTI[x,y,z,:])
    nplistofpoint= np.array(listofpoints)
    voxel = nplistofpoint.mean(axis=0)
    return voxel

    
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


def mainFunction(patient_id):
    path = "./FICHIERSDONNEES" #a changer 

    nft_img, data, affine = get_data(path, patient_id)
    brain_mask = get_brain_mask_optimized(data, affine)
    brain_mask = brain_mask.astype(np.int8)
    mean_brain = np.mean(data, axis=-1)
    mean_brain_img = nib.Nifti1Image(mean_brain, affine)
    mask_img = nib.Nifti1Image(brain_mask, affine)
    lower_mask = get_lower_brain_mask(brain_mask)
    lower_mask = lower_mask.astype(np.int8)
    lower_mask_img = nib.Nifti1Image(lower_mask, affine)
    #temporarily ill just use x coord but also ill add y (and z?)
    #extraction des coordonnées du lower_mask 
    # Find the non-zero indices (coordinates of the mask)
    #coords = np.where(lower_mask)
    #temporarily using brain_mask instead of lower_mask to fix a problem
    coords= np.where(brain_mask)
    print("coords:",coords)
    
  

    # plotting.plot_roi(
    #     roi_img=lower_mask_img,
    #     bg_img=mean_brain_img,
    #     title="Lower Brain Mask Overlay",
    #     cmap="viridis", 
    #     alpha=0.5,
    #     cut_coords=None, 
    #     display_mode="ortho", 
    # )
    # plt.show()
    intensities = calculate_mean_intensities_per_frame(nft_img,lower_mask)
    print("intensities per frame", intensities)
    best_frame_idx = find_best_volume(intensities)
    print(f"Using frame {best_frame_idx} for carotid segmentation")
    threshold_mask = thresholdmask(nft_img, best_frame_idx-1)
    #Valeurs des deux carotdes
    ward_images = Parcellationclusters(nft_img, threshold_mask)
    nifti_3D = index_img(nft_img, best_frame_idx-1)
    intensites = trouver_cluster_intense(ward_images, nifti_3D)
    left_carotid, right_carotid = trouver_clusters_candidats(intensites, ward_images, nifti_3D, coords )
    left_carot= [0] + findingpointsforplot(left_carotid, data)
    right_carot= [0] + findingpointsforplot(right_carotid, data)
    combined_carotid = np.vstack((left_carotid, right_carotid))
    carot= [0] + findingpointsforplot(combined_carotid, data)

    print("we found lef_carotid to be ", left_carotid)
    print("and right carotid", right_carotid)
    #plotcarotid(left_carotid, right_carotid, nifti_3D, nifti_3D.affine,patient_id )
    """
    left_mask,right_mask, bright_mask = extract_carotids(nft_img, lower_mask,affine,patient_id)
    carot = calcul_moy_intensite(data,bright_mask)
    carot = [0] + list(carot)
    
    #Carotide Gauche
    left_carot = calcul_moy_intensite(data,left_mask)
    left_carot = [0] + list(left_carot)
    print(f"taille left_carot: {len(left_carot)}")
    
    #Carotide Droite
    right_carot = calcul_moy_intensite(data,right_mask)
    right_carot = [0] + list(right_carot)
    print(f"taille right_carot: {len(right_carot)}")"""
    
    #Carotide du prof
    prof_carot = open_crv("crv/crv/"+patient_id + "_carot.crv") #a changer 
    prof_carot = [0] + list(prof_carot)
    
    t_min,x_L, idif_L = get_IDIF_FIT(nft_img,list(left_carot))
    _,x_R,     idif_R = get_IDIF_FIT(nft_img,list(right_carot))
    _,x_C,     idif_C = get_IDIF_FIT(nft_img,carot)
    _,x_P,     idif_P = get_IDIF_FIT(nft_img,prof_carot)
    
    posi = ["prof","carotide","carotide_gauche","carotide_droite"]
    axeX = [x_P,x_C,x_L,x_R]
    data = [prof_carot,carot,left_carot,right_carot]
    idif = [idif_P,idif_C,idif_L,idif_R]
    
    plt.figure(figsize=(16, 10))

    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(axeX[i], data[i], 'bo-', label="Données expérimentales", markersize=4)
        plt.plot(axeX[i], idif[i], 'r--', label="Modèle ajusté")
        plt.xlabel("Temps (minutes)")
        plt.ylabel("Concentration (kBq/cc)")
        plt.title(f"Ajustement TAC - {posi[i]}")
        plt.legend()
        plt.grid()

    plt.suptitle(f"Ajustement des TAC par le modèle gamma-variée pour {patient_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    #plt.show()
    plt.savefig(f"TAC_fit_{patient_id}.png", dpi=300, bbox_inches="tight")  # Formats: .png, .jpg, .pdf
    plt.close() 
    print(f"TAC_fit_{patient_id}.png saved")

def main():
    p = input(f"Entrez l'id du patient (e.g : P04) : ")
    mainFunction(p)

if __name__ == "__main__":
    main()