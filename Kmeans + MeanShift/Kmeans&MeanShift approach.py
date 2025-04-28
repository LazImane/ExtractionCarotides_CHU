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
    entry_seuil = 0.8 * max_int
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

def get_lower_brain_mask(brain_mask,offset=5):
    lower_mask = np.zeros_like(brain_mask, dtype=bool)
    X, Y, Z = brain_mask.shape
    
    for x in range(X):
        for y in range(Y):
            brain_voxels = np.where(brain_mask[x, y, :])[0]
            if brain_voxels.size == 0:
                continue
            
            z_bottom = brain_voxels.min()
            z_depth = brain_voxels.size
            z_start = max(z_bottom - z_depth, 0)
            z_end = max(z_bottom-offset, 0) 
            lower_mask[x, y, z_start:z_end] = True
    
    return lower_mask

def restrict_to_carotids(lower_brain_mask, 
                                  anterior_ratio=0.6, 
                                  posterior_ratio=0.4,
                                  lateral_ratio=0.6):
    """
     Paramètres:
        lower_brain_mask (ndarray): Masque 3D binaire du cerveau (forme X, Y, Z).
        anterior_ratio (float): Fraction antérieure à *conserver*.
                               Ex: 0.6 → conserve les 60% antérieurs du masque.
        posterior_ratio (float): Fraction postérieure à *supprimer*.
                                Ex: 0.4 → supprime les 40% postérieurs du masque.
        lateral_ratio (float): Fraction *centrale* (gauche-droite) à conserver.
                               Ex: 0.6 → conserve 60% centraux de l'axe X.
    
    Retour:
        ndarray: Masque 3D binaire restreint aux carotides.
    """
    X, Y, Z = lower_brain_mask.shape
    y_front_cutoff = int(Y * (1 - anterior_ratio))
    y_back_cutoff = int(Y * posterior_ratio)

    x_center = X // 2
    x_half_width = int(X * lateral_ratio / 2)

    carotid_mask = np.zeros_like(lower_brain_mask, dtype=bool)
    carotid_mask[
        x_center - x_half_width : x_center + x_half_width,
        y_back_cutoff : Y - y_front_cutoff,
        :
    ] = lower_brain_mask[
        x_center - x_half_width : x_center + x_half_width,
        y_back_cutoff : Y - y_front_cutoff,
        :
    ]
    return carotid_mask



def process_hemisphere(volume, mask, affine, bandwidth=5.0, subsample_size=10000, min_cluster_size=50):
    region = volume * mask
    masked_values = region[region > 0]
    if len(masked_values) == 0:
        return None

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(masked_values.reshape(-1, 1))
    thresh = np.max(kmeans.cluster_centers_)
    bright_mask = (volume > thresh) & mask
    
    plotting.plot_roi(
        nib.Nifti1Image(bright_mask.astype(np.int8),affine),
        bg_img=nib.Nifti1Image(volume, affine),
        title="K-means Mask",
        cmap="viridis",
        display_mode="ortho"
    )
    # roi_plot.savefig(f"carotid_mask_{patient_id}.png", dpi=300)
    # roi_plot.close() 
    plotting.show()
    
    # plt.figure(figsize=(6, 4))
    # plt.hist(masked_values, bins=50, color='skyblue', edgecolor='black')
    # plt.axvline(x=thresh, color='red', linestyle='--', label=f'Threshold = {thresh:.2f}')
    # plt.title('KMeans clustering on intensity values')
    # plt.xlabel('Intensity')
    # plt.ylabel('Number of voxels')
    # plt.legend()
    # plt.show()
    
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

def extract_carotids(nft_img, lower_brain_mask,affine,patient_id):
    """simple function to get ONLY the mask for carotids, parameters : brain_mean_volume(3d result),then pretty self-explanatory"""
    intensities = calculate_mean_intensities_per_frame(nft_img,lower_brain_mask)
    best_frame_idx = find_best_volume(intensities)
    print(f"Using frame {best_frame_idx} for carotid segmentation")

    best_volume = image.index_img(nft_img, best_frame_idx)
    best_volume_data = best_volume.get_fdata(dtype=np.float32)
    mean_volume = np.mean(nft_img.get_fdata(), axis=-1)

    x_mid = lower_brain_mask.shape[0] // 2
    
    left_mask = lower_brain_mask.copy()
    left_mask[x_mid:, :, :] = False 
    
    right_mask = lower_brain_mask.copy()
    right_mask[:x_mid, :, :] = False  
   
    left_bright = process_hemisphere(best_volume_data, left_mask ,affine)
    right_bright = process_hemisphere(best_volume_data, right_mask ,affine)
    
    bright_mask = np.zeros_like(lower_brain_mask, dtype=bool)
    if left_bright is not None:
        bright_mask = np.logical_or(bright_mask, left_bright)
    if right_bright is not None:
        bright_mask = np.logical_or(bright_mask, right_bright)
    
    roi_plot = plotting.plot_roi(
        nib.Nifti1Image(bright_mask.astype(np.uint8), affine),
        bg_img=nib.Nifti1Image(mean_volume, affine),
        title="Combined Carotid Mask",
        cmap="viridis",
        display_mode="ortho"
    )
    # roi_plot.savefig(f"carotid_mask_{patient_id}.png", dpi=300)
    # roi_plot.close() 
    plotting.show()
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


def mainFunction(patient_id,path_to_nifti_img,path_to_tac_data):

    nft_img, data, affine = get_data(path_to_nifti_img, patient_id)
    brain_mask = get_brain_mask_optimized(data, affine)
    brain_mask = brain_mask.astype(np.int8)
    mean_brain = np.mean(data, axis=-1)
    mean_brain_img = nib.Nifti1Image(mean_brain, affine)
    mask_img = nib.Nifti1Image(brain_mask, affine)
    lower_mask = get_lower_brain_mask(brain_mask)
    lower_mask = lower_mask.astype(np.int8)
    lower_mask_img = nib.Nifti1Image(lower_mask, affine)
    carotid_region = restrict_to_carotids(lower_mask)
    
    plotting.plot_roi(
        roi_img= nib.Nifti1Image(carotid_region.astype(np.int8), affine),
        bg_img=mean_brain_img,
        title="Carotid Region Mask Overlay",
        cmap="viridis", 
        alpha=0.5,
        cut_coords=None, 
        display_mode="ortho", 
    )
    plt.show()
 
    #Valeurs des deux carotdes
    left_mask,right_mask, bright_mask = extract_carotids(nft_img, carotid_region,affine,patient_id)
    carot = calcul_moy_intensite(data,bright_mask)
    carot = [0] + list(carot)
    
    #Carotide Gauche
    left_carot = calcul_moy_intensite(data,left_mask)
    left_carot = [0] + list(left_carot)
    print(f"taille left_carot: {len(left_carot)}")
    
    #Carotide Droite
    right_carot = calcul_moy_intensite(data,right_mask)
    right_carot = [0] + list(right_carot)
    print(f"taille right_carot: {len(right_carot)}")
    
    #Carotide du prof
    prof_carot = open_crv(path_to_tac_data +"/"+patient_id + "_carot.crv")
    prof_carot = [0] + list(prof_carot)
    
    t_min,x_L, idif_L = get_IDIF_FIT(nft_img,left_carot)
    _,x_R,     idif_R = get_IDIF_FIT(nft_img,right_carot)
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
    plt.show()
    # plt.savefig(f"TAC_fit_{patient_id}.png", dpi=300, bbox_inches="tight")  # Formats: .png, .jpg, .pdf
    # plt.close() 
    # print(f"TAC_fit_{patient_id}.png saved")

def main():
    path_to_nifti_img = "./patients"
    path_to_tac_data = "./prof"
    print("Bienvenue ! Ce programme vous permet de :")
    print("1 - Visualiser les masques des artères carotides")
    print("2 - Tracer les courbes d'intensité au cours du temps")
    print()
    print("Vous devez avoir les bibliothèques suivantes installées :")
    print("nibabel, numpy, nilearn, scipy, scikit-image, scikit-learn, matplotlib\n")
    print()
    print(f"Chemin vers les données .nii actuel : {path_to_nifti_img}")
    new_path_nifti = input("Appuyez sur Entrée pour garder ce chemin, ou saisissez un nouveau chemin : ")
    if new_path_nifti.strip():
        path_to_nifti_img = new_path_nifti 
    print(f"Chemin vers les données tac .crv actuel : {path_to_tac_data}")
    new_path_results = input("Appuyez sur Entrée pour garder ce chemin, ou saisissez un nouveau chemin : ")
    if new_path_results.strip():
        path_to_tac_data = new_path_results
        
    p = input(f"Entrez l'id du patient (e.g : P04) : ")
    mainFunction(p,path_to_nifti_img,path_to_tac_data)

if __name__ == "__main__":
    main()