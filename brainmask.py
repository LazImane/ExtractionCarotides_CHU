import os
import numpy as np
import nibabel as nib
from nilearn import plotting ,image,masking
from nilearn.masking import compute_background_mask,compute_brain_mask
from scipy.ndimage import binary_erosion, binary_closing, label
from skimage import filters, measure
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans

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

def calcul_moy_intensite(data,mask):
    num_volumes = data.shape[-1]
    moy = []
    for i in range(num_volumes) :
        volume = data[:,:,:,i]
        val = volume[mask]
        m = np.mean(val) if val.size>0 else 0
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

            z_start = max(z_bottom - z_depth, 0)
            z_end = z_bottom

            lower_mask[x, y, z_start:z_end] = True

    return lower_mask

def process_hemisphere(volume, mask):
    region = volume * mask
    masked_values = region[region > 0]
    if len(masked_values) == 0:
        return None
    # plt.hist(masked_values.flatten(), bins=100)
    # plt.show()
    # print("intensité maximale",np.max(masked_values))
            
    #K-means
    kmeans = KMeans(n_clusters=2, random_state=0).fit(masked_values.reshape(-1, 1))
    thresh = np.mean(kmeans.cluster_centers_)
        
    bright_mask = (volume > thresh) & mask
    bright_mask = binary_closing(bright_mask, structure=np.ones((3,3,3)))
    bright_mask = binary_erosion(bright_mask, iterations=1)
        
    return bright_mask

def extract_carotids(nft_img, lower_brain_mask,affine):
    """simple function to get ONLY the mask for carotids, parameters : brain_mean_volume(3d result),then pretty self-explanatory"""
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
    
    # plotting.plot_roi(
    #     nib.Nifti1Image(bright_mask.astype(np.uint8), affine),
    #     bg_img=nib.Nifti1Image(mean_volume, affine),
    #     title="Combined Carotid Mask",
    #     cmap="viridis",
    #     display_mode="ortho"
    # )
    # plt.show()
    return left_bright,right_bright
    
def generate_graph(nft_img,volume_data_list, patient):
    img_4D = nft_img.get_fdata()
    t = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,150,180,210,240,270,300,360,420,480,540,600, 900, 1200, 1500, 1800, 2100, 2400])
    print(len(t))
    t_min = t/60
    delta_i = np.diff(t_min) # DurÃ©e de chaque frame en min
	# Extraire la courbe temporelle pour le voxel
    tac = img_4D[18, 52, 3, :]  # Données sur la dimension temporelle


########## FAIRE LE FIT A LA PARTIE DECROISSANTE AVEC LE MODELE DE FENG 
    def gammavariate_model_dec(t, A1, A2, A3, lambda1, lambda2, lambda3):
        Cpmodel = (A1 - A2 - A3) * np.exp(-lambda1 * t) + A2 * np.exp(-lambda2 * t) + A3 * np.exp(-lambda3 * t)
        return Cpmodel
    t_data = t_min
    y_data = volume_data_list
    
    if len(y_data) == 0:
        raise ValueError("No intensity data available for fitting")
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
    #print(f"Graphique enregistré : {plot_filename}")

def visualization(patient_id, path = "./patients"):

    nft_img, data, affine = get_data(path, patient_id)
    brain_mask = get_brain_mask_optimized(data, affine)
    brain_mask = brain_mask.astype(np.int8)
    mean_brain = np.mean(data, axis=-1)
    mean_brain_img = nib.Nifti1Image(mean_brain, affine)
    mask_img = nib.Nifti1Image(brain_mask, affine)
    lower_mask = get_lower_brain_mask(brain_mask)
    lower_mask = lower_mask.astype(np.int8)
    lower_mask_img = nib.Nifti1Image(lower_mask, affine)

    plotting.plot_roi(
        roi_img=mask_img,
        bg_img=mean_brain_img,
        title="Brain Mask Overlay",
        cmap="viridis", 
        alpha=0.5,
        cut_coords=None, 
        display_mode="ortho", 
    )
    plt.show()
    plotting.plot_roi(
        roi_img=lower_mask_img,
        bg_img=mean_brain_img,
        title="Lower Brain Mask Overlay",
        cmap="viridis", 
        alpha=0.5,
        cut_coords=None, 
        display_mode="ortho", 
    )
    plt.show()
 
    left_mask,right_mask = extract_carotids(nft_img, lower_mask,affine)
    plotting.plot_roi(
        roi_img = nib.Nifti1Image(left_mask.astype(np.int8), affine), 
        bg_img=mean_brain_img,
        title="Left Carotid",
        cmap="viridis", 
        alpha=0.5,
        cut_coords=None, 
        display_mode="ortho",
        )
    plt.show()
    plotting.plot_roi(
        roi_img = nib.Nifti1Image(right_mask.astype(np.int8), affine),
        bg_img=mean_brain_img,
        title="Right Carotid",
        cmap="viridis", 
        alpha=0.5,
        cut_coords=None, 
        display_mode="ortho",
        )
    plt.show()
    
def graphe(id = "P04",path= "./patients",n=1) :
    
    nft_img, data, affine = get_data(path, id)
    brain_mask = get_brain_mask_optimized(data, affine)
    brain_mask = brain_mask.astype(np.int8)
    mean_brain = np.mean(data, axis=-1)
    lower_mask = get_lower_brain_mask(brain_mask)
    lower_mask = lower_mask.astype(np.int8)
    left_mask,right_mask = extract_carotids(nft_img, lower_mask,affine)
    if n==1:
        volume_data_list = calcul_moy_intensite(data,left_mask)
        volume_data_list = [0] + list(volume_data_list)
        if len(volume_data_list) == 0:
            raise ValueError("No intensity data extracted")
        print(f"taille volume_data_list: {len(volume_data_list)}")
        generate_graph(nft_img, volume_data_list, id)
    if n==2:
        volume_data_list = calcul_moy_intensite(data,right_mask)
        volume_data_list = [0] + list(volume_data_list)
        if len(volume_data_list) == 0:
            raise ValueError("No intensity data extracted")
        print(f"taille volume_data_list: {len(volume_data_list)}")
        generate_graph(nft_img, volume_data_list, id)
    
def graphes(path,n) :
    patients = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P09', 'P10', 'P11']
    for p in patients :
        nft_img, data, affine = get_data(path, p)
        brain_mask = get_brain_mask_optimized(data, affine)
        brain_mask = brain_mask.astype(np.int8)
        mean_brain = np.mean(data, axis=-1)
        lower_mask = get_lower_brain_mask(brain_mask)
        lower_mask = lower_mask.astype(np.int8)
        left_mask,right_mask = extract_carotids(nft_img, lower_mask,affine)
        if n==1:
            volume_data_list = calcul_moy_intensite(data,left_mask)
            volume_data_list = [0] + list(volume_data_list)
            if len(volume_data_list) == 0:
                raise ValueError("No intensity data extracted")
            print(f"taille volume_data_list: {len(volume_data_list)}")
            generate_graph(nft_img, volume_data_list, p)
        if n==2:
            volume_data_list = calcul_moy_intensite(data,left_mask)
            volume_data_list = [0] + list(volume_data_list)
            if len(volume_data_list) == 0:
                raise ValueError("No intensity data extracted")
            print(f"taille volume_data_list: {len(volume_data_list)}")
            generate_graph(nft_img, volume_data_list, p)
    
    
def main():
    print("Welcome! This program allows you to:")
    print("1 - Visualize carotid artery masks")
    print("2 - Plot intensity graphs over time")
    print()
    print("You must have the following libraries installed:")
    print("nibabel, numpy, nilearn, scipy, scikit-image, scikit-learn, matplotlib\n")

    choice = input("Enter 1 for visualization, 2 for intensity graph: ").strip()

    if choice == "1":
        path = input("Path to patient .nii files: ").strip()
        patient_id = input("Patient ID (e.g., P04): ").strip()
        visualization(patient_id, path)

    elif choice == "2":
        graph_choice = input("Enter 1 for a specific patient, 2 for all patients: ").strip()

        path = input("Path to patient .nii files: ").strip()

        if graph_choice == "1":
            patient_id = input("Patient ID (e.g., P04): ").strip()
            side = input("Press 1 for LEFT carotid, 2 for RIGHT: ").strip()
            if side == "1":
                graphe(patient_id, path, 1)
            else:
                graphe(patient_id, path, 2)

        elif graph_choice == "2":
            side = input("Press 1 for LEFT carotid, 2 for RIGHT: ").strip()
            if side == "1":
                graphes(path, 1)
            else:
                graphes(path, 2)

        else:
            print("Invalid graph option.")

    else:
        print("Invalid main choice.")
        
    

if __name__ == "__main__":
    main()