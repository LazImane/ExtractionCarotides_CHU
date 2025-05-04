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
    """path = where the patients are, patien_id = like "p04" """
    filename = os.path.join(path, patient_id + ".nii")
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Le fichier NIFTI {filename} est introuvable.")
    
    try:
        nifti = nib.load(filename)
        data = nifti.get_fdata()
        affine = nifti.affine
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement de l'image NIFTI {filename} : {e}")
    
    if data.ndim != 4:
        raise ValueError(f"L'image {filename} n'est pas une image 4D (trouvée {data.ndim}D).")

    if data.shape[-1] < 10:
        raise ValueError(f"L'image {filename} contient un nombre insuffisant de frames temporels : {data.shape[-1]}")

    return nifti, data, affine


def get_axis_mid(coords, axis=0):
    return (min(coords[axis]) + max(coords[axis])) // 2


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
    entry_seuil = 0.7 * max_int
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


def get_brain_mask(
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



def is_left_cluster(voxel_coordinates, brain_mask_coords):
    """
    Vérifie si tous les voxels sont dans la moitié gauche du cerveau
    """
    x_mid = get_axis_mid(brain_mask_coords, axis=0)

    for coord in voxel_coordinates:
        if coord[0] > x_mid:
            print("this tuple is outside the left side", coord)
            return False
    return True 

def is_right_cluster(voxel_coordinates, brain_mask_coords):
    """
    Vérifie si tous les voxels sont dans la moitié droite du cerveau
    """
    x_mid = get_axis_mid(brain_mask_coords, axis=0)
    for coord in voxel_coordinates:
        if coord[0] < x_mid:
            print("this tuple is outside the right side", coord)
            return False
    return True 


def thresholdmask(nft_img, frame_index):
    #step 1 : finding best threshold: through np.percentile 
    #np.percentile()
    nifti_img_3D = index_img(nft_img, frame_index) 
    nifti_img_3D_2 = index_img(nft_img, frame_index) 

    flatdata1= nifti_img_3D_2.get_fdata().flatten()
    flatdata = flatdata1[flatdata1 > 5]
    #need to find something more efficient for threshold
    threshold = format(np.percentile(flatdata, 96.9)) #ill modify this to see what we got 
    print("the threshold found :", (threshold))
    roi_mask = image.math_img(f"img > {threshold}", img=nifti_img_3D)
    plot_roi(roi_mask, title="Selected Brain Region Mask")
    plt.show()
    return roi_mask

def parcellate_image_with_ward(nifti_img1, roi_mask, n_parcels=30): 
    """
    entrée : image nifti , threshold mask
    sortie : version nifti des clusters 
    """
    roi_data = roi_mask.get_fdata()
    if np.count_nonzero(roi_data) == 0:
        raise ValueError("Le masque fourni à la parcellation est vide. Vérifiez le seuillage ou la frame sélectionnée.")
    
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
    first_plot = plotting.plot_roi(ward_labels_img, title="parcellation:", display_mode="xz") # a supprimer
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

def trouver_clusters_candidats(intensite_clusters, ward_labels_img,brain_mask_coords):
    """
    entree :
        intensite_clusters (list of tuple) : clusters triés par intensité
        ward_labels_img (Nifti1Image) : image contenant les labels de parcellation
        brain_mask_coords (np.ndarray) : coordonnées des voxels du cerveau
    sortie :
        tuple (np.ndarray, np.ndarray) : coordonnées des voxels des deux clusters candidats (gauche et droite)
    """
    i = 0
    left_voxel_coordinates = None
    right_voxel_coordinates = None

    while True:
        if i >= len(intensite_clusters):
            raise RuntimeError("Aucun cluster gauche/droite n'a pu être identifié.")

        clustertrouve = intensite_clusters[i][0]
        print(f"Cluster with the highest intensity: {clustertrouve}, with an intensity of : {intensite_clusters[i][1]}")
        cluster_mask = image.math_img(f"img == {clustertrouve}", img=ward_labels_img)
        datatotreat = cluster_mask.get_fdata()
        voxel_coordinates = np.argwhere(datatotreat)

        if left_voxel_coordinates is None and is_left_cluster(voxel_coordinates, brain_mask_coords):
            left_voxel_coordinates = voxel_coordinates
        elif right_voxel_coordinates is None and is_right_cluster(voxel_coordinates, brain_mask_coords):
            right_voxel_coordinates = voxel_coordinates

        if left_voxel_coordinates is not None and right_voxel_coordinates is not None:
            break

        i += 1

    if left_voxel_coordinates is None:
        raise RuntimeError("Impossible d'identifier un cluster gauche appartenant au cerveau.")

    if right_voxel_coordinates is None:
        raise RuntimeError("Impossible d'identifier un cluster droit appartenant au cerveau.")


    return left_voxel_coordinates, right_voxel_coordinates


def plot_combined_carotid_mask(left_mask, right_mask, full_img, affine, output_path):
    combined_mask = np.zeros_like(full_img.get_fdata(), dtype=bool)

    if left_mask is not None:
        combined_mask = np.logical_or(combined_mask, left_mask)
    if right_mask is not None:
        combined_mask = np.logical_or(combined_mask, right_mask)

    mean_volume = np.mean(full_img.get_fdata(), axis=-1)

    roi_plot = plotting.plot_roi(
        nib.Nifti1Image(combined_mask.astype(np.uint8), affine),
        bg_img=nib.Nifti1Image(mean_volume, affine),
        title="Combined Carotid Mask",
        cmap="viridis",
        display_mode="ortho"
    )
    roi_plot.savefig(output_path, dpi=300)
    roi_plot.close()






def get_IDIF_FIT(nft_img, volume_data_list):
    # Auto-adapte la longueur de t au nombre de points
    default_t = np.array([
        0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
        150, 180, 210, 240, 270, 300, 360, 420, 480, 540, 600,
        900, 1200, 1500, 1800, 2100, 2400
    ])
    t = default_t[:len(volume_data_list)]  # tronque ou adapte automatiquement

    t_min = t / 60  # secondes → minutes

    if len(volume_data_list) != len(t_min):
        raise ValueError(
            f"Les longueurs des données TAC ({len(volume_data_list)}) et du temps ({len(t_min)}) ne correspondent pas."
        )

    def safe_exp(x, max_exp=700):
        return np.exp(np.clip(x, -max_exp, max_exp))

    def gammavariate_model_dec(t, A1, A2, A3, lambda1, lambda2, lambda3):
        return (
            (A1 - A2 - A3) * safe_exp(-lambda1 * t)
            + A2 * safe_exp(-lambda2 * t)
            + A3 * safe_exp(-lambda3 * t)
        )

    idx_max = np.argmax(volume_data_list)
    xx_dec = t_min[idx_max:]
    yy_dec = volume_data_list[idx_max:]

    if len(xx_dec) < 4:
        raise RuntimeError("Pas assez de points dans la phase décroissante pour effectuer un fit fiable.")

    xx_removed = t_min[:idx_max]
    yy_removed = [0] * len(xx_removed)

    xx_full = np.concatenate((xx_removed, xx_dec))
    yy_full = np.concatenate((yy_removed, yy_dec))

    try:
        params_dec, _ = curve_fit(gammavariate_model_dec, xx_dec, yy_dec, maxfev=100000)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du fit de la courbe IDIF : {e}")

    IDIF_fit = np.concatenate((yy_removed, gammavariate_model_dec(xx_dec, *params_dec)))

    return t_min, xx_full, IDIF_fit


#à supprimer
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


###-------------------------------- PLOT DES RESULTATS -------------------------------###



def average_tac_from_voxels(voxel_coordinates, dataNIFTI):
    if voxel_coordinates is None or len(voxel_coordinates) == 0:
        raise ValueError("La liste des coordonnées de voxels est vide ou None.")
    
    listofpoints = []
    for x, y, z in voxel_coordinates:
        listofpoints.append(dataNIFTI[x, y, z, :])
    return np.mean(np.array(listofpoints), axis=0)

    
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




def plot_results(patient_id, t_min, x_L, x_R, x_C, x_P, left_carot, right_carot, carot, ref_carot, idif_L, idif_R, idif_C, idif_P, nft_img, ward_labels_img, left_carotid, right_carotid, affine,sauvegarde_res):
    """
    Fonction pour afficher les résultats
    + affichage de la TAC de référence quand on appuie sur 'a'

    image des clusters     |     TAC carotide en moyenne
    __________________________________________________
    TAC carotide gauche   |      TAC carotide droite
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import numpy as np
    import io
    from PIL import Image
    import os
    import nibabel as nib
    from nilearn import plotting

    # Créer la figure avec 4 panneaux
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    #-------------------partie affichage image-------------------------------#

    # Créer une visualisation des clusters pour l'affichage
    mean_volume = np.mean(nft_img.get_fdata(), axis=-1)
    mean_volume_img = nib.Nifti1Image(mean_volume, affine)
    
    # Créer un masque pour les carotides gauche et droite
    carotid_mask = np.zeros_like(mean_volume, dtype=np.int8)
    
    # Remplir le masque avec 1 pour la carotide gauche, 2 pour la droite
    if left_carotid is not None:
        for x, y, z in left_carotid:
            carotid_mask[x, y, z] = 1
    
    if right_carotid is not None:
        for x, y, z in right_carotid:
            carotid_mask[x, y, z] = 2
    
    carotid_mask_img = nib.Nifti1Image(carotid_mask, affine)
    
    # Créer une figure temporaire pour la capture
    cluster_fig = plt.figure(figsize=(8, 8))
    
    # Utiliser nilearn pour afficher les clusters sur l'image moyenne
    display = plotting.plot_roi(
        carotid_mask_img,
        bg_img=mean_volume_img,
        display_mode="ortho",
        title="Clusters des carotides (gauche: rouge, droite: bleu)",
        cmap="cool",
        figure=cluster_fig
    )
    
    # Enregistrer la figure dans un buffer
    buf = io.BytesIO()
    cluster_fig.savefig(buf, format='png', dpi=100)
    plt.close(cluster_fig)
    
    # Convertir le buffer en image
    buf.seek(0)
    cluster_image = Image.open(buf)

    # Afficher l'image dans le panneau supérieur gauche
    axs[0, 0].imshow(np.array(cluster_image))
    axs[0, 0].axis('off')  # Désactiver les axes pour l'image
    axs[0, 0].set_title("Clusters des carotides identifiés")

    #-------------------------------------------------------------------------#

    # Tracé des tac dans les autres panneaux
    titles = ["carotide", "carotide_gauche", "carotide_droite"]
    data_list = [carot, left_carot, right_carot]
    idif_list = [idif_C, idif_L, idif_R]
    axeX_list = [x_C, x_L, x_R]

    ref_lines = [] # Pour stocker les références

    for i, (ax, title, data, idif, axeX) in enumerate(zip(axs.flatten()[1:], titles, data_list, idif_list, axeX_list)):
        ax.plot(axeX, data, 'bo-', label="Données expérimentales", markersize=3)
        ax.plot(axeX, idif, 'r--', label="Modèle ajusté")

        # Ajout courbes de référence pour comparer (initialement invisibles)
        line1 = ax.plot(x_P, ref_carot , 'go-', label="Référence (TAC M.Kucharczak)", markersize=3, visible=False)[0]
        line2 = ax.plot(x_P, idif_P, 'g--', label="Modèle ajusté référence", alpha=0.7, visible=False)[0]
        ref_lines.extend([line1, line2])

        ax.set_xlabel("Temps (minutes)")
        ax.set_ylabel("Concentration (kBq/cc)")
        ax.set_title(f"Ajustement TAC - {title}")
        ax.legend()
        ax.grid()

    def toggle_ref_visibility(event):
        """Gère l'événement clavier"""
        if event.key == 'a':
            # Visibilité ou non des courbes de référence
            for line in ref_lines:
                line.set_visible(not line.get_visible())
            # Maj légendes
            for ax in axs.flatten()[1:]:
                ax.legend()
            plt.draw()  # Rafraîchir l'affichage

    # Associer la fonction à l'événement clavier 'a'
    fig.canvas.mpl_connect('key_press_event', toggle_ref_visibility)

    # Ajout texte d'instruction en bas de la figure
    plt.figtext(0.5, 0.01, "Appuyez sur 'a' pour afficher/masquer la courbe de référence",
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    plt.suptitle(f"Ajustement des TAC par le modèle gamma-variée pour {patient_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{sauvegarde_res}_{patient_id}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"TAC_{patient_id}.png saved in Resultats_de_la_fonction")



def mainFunction(patient_id,path,path_tac_ref,sauvegarde_res):
    """
    Fonction principale qui réalise la segmentation des carotides et calcule les courbes d'activité temporelle.
    Cette fonction traite les images TEP cérébrales pour identifier les carotides et calculer les TAC
    
    Args:
        patient_id (str): Identifiant du patient (ex: "p04")
    """
   
    # Chargement des données d'imagerie du patient
    nft_img, data, affine = get_data(path, patient_id)
    
    # ----- ÉTAPE 1: Segmentation du cerveau -----
    # Création d'un masque du cerveau pour éliminer les tissus non cérébraux
    brain_mask = get_brain_mask(data, affine)
    brain_mask = brain_mask.astype(np.int8)
    
    # Calcul de l'image moyenne sur toutes les frames temporelles
    mean_brain = np.mean(data, axis=-1)
    mean_brain_img = nib.Nifti1Image(mean_brain, affine)
    mask_img = nib.Nifti1Image(brain_mask, affine)
    
    # ----- ÉTAPE 2: Identification de la partie inférieure du cerveau -----
    # Génération d'un masque pour la partie inférieure du cerveau où se trouvent les carotides
    lower_mask = get_lower_brain_mask(brain_mask)
    lower_mask = lower_mask.astype(np.int8)
    lower_mask_img = nib.Nifti1Image(lower_mask, affine)
    
    # Extraction des coordonnées du cerveau pour analyses futures
    coords = np.where(brain_mask)
    print("coords:", coords)

    # ----- ÉTAPE 3: Sélection de la meilleure frame temporelle -----
    # Calcul des intensités moyennes pour chaque frame temporelle
    intensities = calculate_mean_intensities_per_frame(nft_img, lower_mask)
    print("intensities per frame", intensities)
    
    # Sélection de la frame avec la meilleure concentration de traceur
    best_frame_idx = find_best_volume(intensities)
    print(f"Using frame {best_frame_idx} for carotid segmentation")
    
    # ----- ÉTAPE 4: Génération du masque par seuillage -----
    # Création d'un masque basé sur un seuil d'intensité pour la meilleure frame
    threshold_mask = thresholdmask(nft_img, best_frame_idx - 1)
    
    # ----- ÉTAPE 5: Parcellation et clustering -----
    # Segmentation de l'image en régions (parcelles)
    parcellation_labels_img = parcellate_image_with_ward(nft_img, threshold_mask)
    # Extraction de la frame 3D pour l'analyse des clusters
    nifti_3D = index_img(nft_img, best_frame_idx - 1)
    
    # ----- ÉTAPE 6: Identification des carotides -----
    # Calcul des intensités moyennes de chaque cluster
    intensites = trouver_cluster_intense(parcellation_labels_img, nifti_3D)
    
    # Identification des clusters correspondant aux carotides gauche et droite
    left_carotid, right_carotid = trouver_clusters_candidats(intensites, parcellation_labels_img, coords)
    
    # ----- ÉTAPE 7: Extraction des courbes d'activité temporelle (TAC) -----
    # Calcul des TAC pour la carotide gauche, droite et combinée
    raw_tac_left = average_tac_from_voxels(left_carotid, data)
    if len(raw_tac_left) != 29:
        raise ValueError(f"TAC gauche attendue de longueur 29, obtenue : {len(raw_tac_left)}")
    left_carot = [0] + list(raw_tac_left)
    # Ajout d'un 0 initial pour la courbe
    right_carot = [0] + average_tac_from_voxels(right_carotid, data)
    combined_carotid = np.vstack((left_carotid, right_carotid))
    carot = [0] + average_tac_from_voxels(combined_carotid, data)

    print("we found lef_carotid to be ", left_carotid)
    print("and right carotid", right_carotid)

    # ----- ÉTAPE 8: Chargement des données de référence -----
    # Chargement de la courbe de référence fournie par l'encadrant pour comparaison
    crv_filename = os.path.join(path_tac_ref, f"{patient_id.upper()}_carot.crv")  # ex: P04_carot.crv
    print(f"Fichier .crv utilisé : {crv_filename}")

    ref_carot = open_crv(crv_filename)
    ref_carot = [0] + list(ref_carot)


    # ----- ÉTAPE 9: Modélisation des courbes (IDIF fitting) -----
    # Ajustement des courbes d'activité temporelle avec un modèle exponentiel
    # pour la carotide gauche, droite, combinée et la référence
    t_min, x_L, idif_L = get_IDIF_FIT(nft_img, list(left_carot))
    _, x_R, idif_R = get_IDIF_FIT(nft_img, list(right_carot))
    _, x_C, idif_C = get_IDIF_FIT(nft_img, carot)
    _, x_P, idif_P = get_IDIF_FIT(nft_img, ref_carot)

    # ----- ÉTAPE 10: Visualisation des résultats -----
    # Affichage des résultats avec l'image des clusters et les courbes TAC
    plot_results(
        patient_id,          # ID du patient pour le titre
        t_min,               # Temps en minutes pour l'axe X
        x_L, x_R, x_C, x_P,  # Axes X pour les différentes courbes
        left_carot, right_carot, carot, ref_carot,  # Données des courbes
        idif_L, idif_R, idif_C, idif_P,              # Courbes ajustées
        nft_img, parcellation_labels_img, left_carotid, right_carotid, affine , # Données pour générer l'image des clusters
        sauvegarde_res       #Chemin pour sauvegarder le résultat
    )

def main():
    # Chemin vers le dossier contenant les fichiers patient nifti
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path_to_nifti_img = os.path.join(BASE_DIR, "data/nii")
    path_tac_ref = "./data/nii"
    sauvegarde_res = './Resultats_de_la_fonction'
    #path_tac_ref = f"./PROJETS6/TacReferences" #A MODIFIER si on veut d'autres references que celles dans le dossier prof
    print("Bienvenue ! Ce programme vous permet de :")
    print("1 - Visualiser les masques des artères carotides")
    print("2 - Tracer les courbes d'intensité au cours du temps")
    print()
    print("Vous devez avoir les bibliothèques suivantes installées :")
    print("nibabel, numpy, nilearn, scipy, scikit-image, scikit-learn, matplotlib\n")
    print("Veuillez lancer le programme download_data.py en avance pour avoir les données .nii et .crv qui ne sont pas supportées par GIT dans votre répértoire")
    print()
    
    # Supprime les input() pour fixer les paths  :
    path_to_nifti_img = os.path.join(BASE_DIR, "data/nii")
    path_tac_ref = os.path.join(BASE_DIR, "data/crv")
    sauvegarde_res = os.path.join(BASE_DIR, "Resultats_de_la_fonction")

        
    p = input(f"Entrez l'id du patient (e.g :p04) : ")
    mainFunction(p,path_to_nifti_img,path_tac_ref,sauvegarde_res)

if __name__ == "__main__":
    main()