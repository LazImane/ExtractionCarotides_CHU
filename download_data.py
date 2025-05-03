import os
import gdown

# Créer un dossier local pour stocker les fichiers téléchargés
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Dictionnaire contenant les noms de fichiers et leurs IDs Google Drive
files_to_download = {
    "p01.nii": "1Y9H5IKT5wc6tftkvcEPBApTSd8UO0Oe7",
    "p02.nii": "1pOv0lmz-vAzbql2ganvmcftP73ze7A2B",
    "p03.nii": "10tTK88Wj7oz3ayFrvHXd11eHI-vDysRm",
    "p04.nii": "1HWicUJ8xRgbSSVRldm8XrQkcnKHDosxd",
    "p05.nii": "1PFV1P9uUMSU3ZP-pCV5tL4LGqZ3xYwNV",
    "p06.nii": "1y9Wt3sUVvosMmvfvtCIDc29hiNrI9XWH",
    "p07.nii": "1GMCSTahLZNNrfUXqdp2zfhmQc7Tr8m8a",
    "p09.nii": "19Tw1hUOBtN0KAX17m01R8OSMkVQQ62Rk",
    "p10.nii": "1AtaNIeOeNVNe9pnhwJgaWeTzXIUklqRz",
    "p11.nii": "1-1AWlOsSmMJSmwYQAb_6vAJCkk3f8SL7"
}

# Télécharger les fichiers un par un
for filename, file_id in files_to_download.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(output_path):
        print(f"Téléchargement de {filename}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{filename} déjà présent, téléchargement ignoré.")
