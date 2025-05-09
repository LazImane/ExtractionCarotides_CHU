import os
import gdown

# Dictionnaires contenant les noms de fichiers et leurs IDs Google Drive
files_to_download = {
    "nii": {
        "p01.nii": "1Y9H5IKT5wc6tftkvcEPBApTSd8UO0Oe7",
        "p02.nii": "1pOv0lmz-vAzbql2ganvmcftP73ze7A2B",
        "p03.nii": "10tTK88Wj7oz3ayFrvHXd11eHI-vDysRm",
        "p04.nii": "1HWicUJ8xRgbSSVRldm8XrQkcnKHDosxd",
        "p05.nii": "1PFV1P9uUMSU3ZP-pCV5tL4LGqZ3xYwNV",
        "p06.nii": "1y9Wt3sUVvosMmvfvtCIDc29hiNrI9XWH",
        "p07.nii": "1GMCSTahLZNNrfUXqdp2zfhmQc7Tr8m8a",
        "p09.nii": "19Tw1hUOBtN0KAX17m01R8OSMkVQQ62Rk",
        "p10.nii": "1AtaNIeOeNVNe9pnhwJgaWeTzXIUklqRz"
    },
    "crv": {
        "P01_carot.crv": "1qWXiQbAMT4-vjQNskm-NsHsUnzY4laO0",
        "P02_carot.crv": "19CxhbqxXndvyIA74mQSOSGxtZEK0zHPN",
        "P03_carot.crv": "1-9AydsgB3-vLy_f5x2WdvJAxXRtkM5A0",
        "P04_carot.crv": "1QAgQ9P29OB26rDMWF_x7E5lEr9NdMX-P",
        "P05_carot.crv": "1IYIR4LF7lmTP64InL7oKLC5q_LJeBOhb",
        "P06_carot.crv": "1Xpwb-RPxRhMUNuhWJJgUUkYMfXeTgYK9",
        "P07_carot.crv": "1iHQFa6Pt_fLGcvXLcH0UasfRVB7QHRss",
        "P09_carot.crv": "195dmkJMnrQjGkMBRa1ciBi3lpTAP8j1V",
        "P10_carot.crv": "1-EIGsVTxkwwWonalhjGNL-sHc4NooX-f"
    }
}

# Télécharger les fichiers pour chaque type (nii ou crv)
for filetype, file_dict in files_to_download.items():
    output_dir = os.path.join("Data", filetype)
    os.makedirs(output_dir, exist_ok=True)

    for filename, file_id in file_dict.items():
        url = f"https://drive.google.com/uc?id={file_id}"
        output_path = os.path.join(output_dir, filename)

        if not os.path.exists(output_path):
            print(f"Téléchargement de {filename} dans {output_dir}...")
            gdown.download(url, output_path, quiet=False)
        else:
            print(f"{filename} déjà présent dans {output_dir}, téléchargement ignoré.")
