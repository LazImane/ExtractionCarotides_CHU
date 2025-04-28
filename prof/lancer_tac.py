import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Produit les TAC des fichiers CRV fournis a titre de comparaison

input_path = "./PROJETS6/TAC_donnees/"
crv_files = [f for f in os.listdir(input_path) if f.endswith(".crv")]
print("Fichiers trouvés :", crv_files)

for filename in crv_files:
    patient_name = filename.split("_")[0]
    file_path = os.path.join(input_path, filename)

    try:
        df = pd.read_csv(file_path, sep="\t", skiprows=0)

        CPsim = df["value[kBq/cc]"].to_numpy()
        CPsim = np.insert(CPsim, 0, 0)

        print(f"Patient : {patient_name}")
        print(f"TAC (n={len(CPsim)} points) : {CPsim}\n")

        
        #plt.plot(CPsim, label=patient_name) #si on veut afficher toutes les courbes sur un seul fichier 
        plt.figure()
        plt.plot(CPsim, label=f"TAC {patient_name}")
        plt.xlabel("Temps (s)")
        plt.ylabel("value(kBq/cc)")
        plt.title(f"TAC - {patient_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        #plt.show() #affichier sans enregistrer
        plt.savefig(f"./PROJETS6/prof/TAC_{patient_name}.png")
        plt.close()


    except Exception as e:
        print(f"Erreur avec le fichier {filename} : {e}")

#si on veut afficher toutes les courbes sur un seul fichier 
#plt.xlabel("Temps")
#plt.ylabel("TAC (kBq/cc)")
#plt.title("Courbes TAC par patient")
#plt.legend()
#plt.grid(True)
#plt.show()
