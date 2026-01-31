import os
import pandas as pd

# ======================================================
# FITXER GLOBAL TFM
# ======================================================

GLOBAL_EXCEL = r"C:\Users\sanrr\PycharmProjects\TFM\RESULTATS_GLOBALS\resultats_globals_TFM.xlsx"
os.makedirs(os.path.dirname(GLOBAL_EXCEL), exist_ok=True)

# ======================================================
# RUTES DE TOTES LES MOSTRES (2D + 3D)
# ======================================================

BASE = r"C:\Users\sanrr\PycharmProjects\TFM"

PATHS = {
    "03B": {
        "FD2D": os.path.join(BASE, "RESULTATS_2D_03B", "FD2D_resultats_03B.xlsx"),
        "FD3D": os.path.join(BASE, "RESULTATS_3D_03B", "FD3D_resultats.xlsx")
    },
    "04A": {
        "FD2D": os.path.join(BASE, "RESULTATS_2D_04A", "FD2D_resultats_04A.xlsx"),
        "FD3D": os.path.join(BASE, "RESULTATS_3D_04A", "FD3D_resultats.xlsx")
    },
    "03D": {
        "FD2D": os.path.join(BASE, "RESULTATS_2D_03D", "FD2D_resultats_03D.xlsx"),
        "FD3D": os.path.join(BASE, "RESULTATS_3D_03D", "FD3D_resultats.xlsx")
    },
    "06A": {
        "FD2D": os.path.join(BASE, "RESULTATS_2D_06A", "FD2D_resultats_06A.xlsx"),
        "FD3D": os.path.join(BASE, "RESULTATS_3D_06A", "FD3D_resultats.xlsx")
    },
    "06B": {
        "FD2D": os.path.join(BASE, "RESULTATS_2D_06B", "FD2D_resultats_06B.xlsx"),
        "FD3D": os.path.join(BASE, "RESULTATS_3D_06B", "FD3D_resultats.xlsx")
    },
    "07B": {
        "FD2D": os.path.join(BASE, "RESULTATS_2D_07B", "FD2D_resultats_07B.xlsx"),
        "FD3D": os.path.join(BASE, "RESULTATS_3D_07B", "FD3D_resultats.xlsx")
    }
}

# ======================================================
# FUNCIONS DE LECTURA
# ======================================================

def llegir_fd2d(path):
    df = pd.read_excel(path)
    return float(df["FD2D"].iloc[0]), float(df["slope"].iloc[0]), float(df["R2"].iloc[0])

def llegir_fd3d(path):
    df = pd.read_excel(path)
    return float(df["FD3D"].iloc[0]), float(df["slope"].iloc[0]), float(df["R2"].iloc[0])

# ======================================================
# AFEGIR / ACTUALITZAR MOSTRA
# ======================================================

def afegir_mostra(mostra, E_exp=None, sigma_exp=None, BMD=None, estudi="Marc"):

    if mostra not in PATHS:
        raise ValueError(f"❌ Mostra {mostra} no definida.")

    path2d = PATHS[mostra]["FD2D"]
    path3d = PATHS[mostra]["FD3D"]

    if not os.path.exists(path2d):
        raise FileNotFoundError(f"No trobat FD2D: {path2d}")
    if not os.path.exists(path3d):
        raise FileNotFoundError(f"No trobat FD3D: {path3d}")

    fd2d, slope2d, r22d = llegir_fd2d(path2d)
    fd3d, slope3d, r23d = llegir_fd3d(path3d)

    nova_fila = {
        "Mostra": mostra,
        "FD2D": fd2d,
        "slope2D": slope2d,
        "R2_2D": r22d,
        "FD3D": fd3d,
        "slope3D": slope3d,
        "R2_3D": r23d,
        "E_experimental": E_exp,
        "Sigma_experimental": sigma_exp,
        "BMD": BMD,
        "Estudi": estudi
    }

    if os.path.exists(GLOBAL_EXCEL):
        df = pd.read_excel(GLOBAL_EXCEL)
        df = df[df["Mostra"] != mostra]  # evita duplicats
        df = pd.concat([df, pd.DataFrame([nova_fila])], ignore_index=True)
    else:
        df = pd.DataFrame([nova_fila])

    df.to_excel(GLOBAL_EXCEL, index=False)

    print("✔️ Mostra afegida / actualitzada correctament:")
    print("   →", mostra)
    print("📂", GLOBAL_EXCEL)


# ======================================================
# EXECUCIÓ INTERACTIVA
# ======================================================

if __name__ == "__main__":

    print("======================================")
    print("  AFEGIR RESULTATS GLOBALS TFM")
    print("======================================")

    for i, m in enumerate(PATHS.keys(), start=1):
        print(f"{i} = {m}")

    op = input("Opció: ").strip()
    keys = list(PATHS.keys())

    if not op.isdigit() or int(op) < 1 or int(op) > len(keys):
        raise ValueError("Opció no vàlida.")

    mostra = keys[int(op) - 1]

    E_val = input("Mòdul elàstic E (GPa) [enter si buit]: ").strip()
    E_val = float(E_val) if E_val != "" else None

    sigma_val = input("Tensió mitjana σ (MPa) [enter si buit]: ").strip()
    sigma_val = float(sigma_val) if sigma_val != "" else None

    bmd_val = input("BMD [enter si buit]: ").strip()
    bmd_val = float(bmd_val) if bmd_val != "" else None

    afegir_mostra(
        mostra,
        E_exp=E_val,
        sigma_exp=sigma_val,
        BMD=bmd_val
    )

