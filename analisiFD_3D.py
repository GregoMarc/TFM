# ==============================================================
# ANALISI FRACTAL 3D - Versió ampliada del codi d'Isabel Castelló
# Autor: Marc (TFM 2025)
# Descripció:
#   - Llegeix totes les imatges BMP d'una mostra (3B o 4A)
#   - Construeix el volum 3D
#   - Permet seleccionar una ROI manualment sobre el tall central
#   - Calcula la dimensió fractal 3D (FD3D) mitjançant FFT3D
#   - Desa resultats i gràfics en una carpeta RESULTATS_3D
# ==============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from scipy.fftpack import fftn, fftshift
import pandas as pd

# ----------------------------
# PARÀMETRES GLOBALS
# ----------------------------
pixel_size_mm = 0.038        # resolució espacial (mm/píxel)
roi_size = 50               # mida del cub ROI (costat en píxels)
start_manual = 5             # punts inicials a descartar
end_manual = 20              # punts finals a descartar

# ----------------------------
# SELECCIÓ DE MOSTRA
# ----------------------------
print("Escull la mostra a analitzar:")
print("  3 -> 03B (03B_LR_center)")
print("  4 -> 04A (04A_mCT_center)")
print("  D -> 03D (03D_LR_center)")
print("  6 -> 06A (06A_LR_center)")
print("  B -> 06B (06B_LR_center)")
print("  7 -> 07B (07B_LR_center)")
print("  M -> Fractal Menger 3D")

opcio = input("Escriu 3, 4, D, 6, B, 7 o M: ").strip().upper()

if opcio == "3":
    bmp_folder = r"C:/Users/sanrr/PycharmProjects/TFM/03B-20250619T124743Z-1-001/03B/03B_LR_center"
    output_folder = r"C:/Users/sanrr/PycharmProjects/TFM/RESULTATS_3D_03B"

elif opcio == "4":
    bmp_folder = r"C:/Users/sanrr/PycharmProjects/TFM/04A-20250619T165940Z-1-001/04A/04A_mCT_center"
    output_folder = r"C:/Users/sanrr/PycharmProjects/TFM/RESULTATS_3D_04A"

elif opcio == "D":
    bmp_folder = r"C:/Users/sanrr/PycharmProjects/TFM/03D-20250619T165938Z-1-001/03D/2934D_24_LR_center"
    output_folder = r"C:/Users/sanrr/PycharmProjects/TFM/RESULTATS_3D_03D"

elif opcio == "6":
    bmp_folder = r"C:/Users/sanrr/PycharmProjects/TFM/06A-20250619T124758Z-1-001/06A/06A_LR_center"
    output_folder = r"C:/Users/sanrr/PycharmProjects/TFM/RESULTATS_3D_06A"

elif opcio == "B":
    bmp_folder = r"C:/Users/sanrr/PycharmProjects/TFM/06B-20250619T125027Z-1-001/06B/06B_LR_center"
    output_folder = r"C:/Users/sanrr/PycharmProjects/TFM/RESULTATS_3D_06B"

elif opcio == "7":
    bmp_folder = r"C:/Users/sanrr/PycharmProjects/TFM/07B-20250619T124803Z-1-001/07B/07B_LR_center"
    output_folder = r"C:/Users/sanrr/PycharmProjects/TFM/RESULTATS_3D_07B"

elif opcio == "M":
    bmp_folder = r"C:/Users/sanrr/PycharmProjects/TFM/MENGER_3D"
    output_folder = r"C:/Users/sanrr/PycharmProjects/TFM/RESULTATS_3D_MENGER"

else:
    raise ValueError("Opció no vàlida.")

os.makedirs(output_folder, exist_ok=True)
print(f"\nAnalitzant carpeta: {bmp_folder}\n")

# ----------------------------
# LLEGIR TOTES LES IMATGES BMP
# ----------------------------
bmp_files = sorted([f for f in os.listdir(bmp_folder) if f.endswith('.bmp')])
if not bmp_files:
    raise ValueError(f"No s'han trobat imatges BMP a {bmp_folder}")

imgs = []
for f in bmp_files:
    img = cv2.imread(os.path.join(bmp_folder, f), cv2.IMREAD_GRAYSCALE)
    imgs.append(img)
arr3d = np.stack(imgs, axis=0)
print(f"Volum carregat: {arr3d.shape} (Z, Y, X)")
# ----------------------------
# SELECCIÓ DE ROI MANUAL
# ----------------------------
zmid = arr3d.shape[0] // 2
img_mid = arr3d[zmid]
fig, ax = plt.subplots()
ax.imshow(img_mid, cmap='gray')
cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
coords = []

def onclick(event):
    coords.append((int(event.xdata), int(event.ydata)))
    plt.close()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.title("Fes clic al centre de la ROI")
plt.show()

if not coords:
    raise ValueError("No s'ha seleccionat cap punt.")
x0, y0 = coords[0]
half = roi_size // 2

# ----------------------------
# EXTRACCIÓ DE ROI 3D
# ----------------------------
z_start = max(0, zmid - half)
z_end   = min(arr3d.shape[0], zmid + half)
y_start = max(0, y0 - half)
y_end   = min(arr3d.shape[1], y0 + half)
x_start = max(0, x0 - half)
x_end   = min(arr3d.shape[2], x0 + half)

roi3d = arr3d[z_start:z_end, y_start:y_end, x_start:x_end]
print(f"ROI 3D extreta: {roi3d.shape}")

# ----------------------------
# CÀLCUL DE PSD I FD 3D
# ----------------------------
def get_radial_profile_3d(psd3d):
    """Càlcul del perfil radial 3D de l'espectre de potència"""
    center = np.array(psd3d.shape) // 2
    zz, yy, xx = np.indices(psd3d.shape)
    r = np.sqrt((xx-center[2])**2 + (yy-center[1])**2 + (zz-center[0])**2).astype(np.int32)
    tbin = np.bincount(r.ravel(), psd3d.ravel())
    nr = np.bincount(r.ravel())
    return tbin / (nr + 1e-8)

roi_norm = (roi3d.astype(np.float32) - roi3d.min()) / (roi3d.max() - roi3d.min() + 1e-10)
fft3d = np.abs(fftshift(fftn(roi_norm)))**2
psd3d = fft3d / np.prod(roi3d.shape)
psd_radial = get_radial_profile_3d(psd3d)

# ============================
# CÀLCUL DE λ (longitud d’ona) i PSD
# ============================

r = np.arange(len(psd_radial))
freqs = r / (roi3d.shape[0] * pixel_size_mm)

lambda_vals = 1 / (freqs[1:] + 1e-10)
psd_vals = psd_radial[1:len(lambda_vals) + 1]

# ============================
# FILTRAT DEL RANG λ ENTRE 0.1 I 1.0
# ============================

mask = (lambda_vals >= 0.1) & (lambda_vals <= 1.0)

lambda_used = lambda_vals[mask]
psd_used = psd_vals[mask]

# Si no hi ha punts suficients → evitar crash
if len(lambda_used) < 5:
    raise ValueError("No hi ha punts suficients entre 0.1 i 1.0 per fer l'ajust log-log.")

# ============================
# AJUST LOG-LOG
# ============================

log_lambda = np.log10(lambda_used + 1e-12)
log_psd = np.log10(psd_used + 1e-12)

# Ajust lineal
A = np.vstack([log_lambda, np.ones_like(log_lambda)]).T
slope, intercept = np.linalg.lstsq(A, log_psd, rcond=None)[0]

# Predicció del model
y_pred = slope * log_lambda + intercept

# Coeficient de determinació R²
r2 = 1 - np.sum((log_psd - y_pred)**2) / np.sum((log_psd - np.mean(log_psd))**2)

fd3d = 6 - slope / 2  # relació teòrica per espectre 3D

# ----------------------------
# RESULTATS
# ----------------------------
print("\n=== RESULTATS FD3D ===")
print(f"Dimensió fractal 3D: {fd3d:.4f}")
print(f"Pendent (slope): {slope:.4f}")
print(f"R²: {r2:.4f}")
print("=======================")

# ----------------------------
# GRÀFIC I EXPORTACIÓ
# ----------------------------
plt.figure()
plt.loglog(lambda_vals, psd_vals, 'b-', label='PSD 3D completa')
plt.loglog(lambda_used, 10**(slope*np.log10(lambda_used) + intercept), 'r--', label='Ajust lineal')
plt.xlabel("Longitud d'ona λ (mm)")
plt.ylabel("PSD")
plt.title(f"FD3D = {fd3d:.3f} | R² = {r2:.3f}")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "PSD_loglog_FD3D.png"))
plt.close()

# Desa resultats numèrics
df = pd.DataFrame([{
    'FD3D': fd3d,
    'slope': slope,
    'R2': r2,
    'ROI_center_x': x0,
    'ROI_center_y': y0,
    'ROI_center_z': zmid,
    'ROI_size_px': roi_size
}])
df.to_excel(os.path.join(output_folder, "FD3D_resultats.xlsx"), index=False)

print(f"\nResultats desats a: {output_folder}")
print("Analisi completada amb èxit")
