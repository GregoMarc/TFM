# ======================================================
# ANALISI FD 2D (VERSIÓ MULTI-MOSTRA — MARC)
# ======================================================
# - Et pregunta quina mostra vols analitzar (03B o 04A)
# - Llegeix automàticament la carpeta correcta
# - Calcula FD2D i R²
# - Desa els resultats en una carpeta pròpia:
#       /RESULTATS_2D_03B/
#       /RESULTATS_2D_04A/
# ======================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import pandas as pd
from scipy.fftpack import fft2, fftshift
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter

# -----------------------------
# CONFIGURACIÓ DE MOSTRES
# -----------------------------

BASE = r"C:\Users\sanrr\PycharmProjects\TFM"
output_folder = os.path.join(BASE, "FRACTAL_TEST")
os.makedirs(output_folder, exist_ok=True)
FRACTAL_FILE = os.path.join(output_folder, "P2D_729_suau.bmp")



# Funció per generar Sierpinski Carpet binari
def sierpinski_carpet(level):
    size = 3**level
    carpet = np.ones((size, size), dtype=np.uint8) * 255  # tot blanc

    def carve(x, y, n):
        if n == 0:
            return
        step = 3**(n-1)
        carpet[y+step:y+2*step, x+step:x+2*step] = 0
        for dy in range(0, 3*step, step):
            for dx in range(0, 3*step, step):
                if dy == step and dx == step:
                    continue
                carve(x+dx, y+dy, n-1)

    carve(0, 0, level)
    return carpet

# Generar fractal 729x729 (3^6)
fractal_binari = sierpinski_carpet(6)

# Suavitzar amb filtre gaussià per crear anti-aliasing
fractal_suau = gaussian_filter(fractal_binari.astype(np.float32), sigma=1.0)
fractal_suau = np.clip(fractal_suau, 0, 255).astype(np.uint8)

# Desar BMP
file_path = os.path.join(output_folder, "P2D_729_suau.bmp")
cv2.imwrite(file_path, fractal_suau)

print("✅ Fractal generat i desat a:", file_path)
print("Shape:", fractal_suau.shape)





# ------------------------------------------------------
# NUEVO FRACTAL 2D: Cantor 2D suavizado
# ------------------------------------------------------
def cantor2d(level):
    size = 3**level
    img = np.ones((size, size), dtype=np.uint8) * 255  # todo blanco

    def carve(x, y, n):
        if n == 0:
            return
        step = 3**(n-1)
        # hueco central
        img[y+step:y+2*step, x+step:x+2*step] = 0
        for dy in range(0, 3*step, step):
            for dx in range(0, 3*step, step):
                if dy == step and dx == step:
                    continue
                carve(x+dx, y+dy, n-1)

    carve(0, 0, level)
    return img

# Generar Cantor 2D 729x729 (3^6)
fractal_cantor_bin = cantor2d(6)

# Suavizado con filtro gaussiano para grises
fractal_cantor_suau = gaussian_filter(fractal_cantor_bin.astype(np.float32), sigma=1.0)
fractal_cantor_suau = np.clip(fractal_cantor_suau, 0, 255).astype(np.uint8)

# Guardar BMP
cantor_path = os.path.join(output_folder, "P2D_CANTOR_729_suau.bmp")
cv2.imwrite(cantor_path, fractal_cantor_suau)
print("✅ Nuevo fractal Cantor 2D generado y desado en:", cantor_path)





MOSTRES = {
    # --- MOSTRAS ANTIGUAS ---
    "03B": os.path.join(BASE, r"03B-20250619T124743Z-1-001\03B\03B_LR_center"),
    "04A": os.path.join(BASE, r"04A-20250619T165940Z-1-001\04A\04A_mCT_center"),

    # --- MOSTRAS NUEVAS ---
    "03D": os.path.join(BASE, r"03D-20250619T165938Z-1-001\03D\2934D_24_LR_center"),
    "06A": os.path.join(BASE, r"06A-20250619T124758Z-1-001\06A\06A_LR_center"),
    "06B": os.path.join(BASE, r"06B-20250619T125027Z-1-001\06B\06B_LR_center"),
    "07B": os.path.join(BASE, r"07B-20250619T124803Z-1-001\07B\07B_LR_center"),

    # --- FRACTALES CONTROL ---
    "P": os.path.join(BASE, r"FRACTAL_TEST")
}


print("Selecciona la mostra a analitzar:")
print("1 = 03B")
print("2 = 04A")
print("3 = 03D")
print("4 = 06A")
print("5 = 06B")
print("6 = 07B")
print("7 = P (Sierpinski)")
print("8 = Cantor 2D")

opcio = input("Opció (1–8): ").strip()

if opcio == "1":
    mostra = "03B"
elif opcio == "2":
    mostra = "04A"
elif opcio == "3":
    mostra = "03D"
elif opcio == "4":
    mostra = "06A"
elif opcio == "5":
    mostra = "06B"
elif opcio == "6":
    mostra = "07B"
elif opcio == "7":
    mostra = "P"
elif opcio == "8":
    mostra = "CANTOR"
    MOSTRES["CANTOR"] = os.path.join(BASE, r"FRACTAL_TEST")
else:
    raise ValueError("Opció no vàlida. Escriu un nombre del 1 al 8.")

bmp_folder = MOSTRES[mostra]
output_folder = os.path.join(BASE, f"RESULTATS_2D_{mostra}")
os.makedirs(output_folder, exist_ok=True)

pixel_size_mm = 0.038
roi_size = 729

# -----------------------------
# FUNCIONS
# -----------------------------
def get_radial_profile(psd2d):
    center = np.array(psd2d.shape) // 2
    y, x = np.indices(psd2d.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(np.int32)
    tbin = np.bincount(r.ravel(), psd2d.ravel())
    nr = np.bincount(r.ravel())
    return tbin / (nr + 1e-12)

# -----------------------------
# LLEGIR BMP
# -----------------------------
bmp_files = sorted([f for f in os.listdir(bmp_folder) if f.endswith(".bmp")])
if len(bmp_files) == 0:
    raise ValueError("❌ No s’han trobat BMP.")

central_index = len(bmp_files) // 2
img_path = os.path.join(bmp_folder, bmp_files[central_index])
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# -----------------------------
# ROI MANUAL
# -----------------------------
# -----------------------------
# ROI MANUAL (VERSIÓ CORRECTA)
# -----------------------------
coords = []

def roi_callback(eclick, erelease):
    global coords
    # Coordinates of top-left and bottom-right
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

    # Compute centre of ROI from rectangle
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    coords = (cx, cy)
    plt.close()

fig, ax = plt.subplots()
ax.imshow(img, cmap="gray")
ax.set_title(f"Selecciona la ROI arrossegant un rectangle ({mostra})")

selector = RectangleSelector(
    ax,
    roi_callback,
    useblit=True,
    interactive=True,
    button=[1],
    minspanx=5,
    minspany=5,
    spancoords='pixels'
)

plt.show(block=True)


if coords == []:
    raise ValueError("No s'ha seleccionat cap ROI.")

x0, y0 = coords
half = roi_size // 2



x0, y0 = coords
half = roi_size // 2

# Ajustar ROI per no sortir-se de la imatge
y_start = max(0, y0 - half)
y_end   = min(img.shape[0], y0 + half)
x_start = max(0, x0 - half)
x_end   = min(img.shape[1], x0 + half)

roi = img[y_start:y_end, x_start:x_end]




print("ROI shape:", roi.shape)
if roi.size == 0:
    raise ValueError("ROI està buit! Revisa la selecció de la ROI o la imatge original.")





cv2.imwrite(os.path.join(output_folder, "ROI2D.bmp"), roi)










# -----------------------------
# FFT + PSD
# -----------------------------
img_float = (roi - roi.min()) / (roi.max() - roi.min() + 1e-12)
fft_img = np.abs(fftshift(fft2(img_float)))**2
psd_radial = get_radial_profile(fft_img)

r = np.arange(len(psd_radial))
freqs = r / (roi.shape[0] * pixel_size_mm)
lambda_vals = 1 / (freqs[1:] + 1e-12)
psd_vals = psd_radial[1:len(lambda_vals)+1]

# FILTRE 0.1–1.0 mm
mask = (lambda_vals >= 0.1) & (lambda_vals <= 1.0)
lambda_used = lambda_vals[mask]
psd_used = psd_vals[mask]

if len(lambda_used) < 5:
    raise ValueError("No hi ha punts suficients entre 0.1 i 1.0 mm.")

# -----------------------------
# AJUST LOGLOG
# -----------------------------
log_lambda = np.log10(lambda_used)
log_psd = np.log10(psd_used)

A = np.vstack([log_lambda, np.ones_like(log_lambda)]).T
slope, intercept = np.linalg.lstsq(A, log_psd, rcond=None)[0]
y_pred = slope * log_lambda + intercept
r2 = 1 - np.sum((log_psd - y_pred)**2) / np.sum((log_psd - np.mean(log_psd))**2)

FD2D = 4 - slope / 2

# -----------------------------
# GRÀFIC
# -----------------------------
plt.figure(figsize=(7,6))
plt.loglog(lambda_vals, psd_vals, 'b-', alpha=0.6, label="PSD completa")
plt.loglog(lambda_used, psd_used, 'ro', label="Rang utilitzat (0.1–1.0)")
plt.loglog(lambda_used, 10**(slope * log_lambda + intercept), 'g--', label="Ajust lineal")
plt.xlim(1e-1, 1e0)
plt.xlabel("Longitud d'ona λ (mm)")
plt.ylabel("PSD")
plt.title(f"{mostra} — FD2D = {FD2D:.3f} | R² = {r2:.3f}")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, f"PSD_loglog_FD2D_{mostra}.png"))
plt.close()

# -----------------------------
# EXPORTACIÓ
# -----------------------------
df = pd.DataFrame([{
    "Mostra": mostra,
    "FD2D": FD2D,
    "slope": slope,
    "R2": r2,
    "ROI_center_x": x0,
    "ROI_center_y": y0,
    "ROI_size": roi_size,
    "BMP_file": bmp_files[central_index]
}])

df.to_excel(os.path.join(output_folder, f"FD2D_resultats_{mostra}.xlsx"), index=False)

print("\n==============================")
print(f"  ANÀLISI FD 2D {mostra} COMPLETAT")
print("==============================")
print(f"FD2D = {FD2D:.3f}")
print(f"R²   = {r2:.3f}")
print(f"Resultats desats a: {output_folder}")

