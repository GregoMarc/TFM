import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Configuración de carpetas exactas
# -------------------------------
opc_dict = {
    "1": r"C:/Users/sanrr/PycharmProjects/TFM/03B-20250619T124743Z-1-001/03B/03B_LR_center",
    "2": r"C:/Users/sanrr/PycharmProjects/TFM/03D-20250619T165938Z-1-001/03D/2934D_24_LR_center",
    "3": r"C:/Users/sanrr/PycharmProjects/TFM/04A-20250619T165940Z-1-001/04A/04A_mCT_center",
    "4": r"C:/Users/sanrr/PycharmProjects/TFM/06A-20250619T124758Z-1-001/06A/06A_LR_center",
    "5": r"C:/Users/sanrr/PycharmProjects/TFM/06B-20250619T125027Z-1-001/06B/06B_LR_center",
    "6": r"C:/Users/sanrr/PycharmProjects/TFM/07B-20250619T124803Z-1-001/07B/07B_LR_center"
}

print("Tria el número de la imatge que vols processar:")
for k in opc_dict:
    print(f"{k}: carpeta {opc_dict[k]}")

opc = input("Número: ")
if opc not in opc_dict:
    print("Opció no vàlida. Surt del programa.")
    exit()

bmp_folder = opc_dict[opc]
# Aquí solo cogemos el nombre de la muestra: '03B', '04A', etc.
sample_name = bmp_folder.split("/")[-2]

# -------------------------------
# Carpeta de salida
# -------------------------------
output_folder = r"C:/Users/sanrr/PycharmProjects/TFM/RESULTATS_LACUNARITY"
os.makedirs(output_folder, exist_ok=True)

# -------------------------------
# 2️⃣ Leer todas las imágenes BMP y crear stack 3D
# -------------------------------
bmp_files = sorted([f for f in os.listdir(bmp_folder) if f.endswith(".bmp")])
if not bmp_files:
    print(f"No s'han trobat arxius BMP a: {bmp_folder}")
    exit()

imgs = []
for f in bmp_files:
    img = cv2.imread(os.path.join(bmp_folder, f), cv2.IMREAD_GRAYSCALE)
    imgs.append(img)

stack = np.stack(imgs, axis=0)
print(f"Stack cargado: {stack.shape[0]} slices, tamaño {stack.shape[1]}x{stack.shape[2]}")

# -------------------------------
# 3️⃣ Seleccionar slice central
# -------------------------------
zmid = stack.shape[0] // 2
slice_mid = stack[zmid]

# ⚠️ NO guardar slice central, según tu pedido

# -------------------------------
# 4️⃣ Binarizar la imagen (Otsu)
# -------------------------------
_, slice_bin = cv2.threshold(slice_mid, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
plt.imshow(slice_bin, cmap='gray')
plt.title(f"Slice central binarizada {sample_name}")
plt.show()

# -------------------------------
# 5️⃣ Función lacunaridad 2D
# -------------------------------
def lacunarity_2d(binary_img, box_sizes):
    lac_dict = {}
    img = binary_img.astype(np.float64)
    H, W = img.shape
    for box_size in box_sizes:
        counts = []
        for i in range(0, H - box_size + 1, box_size):
            for j in range(0, W - box_size + 1, box_size):
                patch = img[i:i+box_size, j:j+box_size]
                counts.append(np.sum(patch))
        counts = np.array(counts)
        mean = np.mean(counts)
        var = np.var(counts)
        lac_dict[box_size] = var / (mean**2 + 1e-12) + 1
    return lac_dict

# -------------------------------
# 6️⃣ Definir tamaños de cajas y calcular lacunaridad
# -------------------------------
box_sizes = [2,4,8,16,32,64]
lac_dict = lacunarity_2d(slice_bin, box_sizes)
print("Lacunarity calculada:", lac_dict)

# -------------------------------
# 7️⃣ Guardar resultados en Excel acumulativo
# -------------------------------
excel_path = os.path.join(output_folder, "lacunarity_all_samples.xlsx")
df_sample = pd.DataFrame({
    "Sample": [sample_name]*len(lac_dict),
    "Box size": list(lac_dict.keys()),
    "Lacunarity": list(lac_dict.values())
})

# ⚡ Evitar PermissionError: sobrescribir directamente (cerrar Excel antes)
if os.path.exists(excel_path):
    df_all = pd.read_excel(excel_path)
    df_all = pd.concat([df_all, df_sample], ignore_index=True)
else:
    df_all = df_sample

df_all.to_excel(excel_path, index=False)
print(f"Resultados guardados/actualizados en: {excel_path}")

# -------------------------------
# 8️⃣ Graficar lacunaridad vs tamaño de caja
# -------------------------------
plt.figure()
plt.plot(list(lac_dict.keys()), list(lac_dict.values()), marker='o')
plt.xlabel("Box size")
plt.ylabel("Lacunarity")
plt.title(f"Lacunarity 2D - Slice central {sample_name}")
plt.grid(True)
plt.show()


