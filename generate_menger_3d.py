import os
import numpy as np
import imageio
from scipy.ndimage import gaussian_filter

# -------------------------------
# GENERADOR MENGER SPONGE 3D
# -------------------------------

def menger_sponge(iterations):
    n = iterations
    size = 3**n
    vol = np.zeros((size, size, size), dtype=np.uint8)

    def build(level):
        if level == 0:
            return [(0,0,0)]
        prev = build(level-1)
        res = []
        step = 3**(level-1)
        for z in range(3):
            for y in range(3):
                for x in range(3):
                    # Retirem el centre i els centres de les 3 direccions
                    if (x==1 and y==1) or (x==1 and z==1) or (y==1 and z==1):
                        continue
                    for (px,py,pz) in prev:
                        res.append((
                            x*step + px,
                            y*step + py,
                            z*step + pz
                        ))
        return res

    coords = build(n)

    for (x,y,z) in coords:
        vol[y, x, z] = 1

    return vol.astype(np.float32)



# -------------------------------
# GENERAR + DESAR COM BMP STACK
# -------------------------------

def generate_and_save(iterations=3, out_folder=r"C:\Users\sanrr\PycharmProjects\TFM\MENGER_3D", smooth_sigma=0.7):

    vol = menger_sponge(iterations)

    # Suavitzar una mica per anti-aliasing (opcional)
    vol = gaussian_filter(vol, sigma=smooth_sigma)

    os.makedirs(out_folder, exist_ok=True)

    depth = vol.shape[2]

    for z in range(depth):
        slice_z = vol[:, :, z]

        # Normalitzar manualment (NumPy 2.0 compatible)
        minv = np.min(slice_z)
        maxv = np.max(slice_z)
        ptp = np.ptp(slice_z)

        if ptp < 1e-12:
            slice_norm = np.zeros_like(slice_z)
        else:
            slice_norm = (slice_z - minv) / (ptp + 1e-12)

        slice_img = (slice_norm * 255).astype(np.uint8)

        fname = os.path.join(out_folder, f"menger_z{z:03d}.bmp")
        imageio.imwrite(fname, slice_img)

    print("====================================")
    print("   ✔ MENGER SPONGE 3D GENERADA")
    print("====================================")
    print("Carpeta:", out_folder)
    print("Mida volum:", vol.shape)

    return out_folder



# -------------------------------
# EXECUCIÓ DIRECTA
# -------------------------------

if __name__ == "__main__":
    generate_and_save(
        iterations=4,   # pots provar 3 si vols que sigui més ràpid
        out_folder=r"C:\Users\sanrr\PycharmProjects\TFM\MENGER_3D",
        smooth_sigma=0.5
    )
