import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# LLEGIR DADES


# Excel global (FD + mecànics)
GLOBAL_EXCEL = r"C:\Users\sanrr\PycharmProjects\TFM\RESULTATS_GLOBALS\resultats_globals_TFM.xlsx"
df_global = pd.read_excel(GLOBAL_EXCEL)

# Excel de lacunaritat
LAC_EXCEL = r"C:\Users\sanrr\PycharmProjects\TFM\RESULTATS_LACUNARITY\lacunarity_all_samples.xlsx"
df_lac = pd.read_excel(LAC_EXCEL)


#AGAFAR NOMÉS LACUNARITAT de BOX = 64

df_lac_64 = df_lac[df_lac["Box size"] == 64][["Mostra", "Lacunarity"]].copy()
df_lac_64 = df_lac_64.rename(columns={"Lacunarity": "Lac_64"})

# Merge correcte
df = pd.merge(df_global, df_lac_64, on="Mostra")


# FUNCIONS DE REGRESSIÓ

def regressio_simple(x, y):
    slope, intercept, r, p, _ = linregress(x, y)
    return r**2, p, slope, intercept

def regressio_multiple(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model.score(X, y), model.coef_, model.intercept_


# CÀLCUL DE TOTES LES REGRESSIONS
resultats = []

# FD2D vs FD3D
r2, p, _, _ = regressio_simple(df["FD2D"], df["FD3D"])
resultats.append({"X": "FD2D", "Y": "FD3D", "R2": r2, "p": p})

# Variables experimentals
Y_vars = ["E_experimental", "BMD"]

for Y in Y_vars:
    # FD2D
    r2, p, _, _ = regressio_simple(df["FD2D"], df[Y])
    resultats.append({"X": "FD2D", "Y": Y, "R2": r2, "p": p})

    # FD3D
    r2, p, _, _ = regressio_simple(df["FD3D"], df[Y])
    resultats.append({"X": "FD3D", "Y": Y, "R2": r2, "p": p})

    # Lacunaritat 64
    r2, p, _, _ = regressio_simple(df["Lac_64"], df[Y])
    resultats.append({"X": "Lac64", "Y": Y, "R2": r2, "p": p})

    # FD3D + Lac64
    X = df[["FD3D", "Lac_64"]].values
    y = df[Y].values
    r2, coefs, _ = regressio_multiple(X, y)

    resultats.append({
        "X": "FD3D + Lac64",
        "Y": Y,
        "R2": r2,
        "coef_FD3D": coefs[0],
        "coef_Lac": coefs[1]
    })


# MOSTRAR TOTS ELS RESULTATS PER PANTALLA


print("RESULTATS COMPLETS TFM")
print("==============================\n")

for r in resultats:
    print(f"X: {r['X']}  →  Y: {r['Y']}")
    print(f"R² = {r['R2']:.3f}")

    if "p" in r:
        print(f"p-value = {r['p']:.4f}")

    if "coef_FD3D" in r:
        print(f"coef FD3D = {r['coef_FD3D']:.3f}")
        print(f"coef Lac64 = {r['coef_Lac']:.3f}")

   

# GRÀFICS CLAU

def plot_simple(x, y, xlabel, ylabel, title):
    m, b = np.polyfit(x, y, 1)
    plt.figure(figsize=(7,5))
    plt.scatter(x, y)
    plt.plot(x, m*x + b)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

# Gràfics importants
plot_simple(df["FD2D"], df["FD3D"], "FD2D", "FD3D", "FD2D vs FD3D")
plot_simple(df["FD3D"], df["E_experimental"], "FD3D", "E", "FD3D vs E")
plot_simple(df["Lac_64"], df["E_experimental"], "Lac64", "E", "Lac64 vs E")
plot_simple(df["FD3D"], df["BMD"], "FD3D", "BMD", "FD3D vs BMD")

# Gràfic 3D FD3D + Lac64 vs E
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["FD3D"], df["Lac_64"], df["E_experimental"])
ax.set_xlabel("FD3D")
ax.set_ylabel("Lac64")
ax.set_zlabel("E")
ax.set_title("FD3D + Lac64 vs E")
plt.show()

