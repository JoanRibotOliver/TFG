import pandas as pd
from ics import Calendar
from datetime import datetime
import os
import matplotlib.pyplot as plt

# === CONFIGURACIÓ ===
carpeta_ics = "/Users/joanribot/Desktop/Física/5è curs/2n semestre/TFG/calendari_TFG_exports"  # Canvia-ho per la carpeta on tinguis els .ics
paraula_clau = "TFG"
data_inici = datetime(2024, 5, 1)
data_fi = datetime(2025, 7, 31, 23, 59, 59)

esdeveniments_filtrats = []

# === FUNCIÓ PER LLEGIR UN FITXER ICS I AFEGIR ELS ESDEVENIMENTS ===
def llegir_ics(path_ics):
    with open(path_ics, "r", encoding="utf-8") as f:
        c = Calendar(f.read())
    for e in c.events:
        if not e.name:
            continue
        inici = e.begin.datetime.replace(tzinfo=None)
        final = e.end.datetime.replace(tzinfo=None)
        if paraula_clau in e.name and data_inici <= inici <= data_fi:
            durada = (final - inici).total_seconds() / 3600
            esdeveniments_filtrats.append({
                "Data": inici,
                "Durada (hores)": round(durada, 2),
                "Títol": e.name,
                "Final": final,
                "Origen": os.path.basename(path_ics)  # Opcional: saber de quin fitxer ve
            })

# === LLEGIR TOTS ELS .ICS DE LA CARPETA ===
for fitxer in os.listdir(carpeta_ics):
    if fitxer.endswith(".ics"):
        ruta_fitxer = os.path.join(carpeta_ics, fitxer)
        llegir_ics(ruta_fitxer)

# === CREAR DATAFRAME I ORDENAR ===
df = pd.DataFrame(esdeveniments_filtrats)
if df.empty:
    print("No s'han trobat esdeveniments amb la paraula clau i dins el període indicat.")
    exit()

df = df.sort_values(by="Durada (hores)", ascending=False)

# === RESUM MENSUAL ===
df["Mes"] = df["Data"].dt.to_period("M")
resum_mes = df.groupby("Mes")["Durada (hores)"].sum().reset_index()
resum_mes["Mes"] = resum_mes["Mes"].astype(str)

# === GRÀFIC DE BARRA DEL RESUM MENSUAL ===
plt.figure(figsize=(12, 6))
plt.bar(resum_mes["Mes"], resum_mes["Durada (hores)"], color='skyblue')
plt.title('Hores TFG per Mes')
plt.xlabel('Mes')
plt.ylabel('Durada (hores)')
plt.xticks(rotation=45)
plt.tight_layout()

# === GUARDAR FITXERS EN UNA CARPETA ESPECÍFICA ===
output_dir = "/Users/joanribot/Desktop/Física/5è curs/2n semestre/TFG/Joan/Presentació/Hores_TFG"  # Canvia-ho per la carpeta desitjada
os.makedirs(output_dir, exist_ok=True)

df[["Data", "Durada (hores)", "Títol", "Origen"]].to_csv(
    os.path.join(output_dir, "hores_TFG_complet.txt"), sep="\t", index=False
)
resum_mes.to_csv(
    os.path.join(output_dir, "hores_TFG_resum_mensual.txt"), sep="\t", index=False
)
plt.savefig(os.path.join(output_dir, "hores_TFG_resum_mensual.png"))
print(f"Fitxers exportats a: {output_dir}")
plt.show()

# === IMPRIMIR RESULTATS ===
print(f"Total hores treballades: {df['Durada (hores)'].sum():.2f} hores")
print(f"Total esdeveniments trobats: {len(df)}")
print("Fitxers exportats: hores_TFG_complet.txt, hores_TFG_resum_mensual.txt")
