import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Path al JSON 
FNAME = "/home/giacomo/Codice/Tesi/metrics_log.json"
if not os.path.exists(FNAME):
    raise SystemExit(f"File {FNAME} non trovato. Esegui prima il main che genera {FNAME}")

# Leggi il JSON con tutte le metriche
with open(FNAME, "r") as f:
    data = json.load(f)

# Ordina le chiavi degli attacchi
attack_keys = sorted(data.get("attacks", {}).keys(), key=lambda x: int(x.split('_')[-1]))
all_keys = ["clean"] + attack_keys

# Trova tutte le metriche disponibili (Hit Ratio, Precision, Recall, NDCG)
metrics_set = set()
if data.get("clean"):
    metrics_set.update(data["clean"][0].keys() if isinstance(data["clean"][0], dict) else [])
metrics = sorted(metrics_set)

if len(metrics) == 0:
    raise SystemExit("Nessuna metrica trovata nel file. Controlla il JSON creato.")

# Costruisci matrice valori: rows = metriche, cols = step
values = {m: [] for m in metrics}
for k in all_keys:
    if k == "clean":
        d_list = data.get("clean", [])
        d = d_list[0] if d_list else {}
    else:
        d = data["attacks"].get(k, {})
    for m in metrics:
        val = d.get(m, np.nan)
        values[m].append(val)

# Cartella per salvare i grafici
plot_dir = "/home/giacomo/Codice/Tesi/plots"
os.makedirs(plot_dir, exist_ok=True)

# -----------------------
# 1) Grafico linee: andamento metriche step by step
plt.figure(figsize=(10, 6))
for m in metrics:
    plt.plot(all_keys, values[m], marker='o', label=m)
plt.xlabel("Scenario")
plt.ylabel("Valore metrica")
plt.title("Andamento metriche (Clean vs Attack Steps)")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "metric_trend.png"))
plt.show()

# -----------------------
# 2) Confronto Clean vs ultimo step attack (bar plot)
last_key = all_keys[-1]
x = np.arange(len(metrics))
clean_vals = [values[m][0] for m in metrics]
last_vals = [values[m][-1] for m in metrics]
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, clean_vals, width, label='Clean')
plt.bar(x + width/2, last_vals, width, label=last_key)
plt.xticks(x, metrics, rotation=45)
plt.ylabel("Valore metrica")
plt.title(f"Confronto Clean vs {last_key}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "metric_comparison.png"))
plt.show()

# -----------------------
# 3) Delta plot (differenza attack_last - clean)
plt.figure(figsize=(10, 6))
deltas = [last_vals[i] - clean_vals[i] for i in range(len(metrics))]
plt.bar(metrics, deltas)
plt.xticks(rotation=45)
plt.ylabel("Delta (Attack - Clean)")
plt.title("Variazione delle metriche causata dall'attacco (ultimo step)")
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "metric_delta.png"))
plt.show()

print(f"Tutti i grafici sono stati salvati in: {plot_dir}")
