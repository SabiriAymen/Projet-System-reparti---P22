import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

with open("output/results.json") as f:
    results = json.load(f)

base = results["baseline"]
sent = results["sentiment"]

plt.rcParams.update({"font.size": 11})

# 1. Courbes de perte (loss) - comparaison des deux entraînements
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, res, title, color in zip(axes, [base, sent], ["A - Sans sentiment", "B - Avec sentiment"], ["#065A82", "#1C7293"]):
    ax.plot(res["history_loss"], label="Train loss", color=color)
    ax.plot(res["history_val_loss"], label="Validation loss", color="#F96167")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (échelle normalisée)")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/fig_loss_curves.png", dpi=150)
plt.close()

# 2. Prédictions vs réel - les deux modèles superposés
dates = pd.to_datetime(base["test_dates"])
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(dates, base["actual"], label="Prix réel", color="#212121", linewidth=2)
ax.plot(dates, base["pred"], label="Prédit - Sans sentiment", color="#F96167", linestyle="--")
ax.plot(dates, sent["pred"], label="Prédit - Avec sentiment", color="#028090", linestyle="--")
ax.set_title("Prédiction du cours AAPL (jeu de test) : réel vs modèles")
ax.set_xlabel("Date")
ax.set_ylabel("Prix de clôture ($)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/fig_predictions.png", dpi=150)
plt.close()

# 3. Comparaison des métriques (barres)
metrics = ["rmse", "mae", "mape"]
labels = ["RMSE ($)", "MAE ($)", "MAPE (%)"]
base_vals = [base[m] for m in metrics]
sent_vals = [sent[m] for m in metrics]

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(metrics))
width = 0.35
bars1 = ax.bar([i - width / 2 for i in x], base_vals, width, label="Sans sentiment", color="#B85042")
bars2 = ax.bar([i + width / 2 for i in x], sent_vals, width, label="Avec sentiment", color="#028090")
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_title("Comparaison des métriques d'erreur (plus bas = meilleur)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
for bars in (bars1, bars2):
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("output/fig_metrics_comparison.png", dpi=150)
plt.close()

# 4. R2 comparaison
fig, ax = plt.subplots(figsize=(5, 5))
r2_vals = [base["r2"], sent["r2"]]
bars = ax.bar(["Sans sentiment", "Avec sentiment"], r2_vals, color=["#B85042", "#028090"])
ax.set_title("Score R² (plus haut = meilleur)")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3, axis="y")
for b in bars:
    ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("output/fig_r2_comparison.png", dpi=150)
plt.close()

print("4 graphiques générés dans output/")
