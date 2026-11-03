"""Reporte visible de data drift: un lote del mismo origen (VERDE) y uno con deriva inyectada (ROJO)."""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.drift import detectores as D  # noqa: E402
from src.drift.escenarios import deriva_fuerte  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
EVID = RAIZ / "docs" / "evidencias"


def main():
    ref = pd.read_csv(RAIZ / "data/processed/train.csv")
    pool = pd.concat([pd.read_csv(RAIZ / "data/processed/val.csv"), pd.read_csv(RAIZ / "data/processed/test.csv")])
    base = pool.sample(n=500, random_state=1)
    casos = {"VERDE_mismo_origen": base, "ROJO_deriva_inyectada": deriva_fuerte(base, 1)}
    txt, fig, axs = [], *plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (nombre, lote) in zip(axs, casos.items()):
        t, r = D.evaluar_lote(ref, lote)
        txt += [f"##### {nombre}: estado = {r['estado']}  (variables en alerta: {r['fraccion_en_alerta']:.0%})",
                t.round(4).to_string(index=False), ""]
        col = ["#d62728" if a else "#2ca02c" for a in t["alerta"]]
        ax.barh(t["variable"], t["estadistico"], color=col)
        ax.set_title(f"{r['estado']} - {nombre}\n(barra = KS o PSI; rojo = alerta)")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(EVID / "data_drift_verde_vs_rojo.png", dpi=120)
    (EVID / "data_drift_reporte.txt").write_text("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
