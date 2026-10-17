import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dialect_probing.config import ROOT, load_config

SERIES = ["#2a78d6", "#eb6834"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
AREAS = ["east", "south", "north", "central", "west"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7,
})

def main():
    cfg = load_config()
    out = cfg["out_dir"] / "spont"
    a = pd.read_csv(out / "areas.csv").query("set == 'nbtale_m3' and area != 'all'")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, (model, label) in enumerate([("norsk-asr greedy", "Greedy"), ("norsk-asr lm", "With LM")]):
        d = a[a["model"] == model].set_index("area").loc[AREAS]
        x = [i + (k - 0.5) * 0.18 for i in range(len(AREAS))]
        ax.errorbar(x, d["wer"] * 100, yerr=[(d["wer"] - d["wer_lo"]) * 100, (d["wer_hi"] - d["wer"]) * 100],
                    fmt="o", color=SERIES[k], ecolor=SERIES[k], elinewidth=1.5, capsize=0, label=label)
    n = a[a["model"] == "norsk-asr greedy"].set_index("area").loc[AREAS, "speakers"]
    ax.set_xticks(range(len(AREAS)), [f"{x.capitalize()}\n{n[x]} speakers" for x in AREAS])
    ax.set_ylabel("WER (%)")
    ax.set_title("Spontaneous speech WER by area")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "spontaneous_areas.png", dpi=160)
    print("wrote spontaneous_areas.png")

if __name__ == "__main__":
    main()
