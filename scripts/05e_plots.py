import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dialect_probing.config import ROOT, load_config

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FINE = list(range(13))
ORIG = 13

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
})

def layer_axis(ax):
    ax.set_xticks(FINE + [ORIG + 0.4], [str(i) for i in FINE] + ["12\norig"])
    ax.set_xlabel("Layer")

def series(ax, df, col, label, color, value="ratio"):
    d = df.sort_values("layer_idx")
    fine = d[d["layer_idx"] < ORIG]
    ax.plot(fine["layer_idx"], fine[col if col else value], marker="o", color=color, label=label)
    orig = d[d["layer_idx"] == ORIG]
    ax.plot([ORIG + 0.4] * len(orig), orig[col if col else value], marker="o", markerfacecolor=SURFACE,
            markeredgewidth=2, color=color, linestyle="none")

def save(fig, name):
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=160)
    plt.close(fig)

def main():
    cfg = load_config()
    s5 = cfg["out_dir"] / "step5"
    lag = pd.read_csv(s5 / "lag_selection.csv")
    conv = pd.read_csv(s5 / "convergence.csv")
    strata = pd.read_csv(s5 / "convergence_strata.csv")
    probe = pd.read_csv(s5 / "dialect_probe.csv")
    corr = pd.read_csv(s5 / "wer_distance_correlation.csv")
    dist = pd.read_csv(s5 / "distance_from_east.csv")
    err = pd.read_csv(s5 / "error_rates.csv")

    best = lag.loc[lag.groupby("layer_idx")["word_acc"].idxmax()]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    series(ax, best, "shift_s", "Selected lag", SERIES[0])
    layer_axis(ax)
    ax.set_ylabel("Lag (s)")
    ax.set_title("Selected lag per layer")
    save(fig, "lag_per_layer")

    zero = lag[lag["shift_frames"] == 0]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    series(ax, best, "word_acc", "Selected lag", SERIES[0])
    series(ax, zero, "word_acc", "No shift", SERIES[1])
    layer_axis(ax)
    ax.set_ylabel("Accuracy")
    ax.set_title("Word identity probe")
    ax.legend()
    save(fig, "word_probe")

    main_ = conv[(conv["variant"] == "lag") & (conv["subset"] == "all")]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    series(ax, main_, "within", "Within dialect", SERIES[0])
    series(ax, main_, "cross", "Across dialects", SERIES[1])
    layer_axis(ax)
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Same word similarity")
    ax.legend()
    save(fig, "similarity")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for (variant, label), color in zip([("lag", "Lag matched"), ("zero", "No shift"), ("ctc", "CTC window")], SERIES):
        series(ax, conv[(conv["variant"] == variant) & (conv["subset"] == "all")], None, label, color)
    layer_axis(ax)
    ax.set_ylabel("Across / within")
    ax.set_title("Convergence by window")
    ax.legend()
    save(fig, "convergence_windows")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for (subset, label), color in zip([("all", "All speakers"), ("ntnu", "NTNU only"), ("set a", "Set A only")], SERIES):
        series(ax, conv[(conv["variant"] == "lag") & (conv["subset"] == subset)], None, label, color)
    layer_axis(ax)
    ax.set_ylabel("Across / within")
    ax.set_title("Convergence controls")
    ax.legend()
    save(fig, "convergence_controls")

    titles = {"freq_bin": "By frequency", "length_bin": "By length", "category": "By word type"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (col, title) in zip(axes, titles.items()):
        s = strata[strata["stratum"] == col]
        for level, color in zip(s["level"].unique(), SERIES):
            series(ax, s[s["level"] == level], None, f"{level} ({s[s['level'] == level]['types'].iloc[0]})", color)
        layer_axis(ax)
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Across / within")
    save(fig, "convergence_strata")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for (col, label), color in zip([("lag", "Lag matched"), ("zero", "No shift"), ("ctc", "CTC window"), ("lag_ntnu", "NTNU only")], SERIES):
        series(ax, probe, col, label, color)
    ax.axhline(probe["chance"].iloc[0], color=MUTED, linewidth=1)
    ax.text(0, probe["chance"].iloc[0], " Majority", color=MUTED, va="bottom", fontsize=8)
    layer_axis(ax)
    ax.set_ylabel("Accuracy")
    ax.set_title("Dialect group probe")
    ax.legend()
    save(fig, "dialect_probe")

    c = corr[corr["variant"] == "lag"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for (metric, label), color in zip([("wer", "WER"), ("cer", "CER")], SERIES):
        series(ax, c[c["metric"] == metric], "spearman", label, color)
    ax.axhline(0, color=MUTED, linewidth=1)
    layer_axis(ax)
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Error rate and distance from East")
    ax.legend()
    save(fig, "wer_distance")

    pick = c[c["metric"] == "wer"].loc[lambda d: d["spearman"].abs().idxmax()]
    d = dist[(dist["variant"] == "lag") & (dist["layer"] == pick["layer"])].set_index("group")["distance"]
    bm = err[err["written_standard"] == "bokmål"].set_index("group")["wer"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    east = cfg["analysis"]["east_groups"]
    for g in d.index:
        ax.scatter(d[g], bm[g] * 100, s=48, color=SERIES[1] if g in east else SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(g, (d[g], bm[g] * 100), textcoords="offset points", xytext=(5, 3), fontsize=8, color=INK)
    ax.set_xlabel(f"Distance from East, {pick['layer']}")
    ax.set_ylabel("WER, Bokmål prompts (%)")
    ax.set_title("WER and distance from East")
    save(fig, "wer_distance_scatter")
    we = pd.read_csv(s5 / "word_errors.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for (col, label), color in zip([("d_east", "Distance from East"), ("d_all", "Distance from all regions")], SERIES):
        d = we.sort_values("layer_idx")
        fine, orig = d[d["layer_idx"] < ORIG], d[d["layer_idx"] == ORIG]
        ax.fill_between(fine["layer_idx"], fine[f"b_{col}"] - 1.96 * fine[f"se_{col}"],
                        fine[f"b_{col}"] + 1.96 * fine[f"se_{col}"], color=color, alpha=0.15, linewidth=0)
        series(ax, d, f"b_{col}", label, color)
        ax.errorbar([ORIG + 0.4], orig[f"b_{col}"], yerr=1.96 * orig[f"se_{col}"], color=color, linewidth=1.5, capsize=0)
    ax.axhline(0, color=MUTED, linewidth=1)
    layer_axis(ax)
    ax.set_ylabel("Log odds per SD")
    ax.set_title("Token errors and distance")
    ax.legend()
    save(fig, "word_errors")

    ab = pd.read_csv(s5 / "mix_ablation.csv").query("written_standard == 'bokmål'")
    base = ab[ab["condition"] == "baseline"].set_index("group")["wer"]
    groups = [g for g in base.index if g != "all"] + ["all"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for k, (cond, color) in enumerate(zip(["L7-9 halved", "L7-9 removed", "L0-2 removed"], SERIES)):
        d = (ab[ab["condition"] == cond].set_index("group")["wer"] - base)[groups] * 100
        ax.scatter([i + (k - 1) * 0.22 for i in range(len(groups))], d, s=40, color=color, edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=cond)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel("WER change (points)")
    ax.set_title("Mix weight ablation")
    ax.legend(fontsize=8)
    save(fig, "mix_ablation")
    print("wrote", sorted(p.name for p in (ROOT / "results").glob("*.png")))

if __name__ == "__main__":
    main()
