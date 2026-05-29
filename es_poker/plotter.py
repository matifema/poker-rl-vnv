# plotter per dati di training — salva PNG, no GUI

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

PALETTE = {"A": "#4c72b0", "B": "#dd8452", "A2": "#55a868"}
SPRINT_LABEL = {"A": "A vs Random", "coev-A": "A (coev)", "coev-B": "B (coev)"}


def _carica_df(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["generation"] = df["generation"].astype(int)
    df["profit_mean"] = df["profit_mean"].astype(float)
    df["profit_std"] = df["profit_std"].astype(float)
    df["profit_max"] = df["profit_max"].astype(float)
    df["profit_min"] = df["profit_min"].astype(float)
    df["winrate_mean"] = df["winrate_mean"].astype(float) * 100
    df["update_norm"] = df["update_norm"].astype(float)
    return df


def _plotta(df: pd.DataFrame):
    sprint_nomi = sorted(df["sprint"].unique())

    fig, ((ax_p, ax_wr), (ax_up, ax_sp)) = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle("Training ES — Self-Play", fontsize=16, fontweight="bold", y=0.98)

    for nome in sprint_nomi:
        g = df[df["sprint"] == nome]
        label = SPRINT_LABEL.get(nome, nome)
        col = PALETTE.get(nome, "#999")

        # profit medio ± std
        ax_p.plot(g["generation"], g["profit_mean"],
                  "o-", color=col, label=label, markersize=5, linewidth=2)
        ax_p.fill_between(g["generation"],
                          g["profit_mean"] - g["profit_std"],
                          g["profit_mean"] + g["profit_std"],
                          color=col, alpha=0.12)

        # win rate
        ax_wr.plot(g["generation"], g["winrate_mean"],
                   "o-", color=col, markersize=5, linewidth=2)

        # norma update (convergenza)
        ax_up.plot(g["generation"], g["update_norm"],
                   "o-", color=col, markersize=5, linewidth=2)

        # spread profit (min–max)
        ax_sp.fill_between(g["generation"], g["profit_min"], g["profit_max"],
                           color=col, alpha=0.15, label=label)
        ax_sp.plot(g["generation"], g["profit_mean"],
                   "-", color=col, linewidth=2)

        # annota valori finali
        last = g.iloc[-1]
        ax_p.annotate(f"{last['profit_mean']:+.1f}",
                      (last["generation"], last["profit_mean"]),
                      textcoords="offset points", xytext=(8, 0),
                      fontsize=7.5, color=col, va="center")
        ax_up.annotate(f"{last['update_norm']:.2f}",
                       (last["generation"], last["update_norm"]),
                       textcoords="offset points", xytext=(8, 0),
                       fontsize=7.5, color=col, va="center")

    # --- profit medio ---
    ax_p.axhline(y=0, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax_p.set_ylabel("profit medio", fontsize=12)
    ax_p.legend(fontsize=8, loc="lower left", frameon=True, fancybox=True)
    ax_p.set_title("profit medio ± deviazione std", fontsize=11)

    # --- win rate ---
    ax_wr.axhline(y=50, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax_wr.set_ylabel("win rate (%)", fontsize=12)
    ax_wr.set_title("win rate medio", fontsize=11)

    # --- norma update ---
    ax_up.set_ylabel("||update||", fontsize=12)
    ax_up.set_title("norma update (convergenza ES)", fontsize=11)
    ax_up.set_xlabel("generazione", fontsize=12)

    # --- spread profit ---
    ax_sp.axhline(y=0, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax_sp.set_ylabel("profit", fontsize=12)
    ax_sp.set_title("spread profit (min–max popolazione)", fontsize=11)
    ax_sp.legend(fontsize=8, loc="lower left", frameon=True, fancybox=True)
    ax_sp.set_xlabel("generazione", fontsize=12)

    sns.despine()
    plt.tight_layout()
    return fig


def salva_grafico(csv_path: str | Path, png_path: str | Path | None = None):
    """legge CSV e salva PNG (default: stesso nome con .png)"""
    csv_path = Path(csv_path)
    png_path = Path(png_path) if png_path else csv_path.with_suffix(".png")
    if not csv_path.exists():
        return
    df = _carica_df(csv_path)
    if df.empty:
        return
    fig = _plotta(df)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="plotta dati training da CSV")
    parser.add_argument(
        "csv", nargs="?", default="./training_output/training_history.csv",
        help="percorso file CSV"
    )
    args = parser.parse_args()
    p = Path(args.csv)
    salva_grafico(p)
    print(f"grafico salvato → {p.with_suffix('.png')}")


if __name__ == "__main__":
    main()
