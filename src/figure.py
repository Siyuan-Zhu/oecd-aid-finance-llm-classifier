"""Figure: test accuracy for each 5-digit purpose code, for the two baselines."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLOURS = {"TF-IDF logistic regression": "#0072B2", "TF-IDF nearest neighbours": "#E69F00"}


def make_figure(
    accuracy_by_model: dict[str, dict[str, float]],
    counts: dict[str, int],
    names: dict[str, str],
    sector_names: dict[str, str],
    path: Path,
) -> None:
    codes = sorted(counts)
    models = list(accuracy_by_model)
    height = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 7.6), dpi=130)
    y_labels, y_positions, sector_rows = [], [], {}
    row = 0.0
    previous_sector = None
    for code in codes:
        if code[:3] != previous_sector:
            row += 0.7 if previous_sector is not None else 0.0
            sector_rows[code[:3]] = row - 0.7
            previous_sector = code[:3]
        y_positions.append(row)
        y_labels.append(f"{code}  {names[code]}  (n={counts[code]})")
        for j, model in enumerate(models):
            value = accuracy_by_model[model][code]
            y = row + (j - (len(models) - 1) / 2) * height
            ax.barh(y, value, height=height * 0.92, color=COLOURS.get(model, "#555555"),
                    label=model if code == codes[0] else None)
            ax.text(value + 0.012, y, f"{value:.0%}", va="center", ha="left", fontsize=7.5, color="#333333")
        row += 1.0
    for sector, top in sector_rows.items():
        ax.text(-0.0, top + 0.05, f"Sector {sector}  {sector_names[sector]}", fontsize=8.5, fontweight="bold",
                color="#222222", va="center", transform=ax.get_yaxis_transform())
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_xlabel("Share of test projects given the right 5-digit code (the 3-digit sector is known)", fontsize=9)
    fig.suptitle("OECD CRS health sector: test accuracy by purpose code", fontsize=11, x=0.02, ha="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8.5, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
