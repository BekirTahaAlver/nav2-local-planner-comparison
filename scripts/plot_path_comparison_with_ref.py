#!/usr/bin/env python3
"""
plot_path_comparison_with_ref.py
-----------------------
Her senaryo icin BIR figur: global referans yol + uc algoritmanin gercek
(ground-truth) yorungesi + start/goal + statik/dinamik engeller. Toplam 4 figur.

Yorungeler HIZ PROFILLERIYLE AYNI CSV'lerden: ayni konfig ve ayni temsili run
(toplam sure hucre ORTALAMASINA en yakin) -> figurler birbirini tutar.
    DWA -> p2v2,  MPPI -> p2v1,  RPP -> p1v3

Referans: results/reference_paths/{SCENARIO}_plan.csv (save_reference_path.py)
Start/Goal: referans yolun ilk/son noktasi.

Cikti: results/path_comparison/path_{SCENARIO}.png
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ============================ CONFIG ============================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR  = os.path.join(PACKAGE_ROOT, "results")
REF_DIR      = os.path.join(RESULTS_DIR, "reference_paths")
OUT_DIR      = os.path.join(RESULTS_DIR, "path_comparison")

SCENARIOS = ["Static", "Dynamic", "Narrow_U", "Narrow_Z"]

PLANNER_CFG = {   # planlayici -> (param, renk)
    "DWA":  ("p2v2", "blue"),
    "MPPI": ("p2v1", "green"),
    "RPP":  ("p1v3", "orange"),
}
FAILED_TAG = "FAILED"
GOAL_TOLERANCE = 0.15   # xy_goal_tolerance [m] -> hedef etrafi cember yaricapi

# ─── STATİK VE DİNAMİK ENGEL BİLGİLERİ ──────────────────────────────────────
STATIC_OBSTACLES = [
    {'x':  1.5, 'y':  1.5, 'radius': 0.45},
    {'x': -1.5, 'y':  1.5, 'radius': 0.45},
    {'x':  0.0, 'y':  0.0, 'radius': 0.45},
    {'x':  1.5, 'y': -1.5, 'radius': 0.45},
    {'x': -1.5, 'y': -1.5, 'radius': 0.45},
]

DYNAMIC_OBSTACLES = [
    {
        'name': 'h1', 'color': '#d62728', 'label': r'$h_1$ (CROSS)',
        'start_x':  2.0, 'start_y':  0.5,
        'vel_x':   -0.12, 'vel_y':  -0.12,
        'radius':   0.25,
        'alpha':    0.55,
    },
    {
        'name': 'h2', 'color': '#1f77b4', 'label': r'$h_2$ (HEAD-ON)',
        'start_x': -1.5, 'start_y':  1.5,
        'vel_x':    0.12, 'vel_y':  -0.12,
        'radius':   0.25,
        'alpha':    1.0,  # h2 icin saydamlik kaldirildi (opac)
    },
    {
        'name': 'h3', 'color': '#2ca02c', 'label': r'$h_3$ (REV-CROSS)',
        'start_x': -2.5, 'start_y': -0.5,
        'vel_x':    0.12, 'vel_y':   0.00,
        'radius':   0.25,
        'alpha':    0.55,
    },
]
# ===============================================================


def draw_static_obstacles(ax):
    """Statik senaryoda 5 silindiri (dice-5 deseni) ciz."""
    for i, obs in enumerate(STATIC_OBSTACLES):
        circle = plt.Circle(
            (obs['x'], obs['y']), obs['radius'],
            facecolor='lightgray', edgecolor='black',
            linewidth=1.2, alpha=0.7,
            label='Static obstacle' if i == 0 else None,
            zorder=1)
        ax.add_patch(circle)


def draw_dynamic_obstacles(ax):
    """
    Dinamik senaryoda 3 silindiri baslangic pozisyonlari + hareket oklariyla ciz.
    """
    for obs in DYNAMIC_OBSTACLES:
        circle = plt.Circle(
            (obs['start_x'], obs['start_y']), obs['radius'],
            facecolor=obs['color'], edgecolor='black',
            linewidth=1.2, alpha=obs.get('alpha', 0.55),
            zorder=2.5)
        ax.add_patch(circle)

        ax.text(obs['start_x'], obs['start_y'], obs['name'],
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='black', zorder=2.6)

        vmag = math.sqrt(obs['vel_x']**2 + obs['vel_y']**2)
        if vmag > 1e-6:
            ux = obs['vel_x'] / vmag
            uy = obs['vel_y'] / vmag
            arrow_start_x = obs['start_x'] + ux * obs['radius']
            arrow_start_y = obs['start_y'] + uy * obs['radius']
            arrow_length = 1.0
            arrow_end_x = arrow_start_x + ux * arrow_length
            arrow_end_y = arrow_start_y + uy * arrow_length
            ax.annotate(
                '', xy=(arrow_end_x, arrow_end_y),
                xytext=(arrow_start_x, arrow_start_y),
                arrowprops=dict(arrowstyle='->', color=obs['color'],
                                lw=2.2, mutation_scale=20),
                zorder=2.6)


def completion_time(df):
    return float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])


def pick_representative(param, planner, scenario):
    """Toplam sure hucre ORTALAMASINA en yakin basarili run (hiz profilleriyle ayni)."""
    pat = os.path.join(RESULTS_DIR, scenario,
                       f"{param}_{planner}_{scenario}_run*.csv")
    files = [f for f in glob.glob(pat)
             if FAILED_TAG.lower() not in os.path.basename(f).lower()]
    data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            data.append((completion_time(df), df))
        except Exception:
            pass
    if not data:
        return None
    times = np.array([d[0] for d in data])
    return data[int(np.argmin(np.abs(times - times.mean())))][1]


def load_reference(scenario):
    f = os.path.join(REF_DIR, f"{scenario}_plan.csv")
    if not os.path.exists(f):
        return None
    return pd.read_csv(f)[["plan_x", "plan_y"]].to_numpy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    made = 0
    for scen in SCENARIOS:
        ref = load_reference(scen)
        fig, ax = plt.subplots(figsize=(7.5, 6.5))

        # --- ENGELLERI CIZ (Senaryoya gore) ---
        if scen == "Static":
            draw_static_obstacles(ax)
        elif scen == "Dynamic":
            draw_dynamic_obstacles(ax)

        if ref is not None:
            ax.plot(ref[:, 0], ref[:, 1], color="black", linestyle="--",
                    linewidth=1.8, label="Reference path", zorder=2)
        else:
            print(f"[UYARI] {scen}: referans yok (reference_paths/{scen}_plan.csv). "
                  f"Once save_reference_path.py.")

        drawn = []
        for planner, (param, color) in PLANNER_CFG.items():
            df = pick_representative(param, planner, scen)
            if df is None:
                print(f"[UYARI] {scen} / {planner} ({param}): basarili run yok.")
                continue
            ax.plot(df["gt_x"].to_numpy(), df["gt_y"].to_numpy(),
                    color=color, linewidth=2.0, label=planner, zorder=3)
            drawn.append(planner)

        if not drawn and ref is None:
            plt.close(fig)
            print(f"[UYARI] {scen}: veri yok, atlaniyor.")
            continue

        # start / goal = referansin ilk/son noktasi
        if ref is not None:
            ax.plot(ref[0, 0],  ref[0, 1],  "o", color="green", markersize=12,
                    markeredgecolor="black", label="Start", zorder=4)
            ax.plot(ref[-1, 0], ref[-1, 1], "o", color="red", markersize=12,
                    markeredgecolor="black", label="Goal", zorder=4)
            ax.add_patch(Circle((ref[-1, 0], ref[-1, 1]), GOAL_TOLERANCE,
                                fill=False, linestyle="--", edgecolor="gray",
                                linewidth=1.2, zorder=4,
                                label=f"Goal tol. ({GOAL_TOLERANCE:.2f} m)"))

        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(f"Path Comparison \u2014 {scen}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True)
        fig.tight_layout()

        out = os.path.join(OUT_DIR, f"path_{scen}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        made += 1
        print(f"Kaydedildi -> results/path_comparison/path_{scen}.png "
              f"({', '.join(drawn)})")

    print(f"\nToplam {made} figur uretildi.")


if __name__ == "__main__":
    main()
