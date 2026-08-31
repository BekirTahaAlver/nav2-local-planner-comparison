#!/usr/bin/env python3
"""
plot_path_comparison.py
-----------------------
PARAMETRE KARSILASTIRMA: bir parametrenin farkli degerlerinin yorungelerini
(ground-truth) ust uste cizer -> parametreler arasindaki farki gozlemlemek icin.
Referans yol YOK. Statik/dinamik engeller (with_ref ile ayni) arka planda cizilir.

Cizilecek set PLOT_SET'te tanimlanir (istedigin (planner, param) kombinasyonlari).
Temsili run: toplam sure hucre ORTALAMASINA en yakin (hiz profilleriyle ayni).

Cikti: results/path_comparison/param_{SCENARIO}.png
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

# Cizilecek parametre seti: (planner, param). Her biri ayri renk/label alir.
# Ornek: DWA sim_time taramasi. Istedigin kombinasyonu buraya yaz.
PLOT_SET = [
    ("DWA", "p1v1"),
    ("DWA", "p1v2"),
    ("DWA", "p1v3"),
]
# Yorunge paleti: engel renklerinden (statik gri; dinamik kirmizi/mavi/yesil)
# AYRI tutuldu -> karismasin. kirmizi/mavi/yesil/gri KULLANILMADI.
# Yorunge renkleri PARAMETRE DEGERINE gore (index): 1. deger mavi, 2. yesil,
# 3. turuncu ... (algoritma ici parametre taramasi icin).
COLORS = ["blue", "green", "orange", "magenta", "gold", "black"]

GOAL_TOLERANCE = 0.15   # xy_goal_tolerance [m] -> hedef etrafi cember yaricapi

# Etiketler (planlayici-bilincli). Eslesme yoksa ham param kullanilir.
LABEL_MAP = {
    "DWA": {"p1v1": "sim_time=1.0", "p1v2": "sim_time=2.0", "p1v3": "sim_time=3.0",
            "p2v1": "PathAlign=10.0", "p2v2": "PathAlign=32.0", "p2v3": "PathAlign=64.0"},
    "MPPI": {"p1v1": "time_steps=20", "p1v2": "time_steps=40", "p1v3": "time_steps=60",
             "p2v1": "CostCritic=1.5", "p2v2": "CostCritic=3.81", "p2v3": "CostCritic=8.0"},
    "RPP": {"p1v1": "lookahead=1.0", "p1v2": "lookahead=2.0", "p1v3": "lookahead=3.0"},
}
FAILED_TAG = "FAILED"

# ─── STATİK VE DİNAMİK ENGEL BİLGİLERİ (with_ref ile ayni) ──────────────────
STATIC_OBSTACLES = [
    {'x':  1.5, 'y':  1.5, 'radius': 0.45},
    {'x': -1.5, 'y':  1.5, 'radius': 0.45},
    {'x':  0.0, 'y':  0.0, 'radius': 0.45},
    {'x':  1.5, 'y': -1.5, 'radius': 0.45},
    {'x': -1.5, 'y': -1.5, 'radius': 0.45},
]
DYNAMIC_OBSTACLES = [
    {'name': 'h1', 'color': '#d62728', 'start_x':  2.0, 'start_y':  0.5,
     'vel_x': -0.12, 'vel_y': -0.12, 'radius': 0.25, 'alpha': 0.55},
    {'name': 'h2', 'color': '#1f77b4', 'start_x': -1.5, 'start_y':  1.5,
     'vel_x':  0.12, 'vel_y': -0.12, 'radius': 0.25, 'alpha': 1.0},
    {'name': 'h3', 'color': '#2ca02c', 'start_x': -2.5, 'start_y': -0.5,
     'vel_x':  0.12, 'vel_y':  0.00, 'radius': 0.25, 'alpha': 0.55},
]
# ===============================================================


def draw_static_obstacles(ax):
    for i, obs in enumerate(STATIC_OBSTACLES):
        ax.add_patch(plt.Circle(
            (obs['x'], obs['y']), obs['radius'],
            facecolor='lightgray', edgecolor='black', linewidth=1.2, alpha=0.7,
            label='Static obstacle' if i == 0 else None, zorder=1))


def draw_dynamic_obstacles(ax):
    for obs in DYNAMIC_OBSTACLES:
        ax.add_patch(plt.Circle(
            (obs['start_x'], obs['start_y']), obs['radius'],
            facecolor=obs['color'], edgecolor='black', linewidth=1.2,
            alpha=obs.get('alpha', 0.55), zorder=2.5))
        ax.text(obs['start_x'], obs['start_y'], obs['name'],
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='black', zorder=2.6)
        vmag = math.hypot(obs['vel_x'], obs['vel_y'])
        if vmag > 1e-6:
            ux, uy = obs['vel_x'] / vmag, obs['vel_y'] / vmag
            sx = obs['start_x'] + ux * obs['radius']
            sy = obs['start_y'] + uy * obs['radius']
            ax.annotate('', xy=(sx + ux, sy + uy), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle='->', color=obs['color'],
                                        lw=2.2, mutation_scale=20), zorder=2.6)


def label_for(planner, param):
    return LABEL_MAP.get(planner, {}).get(param, param)


def load_reference(scenario):
    """Referans yolu SADECE start/goal ucları icin oku (cizgi cizilmez)."""
    f = os.path.join(REF_DIR, f"{scenario}_plan.csv")
    if not os.path.exists(f):
        return None
    return pd.read_csv(f)[["plan_x", "plan_y"]].to_numpy()


def completion_time(df):
    return float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])


def pick_representative(param, planner, scenario):
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    made = 0
    for scen in SCENARIOS:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))

        # --- ENGELLER (senaryoya gore, with_ref ile ayni) ---
        if scen == "Static":
            draw_static_obstacles(ax)
        elif scen == "Dynamic":
            draw_dynamic_obstacles(ax)

        drawn = []
        first_df = None
        for i, (planner, param) in enumerate(PLOT_SET):
            df = pick_representative(param, planner, scen)
            if df is None:
                print(f"[UYARI] {scen} / {planner} {param}: basarili run yok.")
                continue
            ax.plot(df["gt_x"].to_numpy(), df["gt_y"].to_numpy(),
                    color=COLORS[i % len(COLORS)], linewidth=2.0,
                    label=label_for(planner, param), zorder=3)
            if first_df is None:
                first_df = df
            drawn.append(f"{planner}:{param}")

        if not drawn:
            plt.close(fig)
            print(f"[UYARI] {scen}: veri yok, atlaniyor.")
            continue

        # start / goal: referans varsa onun uclarindan (cizgi cizilmez),
        # yoksa ilk yorungenin uclarindan
        ref = load_reference(scen)
        if ref is not None:
            (sx, sy), (gx, gy) = ref[0], ref[-1]
        else:
            gt = first_df
            sx, sy = float(gt["gt_x"].iloc[0]),  float(gt["gt_y"].iloc[0])
            gx, gy = float(gt["gt_x"].iloc[-1]), float(gt["gt_y"].iloc[-1])
        ax.plot(sx, sy, "o", color="limegreen", markersize=12,
                markeredgecolor="black", label="Start", zorder=5)
        ax.plot(gx, gy, "*", color="red", markersize=16,
                markeredgecolor="black", label="Goal", zorder=5)
        ax.add_patch(Circle((gx, gy), GOAL_TOLERANCE, fill=False,
                            linestyle="--", edgecolor="dimgray", linewidth=1.2,
                            zorder=5, label=f"Goal tol. ({GOAL_TOLERANCE:.2f} m)"))

        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title(f"Parameter Comparison \u2014 {scen}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True)
        fig.tight_layout()

        out = os.path.join(OUT_DIR, f"param_{scen}.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        made += 1
        print(f"Kaydedildi -> results/path_comparison/param_{scen}.png ({', '.join(drawn)})")

    print(f"\nToplam {made} figur uretildi.")


if __name__ == "__main__":
    main()
