"""
analyze_results.py
==================
DWA / MPPI / RPP Yerel Planlayici Karsilastirma Scripti (Academic Style)

Yeni: Path comparison gorseli senaryoya gore engelleri de cizer:
  - Static  : 5 silindir (dice-5 deseni)
  - Dynamic : 3 silindir (h1/h2/h3) + hareket oklari
  - Narrow_Corridor : engel cizimi yok (duvarlar haritada zaten)

Kullanim:
    python3 analyze_results.py
    python3 analyze_results.py --scenario Static
    python3 analyze_results.py --results_dir ../results/Static \
                               --bags_dir    ../bags/Static \
                               --output_dir  ../figures/Static
"""
import os
import sys
import glob
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- ACADEMIC PLOT STYLING (IEEE / LaTeX Style) ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": '-',
    "grid.color": '#b0b0b0',
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
})

# rosbag2_py — CTE/Heading
try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    HAVE_ROSBAG = True
except ImportError:
    HAVE_ROSBAG = False

# ─── AYARLAR ────────────────────────────────────────────────────────────────
ROBOT_RADIUS  = 0.105
SAFE_DISTANCE = 0.35
MIN_CSV_ROWS  = 5

# Robot hiz limitleri (TB3 Burger) — hiz profili grafiklerindeki referans cizgileri
V_MAX_PLOT = 0.22   # m/s   (Burger max lineer)
W_MAX_PLOT = 2.84   # rad/s (Burger max acisal)

COLORS = {
    'DWA':  "blue",
    'MPPI': "green",
    'RPP':  "orange",
}

# ─── DINAMIK SILINDIR BILGILERI ─────────────────────────────────────────────
# Dynamic senaryosunda gorsele eklenecek silindir bilgileri.
# Baslangic pozisyonlari WORLD dosyasindan (dynamic_world.world) alinmali.
# Hareket vektorleri dynamic_test.py'daki ENCOUNTERS listesiyle uyumlu olmali.
#
# !!! ONEMLI: Baslangic pozisyonlari TAHMINI. WORLD dosyasindaki <pose>
#     degerleriyle dogrula ve gerekirse asagidaki sayilari guncelle.
DYNAMIC_OBSTACLES = [
    {
        'name': 'h1', 'color': '#d62728', 'label': r'$h_1$ (CROSS)',
        'start_x':  2.0, 'start_y':  0.5,
        'vel_x':   -0.12, 'vel_y':  -0.12,
        'radius':   0.25,
    },
    {
        'name': 'h2', 'color': '#1f77b4', 'label': r'$h_2$ (HEAD-ON)',
        'start_x': -1.5, 'start_y':  1.5,
        'vel_x':    0.12, 'vel_y':  -0.12,
        'radius':   0.25,
    },
    {
        'name': 'h3', 'color': '#2ca02c', 'label': r'$h_3$ (REV-CROSS)',
        'start_x': -2.5, 'start_y': -0.5,
        'vel_x':    0.12, 'vel_y':   0.00,
        'radius':   0.25,
    },
]

# Statik senaryoda 5 silindir (dice-5 deseni). Pozisyonlari static_world.world'den.
STATIC_OBSTACLES = [
    {'x':  1.5, 'y':  1.5, 'radius': 0.45},
    {'x': -1.5, 'y':  1.5, 'radius': 0.45},
    {'x':  0.0, 'y':  0.0, 'radius': 0.45},
    {'x':  1.5, 'y': -1.5, 'radius': 0.45},
    {'x': -1.5, 'y': -1.5, 'radius': 0.45},
]

# ─── YARDIMCI FONKSIYONLAR ──────────────────────────────────────────────────
def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

def angle_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d <= -math.pi:
        d += 2.0 * math.pi
    return d

def get_representative_pair(pairs):
    times = [p[1]['timestamp'].iloc[-1] - p[1]['timestamp'].iloc[0] for p in pairs]
    median_t = np.median(times)
    best_idx = np.argmin([abs(t - median_t) for t in times])
    return pairs[best_idx], best_idx + 1

# ─── BAG'DEN ILK /plan'I CIKAR ──────────────────────────────────────────────
def extract_first_plan(bag_path):
    if not HAVE_ROSBAG or not os.path.isdir(bag_path):
        return None
    try:
        storage_opts = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_opts = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr')

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_opts, converter_opts)

        topic_types = reader.get_all_topics_and_types()
        type_map = {t.name: t.type for t in topic_types}
        if '/plan' not in type_map:
            return None
        msg_type = get_message(type_map['/plan'])

        while reader.has_next():
            topic, raw, _ts = reader.read_next()
            if topic != '/plan':
                continue
            msg = deserialize_message(raw, msg_type)
            if len(msg.poses) < 2:
                continue
            path = []
            for ps in msg.poses:
                x = ps.pose.position.x
                y = ps.pose.position.y
                q = ps.pose.orientation
                theta = euler_from_quaternion(q.x, q.y, q.z, q.w)
                path.append((x, y, theta))
            path = np.array(path)

            if np.std(path[:, 2]) < 1e-6:
                dx = np.diff(path[:, 0])
                dy = np.diff(path[:, 1])
                derived = np.arctan2(dy, dx)
                derived = np.append(derived, derived[-1])
                path[:, 2] = derived

            return path
    except Exception as e:
        print(f"    bag okuma hatasi: {e}")
        return None
    return None

# ─── CTE & HEADING ──────────────────────────────────────────────────────────
def compute_cte_heading(df, reference_path):
    if 'gt_x' not in df.columns:
        return None
    valid = df[df['gt_x'].notna() & df['gt_y'].notna() & df['gt_theta'].notna()]
    if valid.empty or reference_path is None or len(reference_path) < 2:
        return None

    ref_xy    = reference_path[:, 0:2]
    ref_theta = reference_path[:, 2]
    gt_pts  = valid[['gt_x', 'gt_y']].to_numpy()
    gt_yaws = valid['gt_theta'].to_numpy()

    cte_vals = np.zeros(len(gt_pts))
    heading_vals = np.zeros(len(gt_pts))

    P1 = ref_xy[:-1]
    P2 = ref_xy[1:]
    segment_vecs = P2 - P1
    segment_lengths_sq = np.sum(segment_vecs**2, axis=1)

    for i, r_pos in enumerate(gt_pts):
        pnt_vecs = r_pos - P1
        t = np.sum(pnt_vecs * segment_vecs, axis=1) / (segment_lengths_sq + 1e-10)
        t = np.clip(t, 0.0, 1.0)
        projections = P1 + t[:, np.newaxis] * segment_vecs
        distances = np.linalg.norm(r_pos - projections, axis=1)

        min_idx = np.argmin(distances)
        cte_vals[i] = distances[min_idx]
        ref_yaw_nearest = ref_theta[min_idx]
        heading_vals[i] = abs(angle_diff(gt_yaws[i], ref_yaw_nearest))

    return {
        'cte_rmse':     float(np.sqrt(np.mean(cte_vals ** 2))),
        'cte_mean':     float(np.mean(cte_vals)),
        'heading_rmse': float(np.sqrt(np.mean(heading_vals ** 2))),
        'heading_mean': float(np.mean(heading_vals)),
    }

# ─── VERI YUKLEME ───────────────────────────────────────────────────────────
def load_csvs_and_bags(results_dir, bags_dir, algorithm, scenario):
    pattern = os.path.join(results_dir, f"{algorithm}_{scenario}_run*.csv")
    all_files = sorted(glob.glob(pattern))
    all_files = [f for f in all_files if '_refpath.csv' not in f]

    success_files = [f for f in all_files if '_FAILED' not in os.path.basename(f)]
    failed_files  = [f for f in all_files if '_FAILED' in os.path.basename(f)]

    pairs = []
    for f in success_files:
        df = pd.read_csv(f)
        if len(df) < MIN_CSV_ROWS:
            continue
        basename = os.path.basename(f).replace('.csv', '')
        bag_path = os.path.join(bags_dir, basename) if bags_dir else None
        pairs.append((basename, df, bag_path))

    failed_names = [os.path.basename(f).replace('.csv', '') for f in failed_files]
    return pairs, len(pairs), len(failed_files), failed_names

# ─── METRIK HESAPLAMA ───────────────────────────────────────────────────────
def compute_metrics(df, bag_path=None):
    df = df.copy().reset_index(drop=True)
    total_time = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
    dx = df['pos_x'].diff().fillna(0)
    dy = df['pos_y'].diff().fillna(0)
    path_length = np.sum(np.sqrt(dx**2 + dy**2))
    avg_linear_vel  = df['linear_vel'].abs().mean()
    avg_angular_vel = df['angular_vel'].abs().mean()
    dt = df['timestamp'].diff().replace(0, np.nan)
    dv = df['linear_vel'].diff()
    velocity_smoothness = (dv / dt).abs().mean()
    path_smoothness = float(np.nansum(dx.diff()**2 + dy.diff()**2))

    valid_lidar = df[df['min_scan_dist'] < 90.0]['min_scan_dist']
    if len(valid_lidar) > 0:
        net_min_dist = valid_lidar.min() - ROBOT_RADIUS
        unsafe_time  = dt[df['min_scan_dist'] < (SAFE_DISTANCE + ROBOT_RADIUS)].sum() if total_time > 0 else 0
        danger_ratio = (unsafe_time / total_time) * 100 if total_time > 0 else 0
    else:
        net_min_dist, danger_ratio = np.nan, np.nan

    cte_rmse, cte_mean, heading_rmse, heading_rmse_deg = np.nan, np.nan, np.nan, np.nan
    if bag_path is not None and HAVE_ROSBAG and 'gt_x' in df.columns:
        ref_path = extract_first_plan(bag_path)
        if ref_path is not None:
            m = compute_cte_heading(df, ref_path)
            if m is not None:
                cte_rmse, cte_mean = m['cte_rmse'], m['cte_mean']
                heading_rmse = m['heading_rmse']
                heading_rmse_deg = math.degrees(m['heading_rmse'])

    return {
        'total_time':           total_time,
        'path_length':          path_length,
        'avg_linear_vel':       avg_linear_vel,
        'avg_angular_vel':      avg_angular_vel,
        'velocity_smoothness':  velocity_smoothness,
        'path_smoothness':      path_smoothness,
        'net_min_dist':         net_min_dist,
        'danger_ratio':         danger_ratio,
        'cte_rmse':             cte_rmse,
        'heading_rmse_deg':     heading_rmse_deg,
    }

def aggregate_metrics(pairs):
    if not pairs:
        return {}
    all_metrics = [compute_metrics(df, bag) for _, df, bag in pairs]
    agg = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics if not (m[key] is None or np.isnan(m[key]))]
        agg[key] = {
            'mean': np.mean(vals) if vals else np.nan,
            'std':  np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
            'n':    len(vals),
            'raw':  vals,
        }
    return agg

# ─── ENGEL CIZIM YARDIMCILARI ───────────────────────────────────────────────
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
    Z-order: 2.5 (referans yolun ustunde, robot yorungelerinin altinda).
    Bu sayede silindirin uzerinden gecen kesik referans cizgisi okunabilir kalir.
    """
    for obs in DYNAMIC_OBSTACLES:
        # Silindirin baslangic cemberi
        circle = plt.Circle(
            (obs['start_x'], obs['start_y']), obs['radius'],
            facecolor=obs['color'], edgecolor='black',
            linewidth=1.2, alpha=0.55,
            zorder=2.5)
        ax.add_patch(circle)

        # Etiket
        ax.text(obs['start_x'], obs['start_y'], obs['name'],
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='black', zorder=2.6)

        # Hareket oku — silindirin KENARINDAN baslat (merkez degil)
        vmag = math.sqrt(obs['vel_x']**2 + obs['vel_y']**2)
        if vmag > 1e-6:
            ux = obs['vel_x'] / vmag    # birim vektor
            uy = obs['vel_y'] / vmag
            # Ok baslangici: silindirin disinda (radius kadar otelenmis)
            arrow_start_x = obs['start_x'] + ux * obs['radius']
            arrow_start_y = obs['start_y'] + uy * obs['radius']
            # Ok uzunlugu (gorsel olcek)
            arrow_length = 1.0   # m
            arrow_end_x = arrow_start_x + ux * arrow_length
            arrow_end_y = arrow_start_y + uy * arrow_length
            ax.annotate(
                '', xy=(arrow_end_x, arrow_end_y),
                xytext=(arrow_start_x, arrow_start_y),
                arrowprops=dict(arrowstyle='->', color=obs['color'],
                                lw=2.2, mutation_scale=20),
                zorder=2.6)

# ─── GORSELLER ──────────────────────────────────────────────────────────────
def plot_path_comparison(pairs_dict, output_dir, scenario='Static'):
    fig, ax = plt.subplots(figsize=(7, 9))

    # ── 1. Engelleri ciz (z-order 1) ──────────────────────────────────────
    if scenario == 'Dynamic':
        draw_dynamic_obstacles(ax)
    elif scenario == 'Static':
        draw_static_obstacles(ax)

    # ── 2. Referans Yol Çizimi (z-order 2) ────────────────────────────────
    first_algo = list(pairs_dict.keys())[0]
    if pairs_dict[first_algo]:
        _, _, bag_path = pairs_dict[first_algo][0]
        if bag_path:
            ref = extract_first_plan(bag_path)
            if ref is not None:
                ax.plot(np.asarray(ref[:, 0]), np.asarray(ref[:, 1]),
                        'k--', linewidth=1.5, label='Reference Path', zorder=2)

    # ── 3. Algoritma yorungelerini ciz (z-order 3) ────────────────────────
    for algo, pairs in pairs_dict.items():
        if not pairs:
            continue
        best_pair, _ = get_representative_pair(pairs)
        _, df, _ = best_pair

        if 'gt_x' in df.columns and df['gt_x'].notna().any():
            valid_df = df.dropna(subset=['gt_x', 'gt_y'])
            xs, ys = valid_df['gt_x'], valid_df['gt_y']
        else:
            valid_df = df.dropna(subset=['pos_x', 'pos_y'])
            xs, ys = valid_df['pos_x'], valid_df['pos_y']

        ax.plot(np.asarray(xs), np.asarray(ys),
                color=COLORS.get(algo, 'gray'),
                linewidth=2.0, label=algo, zorder=3)

    # ── 4. Baslangic ve Hedef noktalari (z-order 4) ───────────────────────
    first_algo = list(pairs_dict.keys())[0]
    if pairs_dict[first_algo]:
        _, first_df, first_bag = pairs_dict[first_algo][0]
        if 'gt_x' in first_df.columns:
            valid_df = first_df.dropna(subset=['gt_x', 'gt_y'])
            if not valid_df.empty:
                sx = valid_df['gt_x'].iloc[0]
                sy = valid_df['gt_y'].iloc[0]
                # Start: dolgulu daire (nokta)
                ax.plot(sx, sy, marker='o', color='green', markersize=12,
                        markeredgecolor='black', markeredgewidth=1.0,
                        label='Start', zorder=4, linestyle='None')

        if first_bag:
            ref = extract_first_plan(first_bag)
            if ref is not None and len(ref) > 0:
                gx, gy = ref[-1, 0], ref[-1, 1]
                # Goal: yildiz
                ax.plot(gx, gy, marker='*', color='red', markersize=18,
                        markeredgecolor='black', markeredgewidth=1.0,
                        label='Goal', zorder=4, linestyle='None')

    ax.set_xlabel(r'$x$ [m]')
    ax.set_ylabel(r'$y$ [m]')
    # Legend: eksenin USTUNDE, yukari dogru buyuyecek sekilde yerlestirilir
    # (loc='lower center' + anchor y>1.0). Boylece grafik alaniyla cakismaz.
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=3, handlelength=1.8, columnspacing=1.0,
              frameon=True, fontsize=10)
    ax.set_aspect('equal')
    plt.tight_layout()

    out = os.path.join(output_dir, 'path_comparison_academic.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Kaydedildi: {out}")

def plot_velocity_profiles(pairs_dict, output_dir):
    fig, axes = plt.subplots(len(pairs_dict), 2, figsize=(10, 3.0 * len(pairs_dict)))
    if len(pairs_dict) == 1:
        axes = [axes]

    for i, (algo, pairs) in enumerate(pairs_dict.items()):
        if not pairs:
            continue

        best_pair, _ = get_representative_pair(pairs)
        _, df, _ = best_pair
        t = df['timestamp'] - df['timestamp'].iloc[0]
        t_arr = np.asarray(t)
        v_arr = np.asarray(df['linear_vel'])
        w_arr = np.asarray(df['angular_vel'])

        xlabel_v = r'$t$ [s]' + '\n\n' + f"({chr(97+i*2)}) {algo} - Linear"
        xlabel_w = r'$t$ [s]' + '\n\n' + f"({chr(98+i*2)}) {algo} - Angular"

        ax_v = axes[i][0]
        ax_v.plot(t_arr, v_arr, color='blue', linewidth=1.2)
        ax_v.set_ylabel(r'$v$ [m/s]')
        ax_v.set_xlabel(xlabel_v, fontsize=12)
        ax_v.axhline(V_MAX_PLOT, color='k', linestyle='--', linewidth=1.0, alpha=0.7)
        ax_v.set_ylim(-0.05, 0.30)

        ax_w = axes[i][1]
        ax_w.plot(t_arr, w_arr, color='blue', linewidth=1.2)
        ax_w.set_ylabel(r'$\omega$ [rad/s]')
        ax_w.set_xlabel(xlabel_w, fontsize=12)
        ax_w.axhline(W_MAX_PLOT,  color='k', linestyle='--', linewidth=1.0, alpha=0.7)
        ax_w.axhline(-W_MAX_PLOT, color='k', linestyle='--', linewidth=1.0, alpha=0.7)
        ax_w.set_ylim(-3.1, 3.1)

    plt.tight_layout(h_pad=3.0, w_pad=2.0)
    out = os.path.join(output_dir, 'velocity_profiles_academic.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Kaydedildi: {out}")

def plot_bar_comparison(results, output_dir):
    metrics = [
        ('total_time', 'Time [s]',     True),
        ('cte_rmse',   'CTE RMSE [m]', True),
    ]
    algos = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (metric, ylabel, lower_is_better) in zip(axes, metrics):
        if metric not in results[algos[0]]:
            continue
        means = [results[a][metric]['mean'] for a in algos]
        stds  = [results[a][metric]['std']  for a in algos]
        ax.bar(algos, means, yerr=stds, capsize=4,
               color=[COLORS.get(a, 'gray') for a in algos],
               alpha=0.8, edgecolor='black', width=0.5)
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    out = os.path.join(output_dir, 'bar_comparison_academic.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Kaydedildi: {out}")

# ─── OZET CIKTI ─────────────────────────────────────────────────────────────
def print_summary_table(results, run_counts):
    metrics_display = [
        ('total_time',          'Sure (s)',              '.2f'),
        ('path_length',         'Yol (m)',               '.3f'),
        ('avg_linear_vel',      'Dog. Hiz (m/s)',        '.3f'),
        ('avg_angular_vel',     'Aci. Hiz (rad/s)',      '.3f'),
        ('velocity_smoothness', 'Hiz Puruz. (m/s2)',     '.4f'),
        ('path_smoothness',     'Yol Puruz. (m2)',       '.4f'),
        ('net_min_dist',        'Min Engel (m)',         '.3f'),
        ('danger_ratio',        'Tehlike Bolg. (%)',     '.2f'),
        ('cte_rmse',            'CTE RMSE (m)',          '.4f'),
        ('heading_rmse_deg',    'Heading RMSE (deg)',    '.2f'),
    ]

    print("\n" + "=" * 100)
    print("                          PERFORMANS OZET TABLOSU (mean +- std)")
    print("=" * 100)

    sr_header = f"  {'Metrik':<22}"
    for algo in results.keys():
        n = results[algo]['total_time']['n']
        sr_header += f"  {algo + ' (N=' + str(n) + ')':<22}"
    print(sr_header)
    print("-" * 100)

    sr_row = f"  {'Basari Orani':<22}"
    for algo in results.keys():
        rc = run_counts.get(algo, {'n_success': 0, 'n_failed': 0, 'success_rate': 0.0})
        ns = rc['n_success']
        nt = rc['n_success'] + rc['n_failed']
        sr_pct = rc['success_rate']
        cell = f"{ns}/{nt} ({sr_pct:.0f}%)"
        sr_row += f"  {cell:<22}"
    print(sr_row)
    print("-" * 100)

    for key, label, fmt in metrics_display:
        row = f"  {label:<22}"
        for algo, agg in results.items():
            if key not in agg or np.isnan(agg[key]['mean']):
                row += f"  {'N/A':<22}"
                continue
            mean = agg[key]['mean']
            std  = agg[key]['std']
            cell = f"{mean:{fmt}} +- {std:{fmt}}"
            row += f"  {cell:<22}"
        print(row)

    print("=" * 100)

def print_ranking(results):
    metrics_with_dir = [
        ('total_time',          'Sure',               True),
        ('path_length',         'Yol Uzunlugu',       True),
        ('avg_linear_vel',      'Dog. Hiz',           False),
        ('velocity_smoothness', 'Hiz Puruzsuzlugu',   True),
        ('net_min_dist',        'Min Engel',          False),
        ('cte_rmse',            'CTE RMSE',           True),
        ('heading_rmse_deg',    'Heading RMSE',       True),
    ]
    algos = list(results.keys())
    score = {a: 0 for a in algos}
    print("\n  METRIK BAZLI SIRALAMA:")
    print("-" * 80)
    for metric, label, lower_better in metrics_with_dir:
        if metric not in results[algos[0]]:
            continue
        means = {a: results[a][metric]['mean'] for a in algos
                 if not np.isnan(results[a][metric]['mean'])}
        if not means:
            continue
        sorted_algos = sorted(means.keys(), key=lambda a: means[a],
                              reverse=not lower_better)
        for rank, algo in enumerate(sorted_algos, 1):
            score[algo] += rank
        ranking_str = " > ".join([f"{a}({means[a]:.3f})" for a in sorted_algos])
        direction = "(kucuk=iyi)" if lower_better else "(buyuk=iyi)"
        print(f"  {label:<22} {direction:<14} {ranking_str}")

    print("\n  GENEL SKOR (kucuk = iyi):")
    print("-" * 40)
    for algo, s in sorted(score.items(), key=lambda x: x[1]):
        print(f"  {algo}: {s} puan")

def save_summary_csv(results, run_counts, output_dir):
    rows = []
    metrics_keys = [
        ('total_time',          'sure_s'),
        ('path_length',         'yol_m'),
        ('avg_linear_vel',      'dog_hiz_mps'),
        ('avg_angular_vel',     'aci_hiz_radps'),
        ('velocity_smoothness', 'hiz_puruz_mps2'),
        ('path_smoothness',     'yol_puruz_m2'),
        ('net_min_dist',        'min_engel_m'),
        ('danger_ratio',        'tehlike_pct'),
        ('cte_rmse',            'cte_rmse_m'),
        ('heading_rmse_deg',    'heading_rmse_deg'),
    ]
    for algo, agg in results.items():
        rc = run_counts.get(algo, {'n_success': 0, 'n_failed': 0, 'success_rate': 0.0})
        row = {
            'algoritma':         algo,
            'n_success':         rc['n_success'],
            'n_failed':          rc['n_failed'],
            'n_total':           rc['n_success'] + rc['n_failed'],
            'basari_orani_pct':  rc['success_rate'],
            'n_run':             agg['total_time']['n'],
        }
        for key, col in metrics_keys:
            if key not in agg or np.isnan(agg[key]['mean']):
                row[f'{col}_mean'] = np.nan
                row[f'{col}_std']  = np.nan
                continue
            row[f'{col}_mean'] = agg[key]['mean']
            row[f'{col}_std']  = agg[key]['std']
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_out = os.path.join(output_dir, 'summary_table.csv')
    df.to_csv(csv_out, index=False)
    print(f"\n  CSV kaydedildi: {csv_out}")

# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='Static')
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--bags_dir', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--algorithms', nargs='+', default=['DWA', 'MPPI', 'RPP'])
    args = parser.parse_args()

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(script_dir)
    results_dir = args.results_dir or os.path.join(package_root, 'results', args.scenario)
    bags_dir    = args.bags_dir    or os.path.join(package_root, 'bags', args.scenario)
    output_dir  = args.output_dir  or os.path.join(package_root, 'figures', args.scenario)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  CSV klasoru   : {results_dir}")
    print(f"  Bag klasoru   : {bags_dir}")
    print(f"  Cikti klasoru : {output_dir}")
    print(f"  Senaryo       : {args.scenario}")
    print(f"  Algoritmalar  : {args.algorithms}")
    print(f"  rosbag2_py    : {'mevcut' if HAVE_ROSBAG else 'YOK (CTE/Heading hesaplanmaz)'}\n")

    pairs_dict = {}
    run_counts = {}
    for algo in args.algorithms:
        pairs, n_success, n_failed, failed_names = load_csvs_and_bags(
            results_dir, bags_dir, algo, args.scenario)
        n_total = n_success + n_failed
        sr = (n_success / n_total * 100.0) if n_total > 0 else 0.0
        run_counts[algo] = {
            'n_success':    n_success,
            'n_failed':     n_failed,
            'success_rate': sr,
            'failed_names': failed_names,
        }
        if pairs:
            pairs_dict[algo] = pairs
            if n_failed > 0:
                print(f"  {algo}: {n_success}/{n_total} basarili "
                      f"(%{sr:.0f}) — {n_failed} basarisiz run dislandi")
            else:
                print(f"  {algo}: {n_success} run yuklendi")
        else:
            print(f"  {algo}: CSV bulunamadi")

    if not pairs_dict:
        print("\nHIC VERI YUKLENEMEDI.")
        sys.exit(1)

    print("\nMetrikler hesaplaniyor (bag'ler CTE/Heading icin okunuyor)...")
    results = {algo: aggregate_metrics(pairs) for algo, pairs in pairs_dict.items()}

    print("\nGorseller olusturuluyor...")
    plot_path_comparison(pairs_dict, output_dir, scenario=args.scenario)
    plot_velocity_profiles(pairs_dict, output_dir)
    plot_bar_comparison(results, output_dir)

    print_summary_table(results, run_counts)
    print_ranking(results)
    save_summary_csv(results, run_counts, output_dir)

    any_failed = any(rc['n_failed'] > 0 for rc in run_counts.values())
    if any_failed:
        print()
        print("=" * 60)
        print("  UYARI: BASARISIZ RUN'LAR ANALIZDEN DISLANDI")
        print("=" * 60)
        for algo, rc in run_counts.items():
            if rc['n_failed'] > 0:
                print(f"  {algo}: {rc['n_failed']} basarisiz run")
                for name in rc['failed_names']:
                    print(f"    - {name}")
        print("  (Dosyalar '_FAILED' suffix'li olarak isaretlenmistir)")
        print("  Performans metrikleri yalnizca BASARILI run'lardan hesaplandi.")
        print("=" * 60)

    print(f"\nAnaliz tamamlandi. Ciktilar: {output_dir}")

if __name__ == '__main__':
    main()
