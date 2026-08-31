#!/usr/bin/env python3
"""
compute_cte_heading.py  (pNvM duzeni)
=====================================
pNvM duyarlilik CSV'lerindeki ground-truth pozlari ile senaryonun referans
yolunu kullanarak her run icin CTE RMSE (dik uzaklik) ve Heading Error RMSE
hesaplar; (planner, param, scenario) hucresi bazinda ozetler.

Verimlilik: referans yol senaryo basina OZDES (ayni global planlayici/costmap/
start/goal). Bu yuzden her run icin ayri bag okumak yerine senaryo basina BIR
kez referans alinir:
    1) results/reference_paths/{SCENARIO}_plan.csv varsa oradan (ROS gerekmez)
    2) yoksa bags/{SCENARIO}/ icindeki ilk /plan'li bag'den (rosbag2_py gerekir)

Kullanim:
    python3 compute_cte_heading.py                       # tum senaryolar
    python3 compute_cte_heading.py --scenario Narrow_U

Cikti:
    results/sensitivity_summary/cte_per_run.csv   (her run: planner,param,scenario,run,cte,heading)
    Terminal: (planner, param, scenario) bazli mean +- std ozet
"""

import argparse
import glob
import math
import os
import re
import numpy as np
import pandas as pd

# rosbag2_py yalnizca bag'den referans alinacaksa gerekir (soft import)
try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    HAVE_ROSBAG = True
except ImportError:
    HAVE_ROSBAG = False

SCENARIOS_DEFAULT = ["Static", "Dynamic", "Narrow_U", "Narrow_Z"]
NAME_RE = re.compile(r"^(?P<param>p\d+v\d+)_(?P<planner>[^_]+)_"
                     r"(?P<scenario>.+?)_run(?P<run>\d+)(?:_FAILED.*)?$")


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


def extract_first_plan(bag_path):
    """Bag'deki ilk /plan mesajini (M,3) [x,y,theta] dondurur; theta ~0 ise tanjanttan."""
    if not HAVE_ROSBAG or not os.path.isdir(bag_path):
        return None
    storage_opts = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_opts = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr', output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_opts, converter_opts)
    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
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
            q = ps.pose.orientation
            path.append((ps.pose.position.x, ps.pose.position.y,
                         euler_from_quaternion(q.x, q.y, q.z, q.w)))
        path = np.array(path)
        if np.std(path[:, 2]) < 1e-6:   # NavFn ara poz yonelimini doldurmaz
            dx, dy = np.diff(path[:, 0]), np.diff(path[:, 1])
            derived = np.append(np.arctan2(dy, dx), 0.0)
            if len(derived) > 1:
                derived[-1] = derived[-2]
            path[:, 2] = derived
        return path
    return None


def get_reference(scenario, pkg_root):
    """Senaryo referansini bir kez getir: once reference_paths, sonra bags."""
    ref_csv = os.path.join(pkg_root, 'results', 'reference_paths',
                           f'{scenario}_plan.csv')
    if os.path.exists(ref_csv):
        d = pd.read_csv(ref_csv)
        return d[['plan_x', 'plan_y', 'plan_theta']].to_numpy(), f"reference_paths/{scenario}_plan.csv"
    bags_dir = os.path.join(pkg_root, 'bags', scenario)
    if os.path.isdir(bags_dir):
        for b in sorted(glob.glob(os.path.join(bags_dir, '*'))):
            if os.path.isdir(b):
                ref = extract_first_plan(b)
                if ref is not None:
                    return ref, f"bags/{scenario}/{os.path.basename(b)}"
    return None, None


def compute_metrics(csv_path, reference_path):
    """CSV gt pozlari + referans -> CTE/Heading RMSE (compute_cte_heading ile ayni)."""
    df = pd.read_csv(csv_path)
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
    P1, P2 = ref_xy[:-1], ref_xy[1:]
    segment_vecs = P2 - P1
    segment_lengths_sq = np.sum(segment_vecs ** 2, axis=1)

    for i, r_pos in enumerate(gt_pts):
        pnt_vecs = r_pos - P1
        t = np.clip(np.sum(pnt_vecs * segment_vecs, axis=1) /
                    (segment_lengths_sq + 1e-10), 0.0, 1.0)
        projections = P1 + t[:, np.newaxis] * segment_vecs
        distances = np.linalg.norm(r_pos - projections, axis=1)
        min_idx = np.argmin(distances)
        cte_vals[i] = distances[min_idx]
        heading_vals[i] = abs(angle_diff(gt_yaws[i], ref_theta[min_idx]))

    return {
        'n_samples':    int(len(gt_pts)),
        'cte_rmse':     float(np.sqrt(np.mean(cte_vals ** 2))),
        'cte_mean':     float(np.mean(cte_vals)),
        'cte_max':      float(np.max(cte_vals)),
        'heading_rmse': float(np.sqrt(np.mean(heading_vals ** 2))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenario', default=None, help='Tek senaryo (yoksa hepsi)')
    ap.add_argument('--package-root',
                    default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args()

    pkg_root = args.package_root
    scenarios = [args.scenario] if args.scenario else SCENARIOS_DEFAULT

    rows = []
    for scenario in scenarios:
        results_dir = os.path.join(pkg_root, 'results', scenario)
        if not os.path.isdir(results_dir):
            print(f"[{scenario}] results klasoru yok, atlaniyor.")
            continue

        ref, ref_src = get_reference(scenario, pkg_root)
        if ref is None:
            print(f"[{scenario}] referans alinamadi "
                  f"(reference_paths/{scenario}_plan.csv yok, bag'de /plan yok).")
            continue
        print(f"\n[{scenario}] referans: {ref_src}  ({len(ref)} nokta)")

        files = [f for f in sorted(glob.glob(os.path.join(results_dir, "*.csv")))
                 if NAME_RE.match(os.path.basename(f)[:-4])
                 and 'FAILED' not in os.path.basename(f).upper()]
        for f in files:
            m0 = NAME_RE.match(os.path.basename(f)[:-4])
            m = compute_metrics(f, ref)
            if m is None:
                continue
            rows.append({
                'planner':  m0['planner'], 'param': m0['param'],
                'scenario': scenario, 'run': f"run{m0['run']}",
                'cte_rmse_m':       m['cte_rmse'],
                'cte_mean_m':       m['cte_mean'],
                'cte_max_m':        m['cte_max'],
                'heading_rmse_deg': math.degrees(m['heading_rmse']),
                'n_samples':        m['n_samples'],
            })
        print(f"[{scenario}] {sum(1 for r in rows if r['scenario']==scenario)} run islendi.")

    if not rows:
        print("\nHic run islenemedi.")
        return

    df = pd.DataFrame(rows)
    out_dir = os.path.join(pkg_root, 'results', 'sensitivity_summary')
    os.makedirs(out_dir, exist_ok=True)
    per_run = os.path.join(out_dir, 'cte_per_run.csv')
    df.to_csv(per_run, index=False)

    # (planner, param, scenario) bazli ozet
    print("\n" + "=" * 78)
    print("  CTE RMSE OZET  (planner | param | scenario : mean +- std) [m]")
    print("=" * 78)
    g = (df.groupby(['planner', 'param', 'scenario'])['cte_rmse_m']
           .agg(['mean', 'std', 'count']).reset_index())
    for _, r in g.iterrows():
        s = 0.0 if pd.isna(r['std']) else r['std']
        print(f"  {r['planner']:<5} {r['param']:<5} {r['scenario']:<9} : "
              f"{r['mean']:.4f} \u00b1 {s:.4f}  (n={int(r['count'])})")
    print(f"\nHer-run CTE -> results/sensitivity_summary/cte_per_run.csv")


if __name__ == '__main__':
    main()
