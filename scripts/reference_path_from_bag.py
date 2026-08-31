#!/usr/bin/env python3
"""
extract_reference_from_bag.py
-----------------------------
Bir senaryonun referans yolunu, o senaryonun bag'lerinden BIRINDEN cikarir
(analyze_results.py'daki extract_first_plan mantiginin aynisi): ilk /plan
mesaji (>=2 poz) referans kabul edilir. Referans senaryo basina ozdes oldugu
icin, /plan iceren TEK bir bag yeterli.

NavFn ara-poz yonelimini doldurmadiginda theta tanjanttan turetilir.

Kullanim:
    python3 extract_reference_from_bag.py --scenario Static
    python3 extract_reference_from_bag.py --scenario Narrow_U --bags_dir ../bags/Narrow_U

Cikti: results/reference_paths/{SCENARIO}_plan.csv  (plan_x, plan_y, plan_theta)
"""

import os
import glob
import math
import argparse
import numpy as np
import pandas as pd

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    HAVE_ROSBAG = True
except ImportError:
    HAVE_ROSBAG = False


def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def extract_first_plan(bag_path):
    """analyze_results.py ile ayni: bag'deki ilk /plan mesajini referans olarak dondur."""
    if not HAVE_ROSBAG or not os.path.isdir(bag_path):
        return None
    try:
        storage_opts = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_opts = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr')
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
            # NavFn tum thetalari ~0 birakirsa tanjanttan turet
            if np.std(path[:, 2]) < 1e-6:
                dx, dy = np.diff(path[:, 0]), np.diff(path[:, 1])
                derived = np.append(np.arctan2(dy, dx), 0.0)
                if len(derived) > 1:
                    derived[-1] = derived[-2]
                path[:, 2] = derived
            return path
    except Exception as e:
        print(f"    bag okuma hatasi: {e}")
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True)
    ap.add_argument('--bags_dir', default=None)
    args = ap.parse_args()

    if not HAVE_ROSBAG:
        print("[HATA] rosbag2_py yok. ROS 2 ortamini source'la.")
        return

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(script_dir)
    bags_dir = args.bags_dir or os.path.join(package_root, 'bags', args.scenario)
    out_dir  = os.path.join(package_root, 'results', 'reference_paths')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(bags_dir):
        print(f"[HATA] bag klasoru yok: {bags_dir}")
        return

    # senaryonun bag'lerini dene; /plan iceren ilkinden referansi al
    bag_dirs = sorted(d for d in glob.glob(os.path.join(bags_dir, '*'))
                      if os.path.isdir(d))
    ref = None
    used = None
    for b in bag_dirs:
        ref = extract_first_plan(b)
        if ref is not None:
            used = os.path.basename(b)
            break

    if ref is None:
        print(f"[HATA] {args.scenario}: hicbir bag'de /plan bulunamadi.\n"
              f"       Bu senaryo icin capture_reference_path.py (getPath) kullan.")
        return

    out = os.path.join(out_dir, f"{args.scenario}_plan.csv")
    pd.DataFrame(ref, columns=['plan_x', 'plan_y', 'plan_theta']).to_csv(out, index=False)
    print(f"Referans yol '{used}' bag'inden alindi ({len(ref)} nokta).")
    print(f"Kaydedildi -> results/reference_paths/{args.scenario}_plan.csv")


if __name__ == '__main__':
    main()
