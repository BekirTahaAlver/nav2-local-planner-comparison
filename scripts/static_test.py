"""
static_test.py
=================
DWA / MPPI / RPP icin evrensel benchmark test scripti.

CSV cikti:
    my_config/results/{SCENARIO_NAME}/{ALGORITHM_NAME}_{SCENARIO_NAME}_run{N}.csv
    + ayni isimle {..}_run{N}_plan.csv  (referans global yol: /plan)

Metrikler (canli):
    - Yol uzunlugu, sure, ortalama hizlar
    - Path/velocity smoothness (MRPB Eq.5, Eq.6 - Wen et al. 2021)
    - Min engel mesafesi (LiDAR)

Notlar:
    - gt_x, gt_y, gt_theta sutunlari ground truth poz (Gazebo) icin CSV'ye yazilir
    - plan_x, plan_y, plan_theta ayri _plan.csv dosyasina yazilir (CTE referansi)
    - CTE RMSE / Heading Error: bag dosyalarindan OFFLINE hesaplanir
      (compute_cte_heading.py scripti ile)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
import pandas as pd
import numpy as np
import argparse
import os
import math
import time

# ===============================================================
# KULLANICI AYARLARI
# ===============================================================
ALGORITHM_NAME = "RPP"
SCENARIO_NAME  = "Static"

# Robot baslangic (map frame)
START_X, START_Y = 2.5, -2.0
START_THETA      = 1.5708

# Hedef pozisyon (map frame)
GOAL_X, GOAL_Y = -2.2, 2.2
GOAL_THETA     = 3.1416

# Gazebo model adi
ROBOT_MODEL_NAME = "burger"

# TurtleBot3 Burger sabitleri
ROBOT_RADIUS = 0.105

LOG_INTERVAL_SEC = 0.05    # ~20 Hz log
TIMEOUT_SEC      = 120.0

parser = argparse.ArgumentParser()
parser.add_argument('--planner', type=str, default='navfn',
                    choices=['navfn', 'smac'],
                    help='Global planner variant')
args, _ = parser.parse_known_args()
PLANNER = args.planner
# ===============================================================


def yaw_to_quaternion(yaw):
    return {
        'x': 0.0, 'y': 0.0,
        'z': math.sin(yaw / 2.0),
        'w': math.cos(yaw / 2.0),
    }


def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


class ExperimentNode(Node):
    def __init__(self):
        super().__init__('experiment_node')

        # Tum callback'lerin paralel calismasi icin
        self.cb_group = ReentrantCallbackGroup()

        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile_sensor_data, callback_group=self.cb_group)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data, callback_group=self.cb_group)
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.amcl_callback, 10, callback_group=self.cb_group)
        self.gt_sub = self.create_subscription(
            ModelStates, '/gazebo/model_states',
            self.model_states_callback,
            qos_profile_sensor_data,
            callback_group=self.cb_group)
        # Global plan (referans yol) aboneligi
        self.plan_sub = self.create_subscription(
            Path, '/plan', self.plan_callback, 10, callback_group=self.cb_group)

        self.current_odom  = None
        self.min_scan_dist = 99.9
        self.scan_received = False

        # AMCL (sadece ilerleme takibi icin)
        self.map_x = START_X
        self.map_y = START_Y

        # Ground truth (Gazebo)
        self.gt_x = None
        self.gt_y = None
        self.gt_theta = None
        self.gt_received = False

        # Global plan: en uzun (ilk tam) plani sakla -> [(x, y, theta), ...]
        self.best_plan = []

        self.data_log    = []
        self.start_time  = None
        self._last_log_t = 0.0

    # ---------- Callback'ler ----------
    def odom_callback(self, msg):
        self.current_odom = msg

    def scan_callback(self, msg):
        valid = [r for r in msg.ranges if not np.isinf(r) and r > 0.05]
        if valid:
            self.min_scan_dist = min(valid)
            self.scan_received = True

    def amcl_callback(self, msg):
        self.map_x = msg.pose.pose.position.x
        self.map_y = msg.pose.pose.position.y

    def model_states_callback(self, msg):
        try:
            idx = msg.name.index(ROBOT_MODEL_NAME)
        except ValueError:
            return
        pose = msg.pose[idx]
        self.gt_x = pose.position.x
        self.gt_y = pose.position.y
        self.gt_theta = euler_from_quaternion(
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w)
        self.gt_received = True

    def plan_callback(self, msg):
        # Plan robot ilerledikce kisalir; en uzun (ilk tam) plani tutariz.
        # NavFn ara poz yonelimlerini doldurmadigindan theta'yi yol
        # tanjantindan (ardisik nokta yonunden) turetiyoruz.
        if len(msg.poses) > len(self.best_plan):
            pts = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
            plan = []
            for i, (x, y) in enumerate(pts):
                if i < len(pts) - 1:
                    nx, ny = pts[i + 1]
                    yaw = math.atan2(ny - y, nx - x)     # tanjant: sonraki noktaya yon
                else:
                    yaw = plan[-1][2] if plan else 0.0    # son nokta oncekini kopyalar
                plan.append((x, y, yaw))
            self.best_plan = plan

    # ---------- Veri kaydi ----------
    def log_step(self):
        if self.current_odom is None:
            return
        now_ns = self.get_clock().now().nanoseconds
        if self.start_time is None:
            self.start_time = now_ns
        elapsed = (now_ns - self.start_time) / 1e9

        if elapsed - self._last_log_t < LOG_INTERVAL_SEC:
            return
        self._last_log_t = elapsed

        q = self.current_odom.pose.pose.orientation
        yaw_odom = euler_from_quaternion(q.x, q.y, q.z, q.w)

        self.data_log.append({
            'timestamp':     elapsed,
            'pos_x':         self.current_odom.pose.pose.position.x,
            'pos_y':         self.current_odom.pose.pose.position.y,
            'pos_theta':     yaw_odom,
            'map_x':         self.map_x,
            'map_y':         self.map_y,
            'gt_x':          self.gt_x if self.gt_x is not None else float('nan'),
            'gt_y':          self.gt_y if self.gt_y is not None else float('nan'),
            'gt_theta':      self.gt_theta if self.gt_theta is not None else float('nan'),
            'linear_vel':    self.current_odom.twist.twist.linear.x,
            'angular_vel':   self.current_odom.twist.twist.angular.z,
            'min_scan_dist': self.min_scan_dist,
        })

    # ---------- Kaydet & analiz ----------
    def save_and_analyze(self, success):
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        package_root = os.path.dirname(script_dir)
        save_dir = os.path.join(package_root, 'results', PLANNER, SCENARIO_NAME)
        os.makedirs(save_dir, exist_ok=True)

        run_id   = 1
        filename = os.path.join(save_dir,
            f"{ALGORITHM_NAME}_{SCENARIO_NAME}_run{run_id}.csv")
        while os.path.exists(filename):
            run_id += 1
            filename = os.path.join(save_dir,
                f"{ALGORITHM_NAME}_{SCENARIO_NAME}_run{run_id}.csv")

        df = pd.DataFrame(self.data_log)
        df.to_csv(filename, index=False)
        rel_path = os.path.relpath(filename, package_root)
        print(f"\nHam veri kaydedildi: {rel_path}")

        # ---- Global plan (referans yol) kaydi ----
        plan_filename = filename.replace('.csv', '_plan.csv')
        pd.DataFrame(self.best_plan,
                     columns=['plan_x', 'plan_y', 'plan_theta']
                     ).to_csv(plan_filename, index=False)
        plan_rel = os.path.relpath(plan_filename, package_root)
        print(f"Plan kaydedildi ({len(self.best_plan)} nokta): {plan_rel}")
        if not self.best_plan:
            print("  UYARI: /plan bos geldi! QoS uyumsuzlugu olabilir "
                  "(asagidaki transient_local varyantini dene).")

        if df.empty:
            print("Veri toplanamadi!")
            return

        # ---- Metrikler ----
        total_time  = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
        dx          = df['pos_x'].diff()
        dy          = df['pos_y'].diff()
        path_length = float(np.nansum(np.sqrt(dx**2 + dy**2)))

        ddx          = dx.diff()
        ddy          = dy.diff()
        path_smooth  = float(np.nansum(ddx**2 + ddy**2))
        dt           = df['timestamp'].diff().replace(0, np.nan)
        dv           = df['linear_vel'].diff()
        acc          = (dv / dt).abs()
        lin_vel_smooth   = float(acc.mean())
        dw           = df['angular_vel'].diff()
        acc_w          = (dw / dt).abs()
        ang_vel_smooth   = float(acc_w.mean())

        valid_lidar = df[df['min_scan_dist'] < 90.0]
        if not valid_lidar.empty:
            net_min = valid_lidar['min_scan_dist'].min() - ROBOT_RADIUS
        else:
            net_min = float('nan')

        # ---- Rapor ----
        print("\n" + "-" * 50)
        print(f"  PERFORMANS RAPORU [{ALGORITHM_NAME}] Run #{run_id}")
        print("-" * 50)
        print(f"  Sonuc              : {'BASARILI' if success else 'BASARISIZ'}")
        print(f"  Sure               : {total_time:.2f} sn")
        print(f"  Yol Uzunlugu       : {path_length:.3f} m")
        print(f"  Ort. Dog. Hiz      : {df['linear_vel'].abs().mean():.3f} m/s")
        print(f"  Ort. Aci. Hiz      : {df['angular_vel'].abs().mean():.3f} rad/s")
        print(f"  Yol Puruzsuzlugu   : {path_smooth:.4f} m^2    [MRPB Eq.5]")
        print(f"  Lineer Hiz Puruzsuzlugu   : {lin_vel_smooth:.4f} m/s^2  [MRPB Eq.6]")
        print(f"  Acisal Hiz Puruzsuzlugu   : {ang_vel_smooth:.4f} rad/s^2")
        if not math.isnan(net_min):
            print(f"  Min Engel (net)    : {net_min:.3f} m")
        else:
            print(f"  Min Engel (net)    : N/A")
        print("-" * 50)
        print("  Not: CTE RMSE ve Heading Error icin")
        print("       'compute_cte_heading.py' scriptini calistirin.")


def make_pose(nav, x, y, theta, frame='map'):
    q = yaw_to_quaternion(theta)
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.header.stamp = nav.get_clock().now().to_msg()
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.x = q['x']
    msg.pose.orientation.y = q['y']
    msg.pose.orientation.z = q['z']
    msg.pose.orientation.w = q['w']
    return msg


def main():
    rclpy.init()
    nav             = BasicNavigator()
    experiment_node = ExperimentNode()

    # MultiThreadedExecutor — tum callback'leri paralel calistir
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(experiment_node)
    import threading
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    nav.setInitialPose(make_pose(nav, START_X, START_Y, START_THETA))

    print(f"\nNav2 bekleniyor... [{ALGORITHM_NAME} | {SCENARIO_NAME}]")
    nav.waitUntilNav2Active()
    print("Nav2 Hazir.")

    print("Lidar bekleniyor...")
    t0 = time.time()
    while not experiment_node.scan_received:
        time.sleep(0.05)
        if time.time() - t0 > 10.0:
            print("UYARI: 10s'de /scan gelmedi!")
            break
    if experiment_node.scan_received:
        print(f"Lidar aktif. Ilk mesafe: {experiment_node.min_scan_dist:.2f} m")

    print("Ground truth (/gazebo/model_states) bekleniyor...")
    t0 = time.time()
    while not experiment_node.gt_received:
        time.sleep(0.05)
        if time.time() - t0 > 10.0:
            print(f"UYARI: 10s'de ground truth gelmedi!")
            print(f"   Robot adi dogru mu? Su an: '{ROBOT_MODEL_NAME}'")
            break
    if experiment_node.gt_received:
        print(f"Ground truth aktif. GT pos: "
              f"({experiment_node.gt_x:.2f}, {experiment_node.gt_y:.2f})")

    print(f"Navigasyon basliyor. Hedef: "
          f"({GOAL_X}, {GOAL_Y}, theta={math.degrees(GOAL_THETA):.1f}deg)")
    nav.goToPose(make_pose(nav, GOAL_X, GOAL_Y, GOAL_THETA))

    nav_start    = time.time()
    last_print_t = 0.0
    timed_out    = False
    while not nav.isTaskComplete():
        time.sleep(0.02)
        experiment_node.log_step()

        elapsed_wall = time.time() - nav_start

        if elapsed_wall - last_print_t >= 10.0:
            last_print_t = elapsed_wall
            n_samples = len(experiment_node.data_log)
            lidar_str = (f"Lidar: {experiment_node.min_scan_dist:.2f}m"
                         if experiment_node.scan_received else "Lidar: YOK")
            gx = experiment_node.gt_x if experiment_node.gt_x is not None else float('nan')
            gy = experiment_node.gt_y if experiment_node.gt_y is not None else float('nan')
            if not math.isnan(gx):
                dist_to_goal = math.sqrt((gx-GOAL_X)**2 + (gy-GOAL_Y)**2)
            else:
                dist_to_goal = float('nan')
            print(f"  {elapsed_wall:.0f}s | "
                  f"GT: ({gx:.2f}, {gy:.2f}) | "
                  f"Hedefe: {dist_to_goal:.2f}m | "
                  f"{lidar_str} | Ornek: {n_samples}")

        if elapsed_wall >= TIMEOUT_SEC:
            print(f"\nTIMEOUT! {TIMEOUT_SEC:.0f}s doldu. Iptal ediliyor...")
            nav.cancelTask()
            timed_out = True
            break

    if timed_out:
        success = False
        print("Zaman asimi. Toplanan veri kaydediliyor...")
    else:
        result = nav.getResult()
        success = (result == TaskResult.SUCCEEDED)
        print(f"\n{'HEDEFE ULASILDI!' if success else 'NAVIGASYON BASARISIZ!'}")

    experiment_node.save_and_analyze(success)
    executor.shutdown()
    experiment_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
