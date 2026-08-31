"""Path + velocity profile — DWPP style."""
import argparse, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'lines.linewidth': 1.8, 'axes.grid': True,
    'grid.alpha': 0.3, 'grid.linestyle': '--',
})

STYLES = {
    'DWA':  {'color': '#1f77b4', 'ls': '-',  'label': 'DWB'},
    'MPPI': {'color': '#ff7f0e', 'ls': '--', 'label': 'MPPI'},
    'RPP':  {'color': '#2ca02c', 'ls': '-.', 'label': 'RPP'},
}

def load_csv(path):
    """CSV'yi dict[col_name] -> np.array olarak yükle (pandas Series sorununu atlatır)."""
    with open(path) as f:
        header = f.readline().strip().split(',')
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {col: data[:, i] for i, col in enumerate(header)}

def plot_path_comparison(runs_dict, ax, start=(2.5,-2.0), goal=(-2.2,2.2),
                          obstacles=None, arena_size=3.0):
    if obstacles is None:
        obstacles = [(0,0),(1.5,1.5),(-1.5,1.5),(1.5,-1.5),(-1.5,-1.5)]
    for ox,oy in obstacles:
        ax.add_patch(Circle((ox,oy), 0.45, color='gray', alpha=0.6, zorder=2))
    ax.add_patch(Rectangle((-arena_size,-arena_size), 2*arena_size, 2*arena_size,
                            fill=False, edgecolor='black', linewidth=1.5, zorder=1))
    for algo, runs in runs_dict.items():
        s = STYLES[algo]
        for i, d in enumerate(runs):
            ax.plot(d['map_x'], d['map_y'], color=s['color'], alpha=0.30,
                    linestyle=s['ls'], linewidth=1.0, zorder=3)
        if runs:
            ax.plot(runs[0]['map_x'], runs[0]['map_y'], color=s['color'],
                    linestyle=s['ls'], linewidth=2.2, label=s['label'], zorder=4)
    ax.plot(start[0], start[1], 'o', markersize=12, color='blue',
            markeredgecolor='black', label='Start', zorder=5)
    ax.plot(goal[0], goal[1], '*', markersize=16, color='red',
            markeredgecolor='black', label='Goal', zorder=5)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_aspect('equal')
    ax.set_xlim(-arena_size-0.2, arena_size+0.2)
    ax.set_ylim(-arena_size-0.2, arena_size+0.2)
    ax.legend(loc='best', framealpha=0.95)
    ax.set_title('(a) Path comparison')

def plot_linear_velocity(runs_dict, ax, v_max=0.26):
    for algo, runs in runs_dict.items():
        s = STYLES[algo]
        for i, d in enumerate(runs):
            t = d['timestamp'] - d['timestamp'][0]
            label = s['label'] if i == 0 else None
            ax.plot(t, d['linear_vel'], color=s['color'], linestyle=s['ls'],
                    alpha=0.7 if i > 0 else 1.0, linewidth=1.5, label=label)
    ax.axhline(y=v_max, color='red', linestyle=':', linewidth=1.2,
               alpha=0.8, label=f'$v_{{max}}$ = {v_max} m/s')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Linear velocity $v$ [m/s]')
    ax.set_ylim(-0.05, v_max * 1.2)
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_title('(b) Linear velocity profile')

def plot_angular_velocity(runs_dict, ax, w_max=1.0):
    for algo, runs in runs_dict.items():
        s = STYLES[algo]
        for i, d in enumerate(runs):
            t = d['timestamp'] - d['timestamp'][0]
            label = s['label'] if i == 0 else None
            ax.plot(t, d['angular_vel'], color=s['color'], linestyle=s['ls'],
                    alpha=0.7 if i > 0 else 1.0, linewidth=1.5, label=label)
    ax.axhline(y=w_max, color='red', linestyle=':', linewidth=1.2, alpha=0.8)
    ax.axhline(y=-w_max, color='red', linestyle=':', linewidth=1.2, alpha=0.8,
               label=f'$\\pm\\omega_{{max}}$ = $\\pm${w_max} rad/s')
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Angular velocity $\\omega$ [rad/s]')
    ax.set_ylim(-w_max * 1.2, w_max * 1.2)
    ax.legend(loc='lower right', framealpha=0.95)
    ax.set_title('(c) Angular velocity profile')

def main(results_dir, output_prefix, scenario, layout):
    print(f"Reading from: {results_dir}")
    runs_dict = {}
    for algo in ['DWA','MPPI','RPP']:
        pattern = os.path.join(results_dir, f'{algo}_{scenario}_run*.csv')
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  WARNING {algo}: no files matching {pattern}")
            continue
        runs = [load_csv(f) for f in files]
        print(f"  {algo}: {len(runs)} runs loaded")
        runs_dict[algo] = runs
    if not runs_dict:
        print("ERROR: No data!")
        return
    if layout == 'vertical':
        fig = plt.figure(figsize=(8, 14))
        ax1 = fig.add_subplot(311); ax2 = fig.add_subplot(312); ax3 = fig.add_subplot(313)
    else:
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1], hspace=0.30, wspace=0.25)
        ax1 = fig.add_subplot(gs[0,:]); ax2 = fig.add_subplot(gs[1,0]); ax3 = fig.add_subplot(gs[1,1])
    plot_path_comparison(runs_dict, ax1)
    plot_linear_velocity(runs_dict, ax2)
    plot_angular_velocity(runs_dict, ax3)
    out_dir = os.path.dirname(output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{output_prefix}.pdf')
    plt.savefig(f'{output_prefix}.png')
    print(f"SAVED: {output_prefix}.pdf, {output_prefix}.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results/Static')
    parser.add_argument('--output', default='figures/figure_static_benchmark')
    parser.add_argument('--scenario', default='Static')
    parser.add_argument('--layout', default='mixed', choices=['vertical','mixed'])
    args = parser.parse_args()
    main(args.results_dir, args.output, args.scenario, args.layout)
