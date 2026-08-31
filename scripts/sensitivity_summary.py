#!/usr/bin/env python3
"""
sensitivity_summary.py
----------------------
Tek-parametre duyarlilik taramasi (or. DWA sim_time = 1.0 / 2.0 / 3.0) icin
ozet tablo uretir. analyze_results.py ile ayni metrik mantigini kullanir.

Beklenen dosya adi formati:
    {PARAM}_{PLANNER}_{SCENARIO}_run{N}[_FAILED_<reason>].csv
Ornekler:
    p1v1_DWA_Static_run1.csv          -> param=p1v1, planner=DWA, scenario=Static
    p1v1_MPPI_Static_run1.csv         -> planner=MPPI
    p1v2_DWA_Narrow_U_run5.csv        -> scenario ic alt cizgi icerebilir
    p1v3_DWA_Dynamic_run4_FAILED_stall.csv  -> basarisiz run

Gruplama:
    Her (PLANNER, PARAM) icin bir ozet tablo; satirlar = senaryolar.
    Basarisiz run'lar (FAILED) metrik ortalamasindan CIKARILIR ama basari
    oranina TOPLAM olarak sayilir.

CSV kolon duzeni (12 kolon):
    timestamp, pos_x, pos_y, pos_theta, map_x, map_y,
    gt_x, gt_y, gt_theta, linear_vel, angular_vel, min_scan_dist

Cikti:
    results/sensitivity_summary/summary_{PLANNER}_{PARAM}.csv  (her varyant)
    results/sensitivity_summary/summary_all.csv                (birlesik)
"""

import os
import re
import glob
import collections
import numpy as np
import pandas as pd

# ============================ CONFIG ============================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR  = os.path.join(PACKAGE_ROOT, "results")     # recursive taranir
OUT_DIR      = os.path.join(RESULTS_DIR, "sensitivity_summary")

ROBOT_RADIUS = 0.105        # TB3 Burger footprint yaricapi [m]
NET_MIN_DIST = True         # min_scan_dist'ten ROBOT_RADIUS dusulsun mu? (net mesafe)
PATH_SOURCE  = "gt"         # 'gt' (ground truth) veya 'pos' (AMCL kestirimi)
FAILED_TAG   = "FAILED"     # basarisiz run'i isaretleyen alt-dize (buyuk/kucuk harf duyarsiz)
EXPECTED_RUNS = 10          # hucre basina beklenen toplam test (sadece uyari; None = kapali)

# Senaryolarin tablodaki mantiksal sirasi (alfabetik degil). Listede olmayan
# senaryolar sona eklenir.
SCENARIO_ORDER = ["Static", "Dynamic", "Narrow_U", "Narrow_Z"]

# Etiketler PLANLAYICIYA gore degisir: 'p1' DWA'da sim_time iken MPPI'de
# time_steps'i temsil eder. Bu yuzden {planner: {param: label}} yapisi.
# Eslesme yoksa ham etiket (or. 'p1v1') kullanilir.
LABEL_MAP = {
    "DWA": {
        "p1v1": "sim_time=1.0",
        "p1v2": "sim_time=2.0",
        "p1v3": "sim_time=3.0",
        "p2v1": "PathAlign.scale=10.0",
        "p2v2": "PathAlign.scale=32.0",
        "p2v3": "PathAlign.scale=64.0",
    },
    "MPPI": {
        # p1 = time_steps (devir-teslim plani: 30/56/80) -- TEST ETTIGIN
        # gercek degerlerle teyit et/degistir.
        "p1v1": "time_steps=20",
        "p1v2": "time_steps=40",
        "p1v3": "time_steps=60",
        # p2 = CostCritic.weight -- gercek degerlerinle doldur.
        "p2v1": "CostCritic.weight=1.5",
        "p2v2": "CostCritic.weight=3.81",
        "p2v3": "CostCritic.weight=8.0",
    },
    # RPP eklendiginde ornek:
    # "RPP": {"p1v1": "lookahead_dist=0.3", "p1v2": "lookahead_dist=0.6",
    #         "p1v3": "lookahead_dist=0.9"},
    "RPP": {
    	"p1v1": "lookahead_time=1.0",
    	"p1v2": "lookahead_time=2.0",
    	"p1v3": "lookahead_time=3.0",
    }
}

# {PARAM}_{PLANNER}_{SCENARIO}_run{N}[_FAILED_...] ; PARAM = p<sayi>v<sayi>.
# scenario alt cizgi icerebilir (non-greedy). Yalniz bu desene UYAN dosyalar
# sayilir; yedekler (backup_..., navfn_..., ..._run1_backup, "(copy)" vb.)
# ve p ile baslamayan hicbir dosya dahil EDILMEZ.
NAME_RE = re.compile(r"^(?P<param>p\d+v\d+)_(?P<planner>[^_]+)_"
                     r"(?P<scenario>.+?)_run(?P<run>\d+)(?:_FAILED.*)?$")
# ===============================================================


def parse_name(path):
    """(param, planner, scenario, failed) uretir. Cozulemezse None."""
    stem = os.path.basename(path)
    if stem.endswith(".csv"):
        stem = stem[:-4]
    m = NAME_RE.match(stem)
    if not m:
        return None
    failed = FAILED_TAG.lower() in stem.lower()
    # '_FAILED' senaryo adina yanlislikla girmis olabilir (or. Narrow_Z_FAILED).
    # Temizle ki basarisiz kosular gercek hucrenin PAYDASINA sayilsin ve
    # success_rate dogru ciksin (ayri hayalet senaryo olusmaz).
    scenario = re.sub(r"_FAILED.*$", "", m["scenario"], flags=re.IGNORECASE)
    return m["param"], m["planner"], scenario, failed


def label_for(planner, param):
    """Planlayici-bilincli okunabilir etiket. Eslesme yoksa ham param."""
    return LABEL_MAP.get(planner, {}).get(param, param)


def completion_time(df):
    """Toplam sure = son timestamp - ilk timestamp [s]."""
    return float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0])


def path_length(df):
    """Ardisik konumlar arasi Oklid mesafelerinin toplami [m]."""
    src = PATH_SOURCE
    if f"{src}_x" not in df.columns or f"{src}_y" not in df.columns:
        src = "pos"   # eski 9-kolonlu CSV'lerde gt yoksa pos'a dus
    dx = df[f"{src}_x"].diff()
    dy = df[f"{src}_y"].diff()
    return float(np.sqrt(dx**2 + dy**2).sum())


def min_obstacle_distance(df):
    """Run boyunca en yakin engel mesafesi; NET_MIN_DIST ise yaricap dusulur [m]."""
    raw = float(df["min_scan_dist"].min())
    return raw - ROBOT_RADIUS if NET_MIN_DIST else raw


def lin_vel_smoothness(df):
    """Ortalama mutlak lineer ivme [m/s^2] (static_test.py ile ayni formul)."""
    dt = df["timestamp"].diff().replace(0, np.nan)
    return float((df["linear_vel"].diff() / dt).abs().mean())


def ang_vel_smoothness(df):
    """Ortalama mutlak acisal ivme [rad/s^2] (static_test.py ile ayni formul)."""
    dt = df["timestamp"].diff().replace(0, np.nan)
    return float((df["angular_vel"].diff() / dt).abs().mean())


def main():
    files = glob.glob(os.path.join(RESULTS_DIR, "**", "*.csv"), recursive=True)

    total   = collections.Counter()   # (planner, param, scenario) -> toplam
    success = collections.Counter()   # (planner, param, scenario) -> basarili
    metric_rows = []
    skipped = []

    for f in files:
        if os.path.abspath(OUT_DIR) in os.path.abspath(f):
            continue
        parsed = parse_name(f)
        if parsed is None:
            skipped.append(os.path.basename(f))
            continue
        param, planner, scenario, failed = parsed
        key = (planner, param, scenario)
        total[key] += 1
        if failed:
            continue
        success[key] += 1
        try:
            df = pd.read_csv(f)
            metric_rows.append({
                "planner": planner, "param": param, "scenario": scenario,
                "time_s": completion_time(df),
                "path_m": path_length(df),
                "min_obs_m": min_obstacle_distance(df),
                "lin_vs": lin_vel_smoothness(df),
                "ang_vs": ang_vel_smoothness(df),
            })
        except Exception as e:
            print(f"[UYARI] okunamadi: {os.path.basename(f)} ({e})")

    if not total:
        print(f"[HATA] '{RESULTS_DIR}' altinda uygun CSV bulunamadi.")
        return

    if metric_rows:
        mdf = pd.DataFrame(metric_rows)
        # Ortalama + ornek standart sapmasi (ddof=1). n=1 ise std NaN doner.
        agg = (mdf.groupby(["planner", "param", "scenario"])
                  .agg(time_s=("time_s", "mean"),
                       time_s_std=("time_s", "std"),
                       path_m=("path_m", "mean"),
                       path_m_std=("path_m", "std"),
                       min_obs_m=("min_obs_m", "mean"),
                       min_obs_m_std=("min_obs_m", "std"),
                       lin_vs=("lin_vs", "mean"),
                       lin_vs_std=("lin_vs", "std"),
                       ang_vs=("ang_vs", "mean"),
                       ang_vs_std=("ang_vs", "std"))
                  .reset_index())
    else:
        agg = pd.DataFrame(columns=["planner", "param", "scenario",
                                    "time_s", "time_s_std", "path_m", "path_m_std",
                                    "min_obs_m", "min_obs_m_std",
                                    "lin_vs", "lin_vs_std", "ang_vs", "ang_vs_std"])

    cells = pd.DataFrame(list(total.keys()),
                         columns=["planner", "param", "scenario"])
    cells["n_total"]   = cells.apply(lambda r: total[(r.planner, r.param, r.scenario)], axis=1)
    cells["n_success"] = cells.apply(lambda r: success[(r.planner, r.param, r.scenario)], axis=1)
    cells["success_rate"] = cells["n_success"] / cells["n_total"]

    summary = cells.merge(agg, on=["planner", "param", "scenario"], how="left")

    # Senaryolari mantiksal sirada tut (alfabetik degil): SCENARIO_ORDER once,
    # listede olmayan senaryolar sona.
    scen_cats = SCENARIO_ORDER + [s for s in summary["scenario"].unique()
                                  if s not in SCENARIO_ORDER]
    summary["scenario"] = pd.Categorical(summary["scenario"],
                                         categories=scen_cats, ordered=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_tables = []

    # Her (planner, param) icin bir tablo (senaryolar mantiksal sirada)
    def fmt_pm(mean, std, dec):
        """'mean ± std' bicimi. mean yoksa bos; std yoksa (n=1) 0 kabul."""
        if pd.isna(mean):
            return ""
        s = 0.0 if pd.isna(std) else std
        return f"{mean:.{dec}f} \u00b1 {s:.{dec}f}"

    for (planner, param), grp in summary.groupby(["planner", "param"], observed=True):
        grp = grp.sort_values("scenario").reset_index(drop=True)
        label = label_for(planner, param)

        disp = pd.DataFrame({
            "Scenario":    grp["scenario"],
            "SuccessRate": grp["success_rate"].round(2),
            "Runs":        grp["n_success"].astype(str) + "/" + grp["n_total"].astype(str),
            "Time_s":      [fmt_pm(m, s, 2) for m, s in zip(grp["time_s"], grp["time_s_std"])],
            "PathLen_m":   [fmt_pm(m, s, 3) for m, s in zip(grp["path_m"], grp["path_m_std"])],
            "MinObs_m":    [fmt_pm(m, s, 3) for m, s in zip(grp["min_obs_m"], grp["min_obs_m_std"])],
            "LinVS":       [fmt_pm(m, s, 4) for m, s in zip(grp["lin_vs"], grp["lin_vs_std"])],
            "AngVS":       [fmt_pm(m, s, 4) for m, s in zip(grp["ang_vs"], grp["ang_vs_std"])],
        })

        print("\n" + "=" * 60)
        print(f"  {planner}  |  {label}  ({param})")
        print("=" * 60)
        print(disp.to_string(index=False))

        out_csv = os.path.join(OUT_DIR, f"summary_{planner}_{param}.csv")
        disp.to_csv(out_csv, index=False)
        print(f"  -> results/sensitivity_summary/summary_{planner}_{param}.csv")

        grp = grp.copy()
        grp["label"] = label
        all_tables.append(grp)

        if EXPECTED_RUNS is not None:
            for _, r in grp.iterrows():
                if r["n_total"] != EXPECTED_RUNS:
                    print(f"  [UYARI] {label} / {r['scenario']}: "
                          f"{int(r['n_total'])} dosya bulundu, {EXPECTED_RUNS} bekleniyordu.")

    combined = pd.concat(all_tables, ignore_index=True)
    combined["scenario"] = combined["scenario"].astype(str)   # kategoriyi metne cevir

    # --- Siralama ---
    # Parametre-oncelikli: once parametre ailesi (p1 tamamen, sonra p2), her
    # aile icinde senaryo (mantiksal sira), her senaryoda deger (v1<v2<v3).
    scen_idx = {s: i for i, s in enumerate(SCENARIO_ORDER)}
    combined["_pfam"] = combined["param"].str.extract(r"^(p\d+)", expand=False)
    combined["_pval"] = combined["param"].str.extract(r"v(\d+)$", expand=False).astype(int)
    combined["_scen"] = combined["scenario"].map(lambda s: scen_idx.get(s, len(SCENARIO_ORDER)))
    combined = combined.sort_values(
        ["planner", "_pfam", "_scen", "_pval"]).reset_index(drop=True)

    # label <-> scenario yer degistirdi
    cols = ["planner", "param", "label", "scenario", "success_rate",
            "n_success", "n_total",
            "time_s", "time_s_std", "path_m", "path_m_std",
            "min_obs_m", "min_obs_m_std",
            "lin_vs", "lin_vs_std", "ang_vs", "ang_vs_std"]
    combined = combined[cols].round({"success_rate": 3,
                                     "time_s": 2, "time_s_std": 2,
                                     "path_m": 3, "path_m_std": 3,
                                     "min_obs_m": 3, "min_obs_m_std": 3,
                                     "lin_vs": 4, "lin_vs_std": 4,
                                     "ang_vs": 4, "ang_vs_std": 4})
    combined.to_csv(os.path.join(OUT_DIR, "summary_all.csv"), index=False)
    print(f"\n  Birlesik tablo -> results/sensitivity_summary/summary_all.csv")

    # --- Basarisizlik dokumu (yalniz >=1 basarisiz kosu olan hucreler) ---
    brk = combined[combined["n_success"] < combined["n_total"]].copy()
    if not brk.empty:
        brk["n_failed"] = brk["n_total"] - brk["n_success"]
        brk = brk[["planner", "param", "label", "scenario",
                   "n_failed", "n_total", "success_rate"]]
        brk.to_csv(os.path.join(OUT_DIR, "summary_failures.csv"), index=False)
        print("\n" + "=" * 60)
        print("  BASARISIZLIK DOKUMU (>=1 basarisiz kosu olan hucreler)")
        print("=" * 60)
        print(brk.to_string(index=False))
        print("  -> results/sensitivity_summary/summary_failures.csv")

    if skipped:
        print(f"\n  [BILGI] pNvM desenine uymayan {len(skipped)} dosya atlandi "
              f"(yedek/eski format sayilmadi): "
              f"{', '.join(skipped[:5])}{' ...' if len(skipped) > 5 else ''}")


if __name__ == "__main__":
    main()
