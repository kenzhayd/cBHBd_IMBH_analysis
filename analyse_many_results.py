"""
analyse_many_results.py
Description:
Aggregates results from multiple configuration folders within a parent directory.

Plots over different initial conditions:
  1. Probability p(M > M) of cluster retaining an intermediate-mass black hole greater than mass M.
  2. Histograms of all merger product masses.
  3. Normalized histogram of final IMBH masses (mIMBH_final).
  4. Normalized histogram of the retained largest merger mass BH.
  5. Diagnostic for maximum merger mass BH vs mIMBH_final.

Usage:
python analyse_many_results.py --parent_dir <PATH_TO_PARENT_FOLDER>
"""
import csv
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
COLOURS = ["#000000", "#E69F00", "#56B4E8", "#009E73",
           "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

CONFIG_KEYS = [
    'M0', 'rhoh0', 'FeH', 'rg', 'tend',
    'M_vir', 'c_halo',
    'rantala_imbh_seed', 'chattopadhyay_seed', 'galpy_potential', 'clumps_seed', 'n_clumps'
]

PARAM_LABEL = {
    'M0':                 lambda v: f"M0_{v:.0e}",
    'rhoh0':              lambda v: f"rhoh0_{v:.2e}",
    'FeH':                lambda v: f"FeH_{v:.1f}",
    'rg':                 lambda v: f"rg_{v:.2f}",
    'tend':               lambda v: f"tend_{v:.2f}",
    'M_vir':              lambda v: f"Mvir_{v:.1e}",
    'c_halo':             lambda v: f"cHalo_{v:.2f}",
    'rantala_imbh_seed':  lambda v: "RantalaSeed" if v else "StdSeed",
    'chattopadhyay_seed': lambda v: "ChattSeed" if v else "StdSeed",
    'galpy_potential':    lambda v: "GalpyPot" if v else "StdPot",
    'use_clumps':         lambda v: "Clumps" if v else "NoClumps", 
    'n_clumps':           lambda v: f"Nclumps_{v}",                
}

def _sci(v, prec=2):
    """4.17e+05 -> 4.17\\times10^{5}"""
    mant, _, exp = f"{v:.{prec}e}".partition('e')
    return rf"{mant}\times10^{{{int(exp)}}}"

PARAM_LABEL_TEX = {
    'M0':                 lambda v: rf"$M_0 = {_sci(v, 0)}$",
    'rhoh0':              lambda v: rf"$\rho_{{\mathrm{{h0}}}} = {_sci(v)}$",
    'FeH':                lambda v: rf"$[\mathrm{{Fe/H}}] = {v:.1f}$",
    'rg':                 lambda v: rf"$r_{{\mathrm{{g}}}} = {v:.2f}$",
    'tend':               lambda v: rf"$t_{{\mathrm{{end}}}} = {v:.0f}\,\mathrm{{Myr}}$",
    'M_vir':              lambda v: rf"$M_{{\mathrm{{vir}}}} = {_sci(v, 0)}\,M_{{\odot}}$",
    'c_halo':             lambda v: rf"$c_{{\mathrm{{halo}}}} = {v:.0f}$",
    'rantala_imbh_seed':  lambda v: "Rantala seed" if v else "Std seed",
    'chattopadhyay_seed': lambda v: "Chattopadhyay seed" if v else "Std IMF",
    'galpy_potential':    lambda v: "Galpy potential" if v else "Std potential",
    'use_clumps':         lambda v: "Clump Seeds" if v else "Std Seeds", 
    'n_clumps':           lambda v: rf"$N_{{clumps}} = {v}$",            

}

def get_config_name(config_dict):
    """To keep naming consistent with single_model.py"""
    return "_".join(PARAM_LABEL[k](config_dict[k]) for k in CONFIG_KEYS)

def config_label(config, varying_params, n_runs):
    """LaTeX legend label + number of runs"""
    if varying_params and config is not None:
        base = ", ".join(PARAM_LABEL_TEX[k](config[k]) for k in varying_params)
    else:
        base = "not_a_useful_name"

    return base + rf" ($N={n_runs}$)"

# Retention criteria
def is_retained_merger(m):
    """True if this merger remnant is retained during the merger. Maybe ejections can occcur still in 3-body encounters."""
    mtype = str(m.get('merger_type', '')).lower()
    if 'ejected' in mtype:
        return False
    v_kick = m.get('v_kick', None)
    v_esc = m.get('v_esc', None)
    if v_kick is not None and v_esc is not None:
        if float(v_kick) >= float(v_esc):
            return False
    return True


def load_config_data(config_dir):
    """
    Reads all run_*/data.json under a config folder.
    Returns (imbh_masses, max_merger_masses, all_merger_masses, config)
    """
    imbh_masses = []
    max_merger_masses = []
    all_merger_masses = []
    config = None

    data_files = sorted(config_dir.glob("run_*/data.json"))
    if not data_files:
        data_files = sorted(p for p in config_dir.glob("**/data.json")
                            if "summary" not in p.parts)

    for f_path in data_files:
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            continue

        stats = data.get('stats', {})
        if config is None and all(k in stats for k in CONFIG_KEYS):
            config = {k: stats[k] for k in CONFIG_KEYS}

        mergers = data.get('mergers', [])

        # 1. Most massive BH at tend = mIMBH_final)
        m_imbh = stats.get('mIMBH_final', None)
        m_imbh = float(m_imbh) if (m_imbh is not None and not np.isnan(m_imbh)) else np.nan


        # 2. Retained maximum merger product masses
        retained_masses = [float(m['m_rem']) for m in mergers 
                           if m.get('m_rem') is not None and is_retained_merger(m)]
        m_max_merger = max(retained_masses) if retained_masses else np.nan

        imbh_masses.append(m_imbh)
        max_merger_masses.append(m_max_merger)

        for m in mergers:
            m_rem = m.get('m_rem')
            if m_rem is not None:
                all_merger_masses.append(float(m_rem))

    return (np.array(imbh_masses), np.array(max_merger_masses),
            np.array(all_merger_masses), config)

def main():
    parser = argparse.ArgumentParser(description="Analyse results for a set of ICs.")
    parser.add_argument("--parent_dir", "--parent-dir", dest="parent_dir", type=str,
                        required=True,
                        help="Path to the parent directory containing configuration subdirectories.")
    parser.add_argument("--output_file", "--output-file", dest="output_file", type=str, help="Filename for the saved plot.")
    args = parser.parse_args()

    parent_path = Path(args.parent_dir)
    if not parent_path.exists():
        print(f"Error: Directory {parent_path} does not exist.")
        return

    config_dirs = sorted(
        (d for d in parent_path.iterdir()
         if d.is_dir() and not d.name.startswith('.') and "summary" not in d.name),
        key=lambda d: d.name,
    )
    if not config_dirs:
        print(f"No configuration subdirectories found in {parent_path}.")
        return
    print(f"Found {len(config_dirs)} configurations in {parent_path}.")

    # Load data
    all_data = {}
    global_min_imbh_mass = float('inf')
    global_max_imbh_mass = 0.0
    global_min_merger_mass = float('inf')
    global_max_merger_mass = 0.0

    for config_dir in config_dirs:
        print(f"Looking at {config_dir.name} :)")
        imbh_masses, max_merger_masses, merger_masses, config = load_config_data(config_dir)

        n_valid = int(np.sum(~np.isnan(imbh_masses)))
        if n_valid == 0 and merger_masses.size == 0:
            print("No retained merger products? Something's weird.")
            continue
        if config is None:
            print("Config not readable from stats. Make sure the stats block is properly formatted.")

        all_data[config_dir.name] = {
            'imbh_mass': imbh_masses,                  # mIMBH_final 
            'max_merger_mass': max_merger_masses,      # Retained maximum merger mass
            'merger_masses': merger_masses,
            'config': config,
        }

        if n_valid > 0:
            global_min_imbh_mass = min(global_min_imbh_mass, float(np.nanmin(imbh_masses)))
            global_max_imbh_mass = max(global_max_imbh_mass, float(np.nanmax(imbh_masses)))
        if merger_masses.size > 0:
            global_min_merger_mass = min(global_min_merger_mass, float(np.min(merger_masses)))
            global_max_merger_mass = max(global_max_merger_mass, float(np.max(merger_masses)))

        print(f"Found {n_valid} runs. Total mergers: {merger_masses.size}")

    if not all_data:
        print("No data found to plot.")
        return

    varying_params = [
        k for k in CONFIG_KEYS
        if len({all_data[name]['config'][k] for name in all_data}) > 1
    ]
    print(f"ICs being varied: {varying_params}")

    ordered_names = sorted(
        all_data,
        key=lambda n: [all_data[n]['config'][k] for k in varying_params] if varying_params else [n]
    )


    # Plots


    # 1. Probability p(M_IMBH > M)
    x_min = global_min_imbh_mass
    x_max = global_max_imbh_mass
   
    #mass_thresholds = np.logspace(np.log10(x_min), np.log10(x_max), 500)
    mass_thresholds = np.linspace(x_min, x_max, 500)

    plt.figure(figsize=(10, 6))
    for i, folder_name in enumerate(ordered_names):
        masses = all_data[folder_name]['imbh_mass']
        masses = masses[~np.isnan(masses)]
        config = all_data[folder_name]['config']
        if masses.size == 0:
            continue
        m_sorted = np.sort(masses)
        probs = 1.0 - np.searchsorted(m_sorted, mass_thresholds, side='right') / m_sorted.size
        label = config_label(config, varying_params, masses.size)
        plt.plot(mass_thresholds, probs, label=label,
                 color=COLOURS[(i + 1) % len(COLOURS)], linewidth=2.5)

    plt.axvline(x=500, color=COLOURS[0], linestyle=':', linewidth=2, label='500 $M_{\odot}$')
    plt.xscale('log')
    plt.xlabel(r'M ($M_{\odot}$)', fontsize=14)
    plt.ylabel(r'$p(M_{\mathrm{IMBH}} > M)$', fontsize=14)
    plt.xlim(x_min, x_max)
    plt.ylim(0, 1.05)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', length=7, width=1.2)
    ax.tick_params(axis='both', which='minor', length=4, width=0.8)
    plt.grid(True, which='major', linestyle='--', alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', alpha=0.35)
    plt.legend(fontsize=12, title_fontsize=13, loc='best')
    plt.tight_layout()
    output_path = parent_path / "imbh_mass_probability.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    # 2. Overlaid Histograms of raw merger masses
    fig, ax = plt.subplots(figsize=(10, 6))
    hist_x_min = global_min_merger_mass
    hist_x_max = global_max_merger_mass
    bins = np.logspace(np.log10(hist_x_min), np.log10(hist_x_max), 50)

    for i, folder_name in enumerate(ordered_names):
        masses = all_data[folder_name]['merger_masses']
        config = all_data[folder_name]['config']
        if masses.size == 0:
            continue
        n_runs = int(np.sum(~np.isnan(all_data[folder_name]['imbh_mass'])))
        label = config_label(config, varying_params, n_runs)
        color = COLOURS[(i + 1) % len(COLOURS)]
        ax.hist(masses, bins=bins, histtype='step', linewidth=2.5,
                color=color, label=label, zorder=10 - i)
        ax.hist(masses, bins=bins, histtype='stepfilled', alpha=0.25,
                color=color, zorder=2)

    ax.set_xlabel('Merger Product Mass ($M_{\odot}$)', fontsize=14)
    ax.set_ylabel('Number of Mergers', fontsize=14)
    ax.axvline(x=500, color=COLOURS[0], linestyle=':', linewidth=2,
               label='500 $M_{\odot}$', zorder=100)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True, linestyle='--', alpha=0.6, which='major')
    ax.legend(fontsize=11, loc='best')
    plt.tight_layout()
    hist_path = parent_path / "mass_histograms.png"
    plt.savefig(hist_path, dpi=300)
    plt.close()

    # 3. Normalized histogram of mIMBH_final
    valid_lists = [all_data[n]['imbh_mass'][~np.isnan(all_data[n]['imbh_mass'])]
                   for n in all_data]
    valid_lists = [v for v in valid_lists if v.size > 0]

    if valid_lists:
        valid_masses_all = np.concatenate(valid_lists)
        hist_min = float(np.min(valid_masses_all))
        max_merger = float(np.max(valid_masses_all))

        fig_imbh, ax_imbh = plt.subplots(figsize=(10, 6))
        for i, folder_name in enumerate(ordered_names):
            masses = all_data[folder_name]['imbh_mass']
            masses = masses[~np.isnan(masses)]
            config = all_data[folder_name]['config']
            if masses.size == 0:
                continue
            bins_imbh = np.linspace(hist_min, max_merger, max(2, len(masses) // 2))
            label = config_label(config, varying_params, masses.size)
            color = COLOURS[(i + 1) % len(COLOURS)]
            ax_imbh.hist(masses, bins=bins_imbh, histtype='step', density=True,
                         linewidth=2.5, color=color, label=label, zorder=100-i)
            ax_imbh.hist(masses, bins=bins_imbh, histtype='stepfilled', density=True,
                         alpha=0.3, color=color, zorder=100-i)

        ax_imbh.set_xscale('linear')
        ax_imbh.axvline(x=500, color=COLOURS[0], linestyle=':', linewidth=2,
                        label='500 $M_{\odot}$', zorder=100)
        ax_imbh.set_xlabel('IMBH Mass ($M_{\odot}$)', fontsize=14)
        ax_imbh.set_ylabel('Normalized Frequency', fontsize=14)
        ax_imbh.grid(True, linestyle='--', alpha=0.6, which='major')
        ax_imbh.legend(fontsize=11, loc='best')
        plt.tight_layout()
        end_hist_path = parent_path / "imbh_mass_histograms.png"
        plt.savefig(end_hist_path, dpi=300)
        plt.close()
    

    # 4. Normalized histogram of retained max merger masses
    mass_lists = [all_data[n]['max_merger_mass'] for n in all_data]
    mass_lists = [v for v in mass_lists if v.size > 0]

    if mass_lists:
        mass_lists_all = np.concatenate(mass_lists)
        min_merger = float(np.min(mass_lists_all))
        max_merger = float(np.max(mass_lists_all))

        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        for i, folder_name in enumerate(ordered_names):
            masses = all_data[folder_name]['max_merger_mass']
            masses = masses[~np.isnan(masses)]
            config = all_data[folder_name]['config']
            if masses.size == 0:
                continue
            bins_imbh = np.linspace(hist_min, max_merger, max(2, len(masses) // 2))
            label = config_label(config, varying_params, masses.size)
            color = COLOURS[(i + 1) % len(COLOURS)]
            ax_hist.hist(masses, bins=bins_imbh, histtype='step', density=True,
                         linewidth=2.5, color=color, label=label, zorder=100-i)
            ax_hist.hist(masses, bins=bins_imbh, histtype='stepfilled', density=True,
                         alpha=0.3, color=color, zorder=100-i)

        ax_hist.set_xscale('linear')
        ax_hist.axvline(x=500, color=COLOURS[0], linestyle=':', linewidth=2,
                        label='500 $M_{\odot}$', zorder=100)
        ax_hist.set_xlabel('Retained Max Merger Mass ($M_{\odot}$)', fontsize=14)
        ax_hist.set_ylabel('Normalized Frequency', fontsize=14)
        ax_hist.grid(True, linestyle='--', alpha=0.6, which='major')
        ax_hist.legend(fontsize=11, loc='best')
        plt.tight_layout()
        max_merger_path = parent_path / "max_merger_retained_mass_histograms.png"
        plt.savefig(max_merger_path, dpi=300)
        plt.close()

    # 5. Diagnostic plot for IMBH mass vs retained maximum merger mass
    fig, (ax_sc, ax_hist_diag) = plt.subplots(1, 2, figsize=(16, 6))

    all_vals = []
    for folder_name in ordered_names:
        d = all_data[folder_name]
        all_vals.append(d['imbh_mass'])
        all_vals.append(d['max_merger_mass'])

    if all_vals:
        all_vals = np.concatenate(all_vals)
        lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
        bins_diagnostic = np.linspace(lo, hi, 50)

        diagnostic_rows = []
        for i, folder_name in enumerate(ordered_names):
            d = all_data[folder_name]
            imbh = d['imbh_mass']
            max_merger = d['max_merger_mass']
            x, y = imbh, max_merger
            color = COLOURS[(i + 1) % len(COLOURS)]
            label = config_label(d['config'], varying_params, int(max_merger.size))

            ax_sc.scatter(x, y, color=color, alpha=0.65, s=40, label=label)

            ax_hist_diag.hist(x, bins=bins_diagnostic, density=True, histtype='step',
                         linewidth=2.5, color=color, label=label + ' [IMBH]', zorder=100-i)
            ax_hist_diag.hist(x, bins=bins_diagnostic, histtype='stepfilled', density=True,
                                     alpha=0.3, color=color, zorder=100-i)
            ax_hist_diag.hist(y, bins=bins_diagnostic, density=True, histtype='step',
                         linewidth=2.0, linestyle='--', color=color, label = label + ' [Max merger]', zorder=100-i)

            diff = y - x
            diagnostic_rows.append({
                'config': folder_name,
                'n_runs': int(max_merger.size),
                'mean_imbh': float(np.mean(x)),
                'mean_max_merger': float(np.mean(y)),
                'mean_diff': float(np.mean(diff)),
            })

        ax_sc.plot([lo, hi], [lo, hi], color='0.4',
                   linestyle='--', linewidth=2, label='1:1')
        ax_sc.set_xlim(lo, hi)
        ax_sc.set_ylim(lo, hi)
        ax_sc.set_xlabel(r'Final IMBH $m_{\mathrm{IMBH}}$', fontsize=12)
        ax_sc.set_ylabel(r'Max retained merger product mass ($M_{\odot}$)', fontsize=12)
        ax_sc.grid(True, linestyle='--', alpha=0.4)
        ax_sc.legend(fontsize=9, loc='best')

        ax_hist_diag.set_xlabel(r'BH Mass ($M_{\odot}$)', fontsize=13)
        ax_hist_diag.set_ylabel('Normalized frequency', fontsize=13)
        ax_hist_diag.grid(True, linestyle='--', alpha=0.4)
        ax_hist_diag.legend(fontsize=9, loc='best')

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        diagnostic_path = parent_path / "imbh_vs_retained_max_merger.png"
        
        plt.savefig(diagnostic_path, dpi=300)
        plt.close()

        if diagnostic_rows:
            csv_path = parent_path / "imbh_vs_max_retained_stats.csv"
            with open(csv_path, 'w', newline='') as fcsv:
                writer = csv.DictWriter(fcsv, fieldnames=list(diagnostic_rows[0].keys()))
                writer.writeheader()
                writer.writerows(diagnostic_rows)
            
    print(f"\nAll plots saved to {parent_path}")

if __name__ == "__main__":
    main()