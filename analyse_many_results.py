"""
analyse_many_results.py
Description:
Aggregates results from multiple configuration folders within a parent directory.
Plots probability of forming a Black Hole above a certain mass,
p(M_BH > M), for different initial conditions.
ALSO plots overlaid histograms of ALL retained merger masses AND Final IMBH masses.
Legends show LaTeX-formatted ICs, the number of runs per config (N=...),
ordered by the varied parameter from smallest to highest.
Usage:
python analyse_many_results.py --parent_dir <PATH_TO_PARENT_FOLDER>
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
COLOURS = ["#000000", "#E69F00", "#56B4E8", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# ICs defining a unique configuration
CONFIG_KEYS = [
    'M0', 'rhoh0', 'FeH', 'rg', 'tend',
    'M_vir', 'c_halo',
    'rantala_imbh_seed', 'chattopadhyay_seed', 'galpy_potential',
]

# Plain labels: used ONLY for folder names (keep consistent with single_model.py)
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
}

def _sci(v, prec=2):
    """4.17e+05 -> 4.17\times10^{5}"""
    mant, _, exp = f"{v:.{prec}e}".partition('e')
    return rf"{mant}\times10^{{{int(exp)}}}"

# LaTeX labels: used ONLY for legends
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
}

def get_config_name(config_dict):
    "To keep naming consistent with single_model.py"
    return "_".join(PARAM_LABEL[k](config_dict[k]) for k in CONFIG_KEYS)

def config_label(config, varying_params, folder_name, n_runs):
    """LaTeX legend label + number of runs, e.g.  rho_h0 = 4.17e5 (N=200)"""
    if varying_params and config is not None:
        base = ", ".join(PARAM_LABEL_TEX[k](config[k]) for k in varying_params)
    else:
        base = folder_name if len(folder_name) <= 40 else folder_name[:37] + "..."
    return base + rf" ($N={n_runs}$)"

def load_config_data(config_dir):
    """
    Reads all run_*/data.json under a config folder.
    Returns (final_masses_array, all_merger_masses_array, config_dict).
    """
    final_masses = []
    all_merger_masses = []
    config = None

    data_files = sorted(config_dir.glob("run_*/data.json"))
    if not data_files:
        data_files = sorted(p for p in config_dir.glob("**/data.json") if "summary" not in p.parts)

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

        # Get Final IMBH Mass (for the probability plot & new normalized histogram)
        mass = stats.get('mIMBH_final', None)
        if mass is not None and not np.isnan(mass):
            final_masses.append(float(mass))

        # Get All Merger Masses (for the original histogram)
        mergers = data.get('mergers', [])
        for m in mergers:
            m_rem = m.get('m_rem')
            if m_rem is not None:
                all_merger_masses.append(float(m_rem))

    return np.array(final_masses), np.array(all_merger_masses), config

def main():
    parser = argparse.ArgumentParser(description="analyse multiple configuration results.")
    parser.add_argument("--parent_dir", "--parent-dir", dest="parent_dir", type=str, required=True,
                        help="Path to the parent folder containing configuration subfolders.")
    parser.add_argument("--output_file", "--output-file", dest="output_file", type=str,
                        default="mass_probability.png", help="Filename for the saved plot.")
    args = parser.parse_args()

    parent_path = Path(args.parent_dir)
    if not parent_path.exists():
        print(f"Error: Directory {parent_path} does not exist.")
        return

    # Find configuration subdirectories
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
    global_max_final_mass = 0.0
    global_min_merger_mass = float('inf')
    global_max_merger_mass = 0.0

    for config_dir in config_dirs:
        print(f"Looking at {config_dir.name} :)")
        final_masses, merger_masses, config = load_config_data(config_dir)

        if final_masses.size == 0 and merger_masses.size == 0:
            print("No BHs? Something's weird.")
            continue

        if config is None:
            print("Config not readable from stats. Make sure the stats block is properly formatted.")

        all_data[config_dir.name] = {
            'final_masses': final_masses,
            'merger_masses': merger_masses,
            'config': config
        }

        if final_masses.size > 0:
            global_max_final_mass = max(global_max_final_mass, float(np.max(final_masses)))
        if merger_masses.size > 0:
            global_min_merger_mass = min(global_min_merger_mass, float(np.min(merger_masses)))
            global_max_merger_mass = max(global_max_merger_mass, float(np.max(merger_masses)))

        print(f"Found {final_masses.size} runs. Max Final Mass: {np.max(final_masses) if final_masses.size > 0 else 0:.2f} Msun. Total Mergers: {merger_masses.size}")

    if not all_data:
        print("No data found to plot.")
        return

    # What ICs vary across the folders?
    varying_params = [
        k for k in CONFIG_KEYS
        if len({all_data[name]['config'][k] for name in all_data}) > 1
    ]
    print(f"ICs being varied: {varying_params}")

    # Order configs by the varied IC value(s), smallest -> highest.
    # Legend order (and colour assignment) follows this order.
    ordered_names = sorted(
        all_data,
        key=lambda n: [all_data[n]['config'][k] for k in varying_params] if varying_params else [n]
    )

    # ==========================================
    # 1. Probability p(M_BH > M) Plot
    # ==========================================
    x_min = 100.0
    x_max = global_max_final_mass
    if x_max <= x_min:
        x_max = 1e3

    mass_thresholds = np.logspace(np.log10(x_min), np.log10(x_max), 500)
    print(f"\nPlotting probability mass range: {x_min:.0f} to {x_max:.2f} Msun")

    plt.figure(figsize=(10, 6))
    for i, folder_name in enumerate(ordered_names):
        masses = all_data[folder_name]['final_masses']
        config = all_data[folder_name]['config']
        if masses.size == 0: continue

        m_sorted = np.sort(masses)
        probs = 1.0 - np.searchsorted(m_sorted, mass_thresholds, side='right') / m_sorted.size

        label = config_label(config, varying_params, folder_name, masses.size)

        plt.plot(mass_thresholds, probs, label=label,
                 color=COLOURS[(i+1) % len(COLOURS)], linewidth=2.5)

    plt.axvline(x=500, color=COLOURS[0], linestyle=':', linewidth=2, label='500 $M_{\odot}$')
    plt.xscale('log')
    plt.xlabel(r'Mass ($M_{\odot}$)', fontsize=14)
    plt.ylabel(r'$p(M_{\text{IMBH}} > M)$', fontsize=14)
    plt.xlim(x_min, x_max)
    plt.ylim(0, 1.05)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', length=7, width=1.2)
    ax.tick_params(axis='both', which='minor', length=4, width=0.8)
    plt.grid(True, which='major', linestyle='--', alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', alpha=0.35)
    plt.legend(fontsize=12, title_fontsize=13, loc='best')
    plt.tight_layout()

    output_path = parent_path / args.output_file
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nPlot saved to {output_path}")

    # ==========================================
    # 2. Overlaid Histograms (All Merger Masses)
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    hist_x_min = global_min_merger_mass
    hist_x_max = global_max_merger_mass
    print(f"\nPlotting histogram mass range: {hist_x_min:.2f} to {hist_x_max:.2f} Msun")

    bins = np.logspace(np.log10(hist_x_min), np.log10(hist_x_max), 50)

    for i, folder_name in enumerate(ordered_names):
        masses = all_data[folder_name]['merger_masses']
        config = all_data[folder_name]['config']
        if masses.size == 0: continue

        n_runs = all_data[folder_name]['final_masses'].size
        label = config_label(config, varying_params, folder_name, n_runs)

        color = COLOURS[(i + 1) % len(COLOURS)]
        ax.hist(masses, bins=bins, histtype='step', linewidth=2.5,
                color=color, label=label, zorder=10-i)
        ax.hist(masses, bins=bins, histtype='stepfilled', alpha=0.25,
                color=color, zorder=2)

    ax.set_xlabel('Merger Product Mass ($M_{\odot}$)', fontsize=14)
    ax.set_ylabel('Number of Mergers', fontsize=14)
    ax.axvline(x=500, color=COLOURS[0], linestyle=':', linewidth=2, label='500 $M_{\odot}$', zorder=100)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True, linestyle='--', alpha=0.6, which='major')
    ax.legend(fontsize=11, loc='best')
    plt.tight_layout()

    hist_path = parent_path / "mass_histograms.png"
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Plot saved to {hist_path}")

    # ==========================================
    # 3. Normalized Histogram of Final IMBH Masses (log space)
    # ==========================================
    fig_imbh, ax_imbh = plt.subplots(figsize=(10, 6))

    imbh_masses_all = np.concatenate([all_data[n]['final_masses']
                                      for n in all_data
                                      if all_data[n]['final_masses'].size > 0])
    hist_imbh_min = float(np.min(imbh_masses_all))
    hist_imbh_max = float(np.max(imbh_masses_all))
    print(f"\nPlotting normalized IMBH mass histogram range: "
          f"{hist_imbh_min:.2f} to {hist_imbh_max:.2f} Msun")

    for i, folder_name in enumerate(ordered_names):
        masses = all_data[folder_name]['final_masses']
        config = all_data[folder_name]['config']
        if masses.size == 0:
            continue

        # Extract bin count from the number of successful runs for this config
        num_bins = len(masses)

        # Log-spaced edges: num_bins + 1 edges -> num_bins bins
        #bins_imbh = np.logspace(np.log10(hist_imbh_min), np.log10(hist_imbh_max), num_bins + 1)
        bins_imbh = np.linspace((hist_imbh_min),(hist_imbh_max), int(num_bins / 4))
        label = config_label(config, varying_params, folder_name, masses.size)

        color = COLOURS[(i + 1) % len(COLOURS)]

        ax_imbh.hist(masses, bins=bins_imbh, histtype='step', density=True,
                     linewidth=2.5, color=color, label=label, zorder=10 - i)
        ax_imbh.hist(masses, bins=bins_imbh, histtype='stepfilled', density=True,
                     alpha=0.25, color=color, zorder=2)

    ax_imbh.set_xscale('linear')
    ax_imbh.axvline(x=500, color=COLOURS[0], linestyle=':', linewidth=2,
                    label='500 $M_{\odot}$', zorder=100)
    ax_imbh.set_xlabel('Final IMBH Mass ($M_{\odot}$)', fontsize=14)
    ax_imbh.set_ylabel('Normalized Frequency', fontsize=14)
    ax_imbh.grid(True, linestyle='--', alpha=0.6, which='major')
    ax_imbh.legend(fontsize=11, loc='best')
    plt.tight_layout()

    imbh_hist_path = parent_path / "imbh_mass_histograms.png"
    plt.savefig(imbh_hist_path, dpi=300)
    plt.close()
    print(f"Plot saved to {imbh_hist_path}")

if __name__ == "__main__":
    main()