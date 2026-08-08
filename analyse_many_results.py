"""
analyse_many_results.py

Description:
Aggregates results from multiple configuration folders within a parent directory.
Plots probability of forming a Black Hole above a certain mass,
p(M_BH > M), for different initial conditions.

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

# IC labelling functions for legend generation
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


def get_config_name(config_dict):
    "To keep nameing consistent with single_model.py"
    return "_".join(PARAM_LABEL[k](config_dict[k]) for k in CONFIG_KEYS)


# def parse_config_from_folder(folder_name):
#     """
#     Fallback: parse the folder name back into a config dict.
#     Only used if the stats blocks are missing/unreadable.
#     """
#     config = {}
#     parts = folder_name.split('_')
#     key_map = {
#         'M0': float, 'rhoh0': float, 'FeH': float, 'rg': float, 'tend': float,
#         'Mvir': float, 'cHalo': float,
#     }
#     bool_map = {
#         'RantalaSeed': ('rantala_imbh_seed', True),  'StdSeed': ('rantala_imbh_seed', False),
#         'ExtIMF':      ('chattopadhyay_seed', True), 'StdIMF':  ('chattopadhyay_seed', False),
#         'GalpyPot':    ('galpy_potential', True),    'StdPot':  ('galpy_potential', False),
#     }
#     i = 0
#     while i < len(parts):
#         part = parts[i]
#         if part in key_map and i + 1 < len(parts):
#             try:
#                 val = key_map[part](parts[i + 1])
#                 config['M_vir' if part == 'Mvir' else 'c_halo' if part == 'cHalo' else part] = val
#                 i += 2
#                 continue
#             except ValueError:
#                 pass
#         if part in bool_map:
#             config[bool_map[part][0]] = bool_map[part][1]
#             i += 1
#             continue
#         i += 1

#     # Defaults so label generation can never KeyError
#     for k in CONFIG_KEYS:
#         config.setdefault(k, False if k in ('rantala_imbh_seed', 'chattopadhyay_seed', 'galpy_potential') else 0.0)
#     return config


def load_config_data(config_dir):
    """
    Reads all run_*/data.json under a config folder.
    Returns (masses_array, config_dict). Config is taken from the stats block
    of the first readable file (same source of truth as analyse_results.py).
    """
    masses = []
    config = None

    data_files = sorted(config_dir.glob("run_*/data.json"))
    if not data_files:
        data_files = sorted(p for p in config_dir.glob("**/data.json") if "summary" not in p.parts)

    for f_path in data_files:
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f" Error reading {f_path}: {e}")
            continue

        stats = data.get('stats', {})
        if config is None and all(k in stats for k in CONFIG_KEYS):
            config = {k: stats[k] for k in CONFIG_KEYS}

        mass = stats.get('mIMBH_final', None)
        if mass is not None and not np.isnan(mass):
            masses.append(float(mass))

    return np.array(masses), config

# Probability of forming a Black Hole above a certain mass threshold,
# p(M_BH > M), for different initial conditions.
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

    # Load data and find max mass
    all_data = {}
    global_max_mass = 0.0
    for config_dir in config_dirs:
        print(f"Looking at {config_dir.name} :)")
        masses, config = load_config_data(config_dir)

        if masses.size == 0:
            print("No BHs? Something's weird.")
            continue
        if config is None:
            print("Config not readable from stats. Make sure the stats block is properly formatted.")

        all_data[config_dir.name] = {'masses': masses, 'config': config}
        global_max_mass = max(global_max_mass, float(np.max(masses)))
        print(f"Found {masses.size} runs. Max Mass: {np.max(masses):.2f} Msun")

    if not all_data:
        print("No data found to plot.")
        return

    # X-axis grid: 100 to biggest IMBH formed across all runs
    x_min = 100.0
    x_max = global_max_mass
    if x_max <= x_min:
        print(f"[Warning] Largest IMBH ({x_max:.1f} Msun) <= {x_min:.0f} Msun; "
              f"extending axis to 1e3 for a valid log plot.")
        x_max = 1e3
    mass_thresholds = np.logspace(np.log10(x_min), np.log10(x_max), 500)
    print(f"\nPlotting mass range: {x_min:.0f} to {x_max:.2f} Msun")

    # What ICs vary across the folders (for the legend)
    varying_params = [
        k for k in CONFIG_KEYS
        if len({all_data[name]['config'][k] for name in all_data}) > 1
    ]
    print(f"ICs being varied: {varying_params}")

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, folder_name in enumerate(sorted(all_data.keys())):
        masses = all_data[folder_name]['masses']
        config = all_data[folder_name]['config']

        # p(M_BH > M) = 1 - (#masses <= M)/N
        m_sorted = np.sort(masses)
        probs = 1.0 - np.searchsorted(m_sorted, mass_thresholds, side='right') / m_sorted.size

        if varying_params:
            label = ", ".join(PARAM_LABEL[k](config[k]) for k in varying_params)
        else:
            label = folder_name if len(folder_name) <= 40 else folder_name[:37] + "..."

        plt.plot(mass_thresholds, probs, label=label,
                 color=COLOURS[i+1 % len(COLOURS)], linewidth=2.5)

    plt.axvline(x=500, color=COLOURS[0], linestyle=':', linewidth=2, label='500 $M_{\odot}$')

    # Formatting
    plt.xscale('log')
    plt.xlabel(r'Mass ($M_{\odot}$)', fontsize=14)
    plt.ylabel(r'$p(M_{\text{BH}} > M)$', fontsize=14)
    plt.xlim(x_min, x_max)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)

    
    plt.legend( fontsize=12, title_fontsize=13, loc='best')
    plt.tight_layout()

    output_path = parent_path / args.output_file
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    main()