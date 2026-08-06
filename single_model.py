"""
single_model.py

Description:
    Runs a single CBHBD simulation.
    It reads a specific parameter set from a job list CSV based on a provided `task_id`, 
    initializes the `cbhbd` model, runs and saves the results 
    to a  JSON file.

Usage:
    python single_model.py --task_id <ID> --jobs_file <CSV_PATH> --output_dir <DIR_PATH>


Outputs:
    - A directory named `run_<task_id>` containing a `data.json` file with the complete 
      simulation state, statistics, trajectory, and merger details.
"""

import argparse
import csv
import json
import os
import numpy as np
from cbhbd import cbhbd


def get_config_name(config_dict):
    """
    Creates a filename from all config parameters.
    """
    seed_str = "RantalaSeed" if config_dict["rantala_imbh_seed"] else "StdSeed"
    imf_str = "ExtIMF" if config_dict["chattopadhyay_seed"] else "StdIMF"
    pot_str = "GalpyPot" if config_dict["galpy_potential"] else "StdPot"

    parts = [
        f"M0_{config_dict['M0']:.0e}",
        f"rhoh0_{config_dict['rhoh0']:.2e}",
        f"FeH_{config_dict['FeH']:.1f}",
        f"rg_{config_dict['rg']:.2f}",
        f"tend_{config_dict['tend']:.2f}",
        f"Mvir_{config_dict['M_vir']:.1e}",
        f"cHalo_{config_dict['c_halo']:.2f}",
        seed_str,
        imf_str,
        pot_str,
    ]

    return "_".join(parts)

def extract_growth(model, retained_only=True):
    """Extract max merger mass over time"""
    if not hasattr(model, 'mergers') or model.mergers.empty:
        return np.array([]), np.array([])

    df = model.mergers.sort_values('t_merge').copy()
    if retained_only:
        is_ejected_type = df['merger_type'].astype(str).str.contains('Ejected', case=False, na=False)
        is_kicked_out = df['v_kick'] >= df['v_esc']
        df = df[~(is_ejected_type | is_kicked_out)]
        
    if df.empty:
        return np.array([]), np.array([])

    times = df['t_merge'].to_numpy()
    max_merger_mass = np.maximum.accumulate(df['m_rem'].to_numpy())
    return times, max_merger_mass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--jobs_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.jobs_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['task_id']) == args.task_id:
                job = row
                break
        else:
            raise ValueError(f"Task ID {args.task_id} not found in {args.jobs_file}")

    M0 = float(job['M0'])
    rhoh0 = float(job['rhoh0'])
    FeH = float(job['FeH'])
    rg = float(job['rg'])
    tend = float(job['tend'])
    seed = int(job['seed'])
    
    rantala_imbh_seed = job['rantala_imbh_seed'] == 'True'
    chattopadhyay_seed= job['chattopadhyay_seed'] == 'True'
    galpy_potential = job['galpy_potential'] == 'True'
    M_vir = float(job['M_vir'])
    c_halo = float(job['c_halo'])
    
    config_dict = {
            "M0": M0,
            "rhoh0": rhoh0,
            "FeH": FeH,
            "rg": rg,
            "tend": tend,
            "rantala_imbh_seed": rantala_imbh_seed,
            "chattopadhyay_seed": chattopadhyay_seed,
            "galpy_potential": galpy_potential,
            "M_vir": M_vir,
            "c_halo": c_halo,
        }
    
    config_name = get_config_name(config_dict)

    model = cbhbd.CBHBD(
        M0=M0, rhoh0=rhoh0, FeH=FeH, rg=rg, tend=tend,
        integration_method="RK45", compute_mergers=True,
        remnant_model="RemnantModelIslam26", ifmr="sevn-rapid",
        seed=seed, 
        rantala_imbh_seed=rantala_imbh_seed,
        chattopadhyay_seed=chattopadhyay_seed,
        verbose=False,
        a_slopes=[-0.3, -1.65, -2.3], 
        m_breaks=[0.08, 0.4, 1, 150],
        galpy_potential=galpy_potential,
        M_vir=M_vir,
        conc=c_halo,
    )

    tend_in_years = tend * 1e6
    if hasattr(model, 'mergers') and not model.mergers.empty:
        model.mergers = model.mergers[model.mergers['t_merge'] <= tend_in_years].copy()

    # Extract Stats 
    stats = {
        "task_id": int(args.task_id), 
        "config_name": config_name,
        "seed": int(seed),
        "M0": float(M0), 
        "rhoh0": float(rhoh0), 
        "FeH": float(FeH), 
        "rg": float(rg), 
        "tend": float(tend),
        "rantala_imbh_seed": bool(rantala_imbh_seed),
        "chattopadhyay_seed": bool(chattopadhyay_seed),
        "galpy_potential": bool(galpy_potential),
        "M_vir": float(M_vir),
        "c_halo": float(c_halo),
        "mIMBH_final": float(model.mIMBH),
        "chiIMBH_final": float(model.chiIMBH),
        "genIMBH_final": float(model.genIMBH),
        "mIMBHej_final": float(model.mIMBHej) if model.mIMBHej is not None else None,
        "v_esc0": float(model.cluster.vesc0),
        "v_esc_final": float(model.cluster.vesc[-1]) if len(model.cluster.vesc) > 0 else None,
        "tcr0": float(model.cluster.tcr[0]),
        "n_mergers": int(len(model.mergers)) if hasattr(model, 'mergers') else 0,
        "vesc_t_myr": list(np.array(model.cluster.t) * 1e3), # Gyr to Myr
        "vesc": list(model.cluster.vesc),
        "vesc_cl": list(model.cluster.vesc_cl) if model.cluster.vesc_cl is not None else None, 
        
    }
    

    t_traj, max_mass_traj = extract_growth(model)
    trajectory = {
        "t_merge": [float(t) for t in t_traj], 
        "max_merger_mass": [float(m) for m in max_mass_traj]
    }

    mergers_data = []
    if hasattr(model, 'mergers') and not model.mergers.empty:
        for _, row in model.mergers.iterrows():
            mergers_data.append({
                't_merge': float(row['t_merge']),
                'm_rem': float(row['m_rem']),
                'v_kick': float(row['v_kick']),
                'v_esc': float(row['v_esc']),
                'merger_type': str(row['merger_type'])
            })

    run_data = {
        "stats": stats,
        "trajectory": trajectory,
        "mergers": mergers_data  
    }

    run_dir = os.path.join(args.output_dir, config_name, f"run_{args.task_id:05d}")
    os.makedirs(run_dir, exist_ok=True)
    
    with open(os.path.join(run_dir, "data.json"), 'w') as f:
        json.dump(run_data, f, indent=4)
        
    print(f"Task {args.task_id} (seed {seed}) done. IMBH: {model.mIMBH:.1f} Msun")

if __name__ == "__main__":
    main()