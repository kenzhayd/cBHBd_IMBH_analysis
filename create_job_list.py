"""
create_job_list.py

Description:
    Generates a parameter grid for  CBHBD cluster simulations 
    and exports it as a CSV job list. This script is designed to set up a SLURM array of models.

Outputs:
    - A CSV file (default: 'test.csv') containing a unique 'task_id' and the corresponding 
      parameter set for every combination in the generated grid.

Note: Ensure your SLURM array range matches this total!

"""
import itertools
import csv

# =============================================================================
# CONFIGURATION
# =============================================================================
job_list_name = "vanilla_densities_200.csv"

M0 = [1e7]
LOG_RHO = [5.62]  # 5.62 in paper
RHO = [10 ** x for x in LOG_RHO]
Z_FEH = [-1.7]
RG = [26.25]
TEND = [3000]

# Set this to range(1, 501) to run 500 seeds per configuration
SEEDS = range(1, 201) 

RANTALA_IMBH_SEED = [True]  # Toggle Rantala 2026 IMBH seed formation
CHATTOPADHYAY_SEED  = [False]       # Toggle Extended IMF
GALPY_POTENTIAL = [True]    # Toggle Galpy NFW Halo Potential
M_VIR = [1e9]              # Virial mass used if galpy potential is enabled. [1e12 Msun]
C_HALO = [5,7,12,17]               # NFW Concentration parameter used if galpy potential is enabled.

# Generate of runs grid
GRID = list(itertools.product(
    M0, RHO, Z_FEH, RG, TEND, SEEDS,
    RANTALA_IMBH_SEED, CHATTOPADHYAY_SEED , GALPY_POTENTIAL, M_VIR, C_HALO
))

# Write to jobs.csv
with open(job_list_name, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'task_id', 'M0', 'rhoh0', 'FeH', 'rg', 'tend', 'seed',
        'rantala_imbh_seed', 'chattopadhyay_seed', 'galpy_potential', 
        'M_vir', 'c_halo'
    ])
    for i, params in enumerate(GRID, 1):
        row = [i] + [str(p) for p in params]
        writer.writerow(row)

print(f"Generated {job_list_name} with {len(GRID)} total tasks.")
print("NOTE: Ensure your SLURM array range matches this total!")
