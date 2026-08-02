# import itertools
# import os
# import subprocess
# import sys

# # Configuration Parameters
# M0_GRID = [1e4, 1e5]
# RHO_GRID = [1e5, 1e6]
# Z_FEH_GRID = [-1.3, -0.7] # FeH values
# RG_GRID = [2.0, 5.0]
# TEND_GRID = [10.0]
# SEEDS = range(100, 150) # Example: 50 runs per config
# IMBH_SEEDS_CONFIG = [True] # Use Rantala prescription?
# M_POT_GRID = [1e10]
# M_HALO_GRID = [1e10]
# R_HALO_KPC_GRID = [10.0]
# T_SEG_GRID = [5.0]

# INTEGRATION_METHOD = "RK45"
# REMNANT_MODEL = "RemnantModelIslam26"
# IFMR = "sevn-rapid"

# GRID = list(itertools.product(
#     M0_GRID, RHO_GRID, Z_FEH_GRID, RG_GRID, TEND_GRID, SEEDS,
#     IMBH_SEEDS_CONFIG, M_POT_GRID, M_HALO_GRID, R_HALO_KPC_GRID, T_SEG_GRID
# ))

# # --- Job Submission Configuration
# SBATCH_TEMPLATE_FILE = "job_template.sh" # Path to the template file
# MAIN_OUTPUT_DIR = "simulation_output" # Base directory for all results

# def submit_job(M0, rhoh0, FeH, rg, tend, seed, imbh_seeds_config, M_POT, M_halo, r_halo_kpc, t_seg):
#     """Submits a single job using sbatch."""
#     job_script_path = f"submit_job_M0_{M0}_rhoh0_{rhoh0}_FeH_{FeH}_seed_{seed}.sh"

#     # Read the template
#     with open(SBATCH_TEMPLATE_FILE, 'r') as f:
#         template_content = f.read()

#     # Replace placeholders with actual values
#     filled_content = template_content.format(
#         M0=M0,
#         RHOH0=rhoh0,
#         FEH=FeH,
#         RG=rg,
#         TEND=tend,
#         SEED=seed,
#         IMBH_SEEDS_CONFIG=str(imbh_seeds_config),
#         M_POT=M_POT,
#         M_HALO=M_halo,
#         R_HALO_KPC=r_halo_kpc,
#         T_SEG=t_seg,
#         INTEGRATION_METHOD=INTEGRATION_METHOD,
#         REMNANT_MODEL=REMNANT_MODEL,
#         IFMR=IFMR,
#         OUTPUT_DIR_BASE=MAIN_OUTPUT_DIR
#     )

#     # Write the filled script
#     with open(job_script_path, 'w') as f:
#         f.write(filled_content)

#     # Submit the job
#     try:
#         result = subprocess.run(['sbatch', job_script_path], check=True, capture_output=True, text=True)
#         print(f"Submitted job for seed {seed}: {result.stdout.strip()}")
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to submit job for seed {seed}: {e}")
#         print(f"Error output: {e.stderr}")

# if __name__ == "__main__":
#     os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True) # Create main output directory

#     n_total = len(GRID)
#     print(f"Submitting {n_total} jobs...")

#     for i, (M0, rhoh0, FeH, rg, tend, seed, imbh_seeds_config, M_POT, M_halo, r_halo_kpc, t_seg) in enumerate(GRID, 1):
#         print(f"Submitting job {i}/{n_total} (M0={M0}, rhoh0={rhoh0}, FeH={FeH}, seed={seed})...")
#         submit_job(M0, rhoh0, FeH, rg, tend, seed, imbh_seeds_config, M_POT, M_halo, r_halo_kpc, t_seg)

#     print("All jobs submitted.")
