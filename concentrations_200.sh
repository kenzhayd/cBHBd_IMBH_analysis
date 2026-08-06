#!/bin/bash
#SBATCH --account=def-vhenault
#SBATCH --job-name=Ocen_cbhbd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1       
#SBATCH --mem=3G                
#SBATCH --time=00:20:00
#SBATCH --array=1-500%192               # Change to match jobs.csv line count. --array=1-500%192  allows no more than 192 of the 500 jobs to run at once
#SBATCH --output=logs/slurm-%A_%a.out   # Job ID and Array Task ID 
#SBATCH --error=logs/slurm-%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mackenzie.hayduk@smu.ca


mkdir -p logs
mkdir -p output

module load gcc arrow


# Load environment
module load python/3.11
source /home/kenzhayd/projects/def-vhenault/kenzhayd/cbhbd_env/bin/activate

cd /home/kenzhayd/projects/def-vhenault/kenzhayd/cBHBd_IMBH_analysis

# Run the single model script, passing the SLURM Array Task ID
python single_model.py --task_id $SLURM_ARRAY_TASK_ID --jobs_file concentrations_200.csv --output_dir output
