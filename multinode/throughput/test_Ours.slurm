#!/bin/bash
#SBATCH -J test               # Job name
#SBATCH -o Ours_throughput_28gpus.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 7                  # Total number of nodes requested (devel is usually for single-node testing)
#SBATCH -n 29                  # Total number of tasks (single process)
#SBATCH -t 6:00:00           # Run time (hh:mm:ss) - 30 minutes max for devel
#SBATCH -p mi2104x             # Use the devel partition


# Use your local environment
export PATH=$HOME/.local/bin:$PATH  # Ensure local binaries are accessible
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29511
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
# Run the script
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "RANK=$RANK"

srun python serving_system.py --num_workers=28 --rate=10 --num_small=9 --image_directory ./MoDM_throughput --metadata_path 
