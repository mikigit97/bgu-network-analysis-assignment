#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name chess-agg
#SBATCH --output /home/mickaelz/na_logs/chess-agg-%J.out
#SBATCH --time 0-02:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

echo "node=$(hostname) jobid=$SLURM_JOBID cpus=$SLURM_CPUS_PER_TASK"
cd "/home/mickaelz/Network analysis"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=TRUE
python3 scripts/chess_aggregate.py
echo "CHESS_SBATCH_DONE rc=$?"
