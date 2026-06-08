#!/bin/bash
#SBATCH --partition main
#SBATCH --job-name ollama-na
#SBATCH --output /home/mickaelz/na_ollama_logs/ollama-%J.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=60G
#SBATCH --time=08:00:00

echo "$(date)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "NODE: $(hostname)  IP: $(hostname -i)"
bash "/home/mickaelz/Network analysis/scripts/ollama_serve.sh"
