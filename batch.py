#!/usr/bin/env bash
#SBATCH --job-name SpectGen # CHANGE this to a name of your choice
#SBATCH --time 24:00:00 # Run 24 hours
#SBATCH --qos=normal # possible #values##: short, normal, allgpus, 1gpulong
#SBATCH --gres=gpu:1 # CHANGE this if you need more or less GPUs
#SBATCH --nodelist=nv-ai-03.srv.aau.dk # CHANGE this to nodename of your choice. Currently only two possible nodes are available: nv-ai-01.srv.aau.dk, nv-ai-03.srv.aau.dk
##SBATCH --dependency=aftercorr:498 # More info slurm head node: `man --pager='less -p \--dependency' sbatch`

## Run actual analysis
## The benefit with using multiple srun commands is that this creates sub-jobs for your sbatch script and be uded for advanced usage with SLURM (e.g. create checkpoints, recovery, ect)
srun python spectrogram_generation.py
srun echo finish analysis
