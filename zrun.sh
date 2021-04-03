#!/bin/bash
#
#SBATCH --mail-user=zcrenshaw@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home-nfs/zcrenshaw/slurm/out/%j.%N.stdout
#SBATCH --error=/home-nfs/zcrenshaw/slurm/out/%j.%N.stderr
#SBATCH --partition=speech-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
python train_zennet.py



