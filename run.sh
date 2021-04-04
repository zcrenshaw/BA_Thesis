#!/bin/bash
#
#SBATCH --mail-user=zcrenshaw@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home-nfs/zcrenshaw/slurm/out/%j.%N.stdout
#SBATCH --error=/home-nfs/zcrenshaw/slurm/out/%j.%N.stderr
#SBATCH --partition=speech-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
python train_vqvae.py -L 16 -D 64 -K 512 
python train_vqvae.py -L 8 -D 64 -K 512
python train_vqvae.py -L 4 -D 64 -K 512
python train_vqvae.py -L 2 -D 64 -K 512





