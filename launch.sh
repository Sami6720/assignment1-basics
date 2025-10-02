#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --time=0:80:00
#SBATCH --gpus=h100_3g.40gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=62G

module load httpproxy
source .venv/bin/activate

echo $(pwd)

echo "Python command being run is $1"

$1
