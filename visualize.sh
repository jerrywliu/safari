#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu

rm -rf ~/.cache/torch_extensions

start=`date +%s`

set -ex

python3 visualize.py

end=`date +%s`
runtime=$((end-start))
