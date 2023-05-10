#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu

rm -rf ~/.cache/torch_extensions

start=`date +%s`

set -ex

#python3 -m train experiment=synthetics/associative_recall/hyena-131k-30vs.yaml
# python3 -m train experiment=synthetics/associative_recall/hyena-2048-16vs.yaml
#python3 -m train experiment=synthetics/associative_recall/hyena-2048-16_FHN.yaml

python3 -m train experiment=synthetics/associative_recall/hyena_1dBurgers_seq.yaml
# python3 -m train experiment=synthetics/associative_recall/hyena_1dBurgers_icl_t.yaml
# python3 -m train experiment=synthetics/associative_recall/hyena_1dBurgers_fno.yaml

end=`date +%s`
runtime=$((end-start))
