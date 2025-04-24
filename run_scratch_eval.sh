#!/bin/bash
#SBATCH -p long
#SBATCH --time=2-00:00:00
#SBATCH --error=error.txt
#SBATCH --mem=64GB

module load anaconda/3

conda activate rs_fairness

python -m src.eval --pretraining_dataset "none" --imagenet_model_identifier "swinb" --saved_weights "scratch_sgd_lr029_model_weights.pth"
python -m src.eval --pretraining_dataset "none" --imagenet_model_identifier "swinb" --saved_weights "scratch_sgd_lr029_model_weights.pth" --split "random"
python -m src.eval --pretraining_dataset "none" --imagenet_model_identifier "swinb" --saved_weights "scratch_sgd_lr029_model_weights.pth" --split "urban_rural"