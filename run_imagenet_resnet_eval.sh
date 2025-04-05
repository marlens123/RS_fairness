#!/bin/bash
#SBATCH -p long
#SBATCH --time=2-00:00:00
#SBATCH --error=error.txt
#SBATCH --mem=64GB

module load anaconda/3

conda activate rs_fairness

export WANDB_API_KEY=d2ca547c9f807e8db70308537f4d7b64b6077b81

python -m src.eval --pretraining_dataset "ImageNet" --imagenet_model_identifier "resnet50" --saved_weights "ImageNet_resnet50_adamw_lr070_model_weights.pth