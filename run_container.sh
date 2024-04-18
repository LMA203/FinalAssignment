export WANDB_API_KEY=d9959c91dada8ed7e3d8b12b666ecc4a55555c7f
export WANDB_DIR=wandb/$SLURM_JOBID
export WANDB_CONFIG_DIR=wandb/$SLURM_JOBID
export WANDB_CACHE_DIR=wandb/$SLURM_JOBID
export WANDB_START_METHOD="thread"
wandb login

torchrun --nnodes=1 --nproc_per_node=1 train.py \
         --data_path "/gpfs/home1/scur0770/FinalAssignment/augmented_data"
