#!/bin/bash
#SBATCH --job-name=py-matinv     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=vg0233@princeton.edu

module purge
module load anaconda3/2020.11
conda activate svae4

python3 bin/train_svae.py bin.train_svae.prior='GAUSSIAN' bin.train_svae.learning_rate=1e-04 bin.train_svae.num_epochs=256
# python3 tests/evaluate_ll.py tests.evaluate_ll.mdl_path='outputs/2022-09-07/10-03-20' tests.evaluate_ll.hmc_epsilon=0.8 tests.evaluate_ll.chain_length=1024
