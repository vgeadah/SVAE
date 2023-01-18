#!/bin/bash
#SBATCH --job-name=SVAE     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=5        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=5G         # memory per cpu-core (4G is default)
#SBATCH --time=20:02:00          # total run time limit (HH:MM:SS)
# #SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=vg0233@princeton.edu

module purge
module load anaconda3/2022.5
conda activate svae4

# python3 likelihood_eval/evaluate_ll.py tests.evaluate_ll.mdl_path="outputs/2022-12-12/15-42-19" tests.evaluate_ll.chain_length=4096
# python3 likelihood_eval/evaluate_ll.py tests.evaluate_ll.mdl_path="outputs/2022-12-12/15-42-19" tests.evaluate_ll.hmc_epsilon=0.3258 tests.evaluate_ll.chain_length=1024
# python3 likelihood_eval/evaluate_ll.py tests.evaluate_ll.mdl_path="outputs/SavedModels/SVAE/LAPLACE_e64_ll-2.0_lr1e-04/" tests.evaluate_ll.chain_length=4096
python3 likelihood_eval/evaluate_ll.py tests.evaluate_ll.mdl_path="outputs/2022-12-12/16-59-19" tests.evaluate_ll.chain_length=4096

#python3 X/retrieve_ll.py
#python3 bin/train_svae.py --multirun bin.train_svae.prior='LAPLACE','CAUCHY','GAUSSIAN' bin.train_svae.learning_rate=1e-03,1e-04,1e-05 bin.train_svae.num_epochs=128
#python3 bin/train_svae.py --multirun bin.train_svae.prior='LAPLACE','CAUCHY','GAUSSIAN' bin.train_svae.learning_rate=1e-05 bin.train_svae.num_epochs=4,32 bin.train_svae.likelihood_logscale=-0.602
# for i in {0..1} 
# do
#        hmcepsilon=$(python3 tests/find_hmc_epsilon.py tests.evaluate_ll.chain_length=1024 tests.evaluate_ll.mdl_path="multirun/2022-11-17/12-37-41/$i")
#        python3 tests/evaluate_ll.py tests.evaluate_ll.mdl_path="multirun/2022-11-17/12-37-41/$i" tests.evaluate_ll.chain_length=1024 tests.evaluate_ll.hmc_epsilon=$hmcepsilon
# done
# python3 tests/evaluate_ll.py tests.evaluate_ll.mdl_path='outputs/2022-09-07/10-03-20' tests.evaluate_ll.hmc_epsilon=0.8 tests.evaluate_ll.chain_length=1024
