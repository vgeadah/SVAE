#!/bin/bash
#SBATCH --job-name=sparsenet     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=12        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=5G         # memory per cpu-core (4G is default)
#SBATCH --time=24:02:00          # total run time limit (HH:MM:SS)
# #SBATCH --gres=gpu:0             # number of gpus per node
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=vg0233@princeton.edu

module purge
module load anaconda3/2022.5
conda activate svae4

# python3 bin/train_sparsenet.py --multirun models.sparsenet.prior='CAUCHY' train.sparsenet.learning_rate=0.1,1.0,5.0 train.sparsenet.num_steps=50 models.sparsenet.likelihood_logscale=-2.0 tests.evaluate_ll.chain_length=128
python3 bin/train_sparsenet.py --multirun models.sparsenet.prior='GAUSSIAN' \
    train.sparsenet.learning_rate=1.0 \
    train.sparsenet.num_steps=5000 \
    models.sparsenet.likelihood_logscale=1.0 \
    tests.evaluate_ll.chain_length=256 \
    train.sparsenet.minibatch_size=100

# python3 bin/train_sparsenet.py --multirun models.sparsenet.prior='LAPLACE','GAUSSIAN' models.sparsenet.likelihood_logscale=-2.0,-1.0,-0.602 train.sparsenet.num_steps=100,1000,10000,100000 train.sparsenet.learning_rate=1.0,5.0
#python3 bin/train_svae.py --multirun bin.train_svae.prior='LAPLACE','CAUCHY','GAUSSIAN' bin.train_svae.learning_rate=1e-03,1e-04,1e-05 bin.train_svae.num_epochs=128
#python3 bin/train_svae.py --multirun bin.train_svae.prior='LAPLACE','CAUCHY','GAUSSIAN' bin.train_svae.learning_rate=1e-05 bin.train_svae.num_epochs=4,32 bin.train_svae.likelihood_logscale=-0.602
#for i in {14..47} 
#do
#	hmcepsilon=$(python3 tests/find_hmc_epsilon.py tests.evaluate_ll.chain_length=1024 tests.evaluate_ll.mdl_path="multirun/2022-09-28/17-25-06/$i")
#	echo $hmcepsilon
#	python3 tests/evaluate_ll.py tests.evaluate_ll.mdl_path="multirun/2022-09-28/17-25-06/$i" tests.evaluate_ll.chain_length=1024 tests.evaluate_ll.hmc_epsilon=$hmcepsilon
#done
# python3 tests/evaluate_ll.py tests.evaluate_ll.mdl_path='outputs/2022-09-07/10-03-20' tests.evaluate_ll.hmc_epsilon=0.8 tests.evaluate_ll.chain_length=1024
