#!/bin/bash
chainlength=32
echo "Start script." > gridsearch_epsilon_results_2022-08-26-08-37.txt
for schedule in "sigmoid" "linear"
do
    for eps in 10.0 5.0 2.0 1.0 0.5 0.1 0.05 0.01
    do
        python3 tests/evaluate_ll.py tests.evaluate_ll.chain_length=$chainlength tests.evaluate_ll.schedule_type=$schedule tests.evaluate_ll.sampler='hmc' tests.evaluate_ll.hmc_epsilon=$eps tests.evaluate_ll.mdl_path='outputs/2022-08-26/08-37' >> gridsearch_epsilon_results_2022-08-26-08-37.txt
    done
done