#!/bin/bash
nsample=16
echo "Start script." > eval_ll.txt
for schedule in "sigmoid" "linear"
do
    for sampler in "hmc" "metropolis"
    do
        for chainlength in 2 10 100 1000
        do
            python3 tests/evaluate_ll.py tests.evaluate_ll.n_sample=$nsample tests.evaluate_ll.chain_length=$chainlength tests.evaluate_ll.sampler=$sampler tests.evaluate_ll.schedule_type=$schedule >> eval_ll.txt
        done
    done
done