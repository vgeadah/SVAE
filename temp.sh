#!/bin/bash
i=0
hmcepsilon=$(python3 tests/find_hmc_epsilon.py tests.evaluate_ll.chain_length=16 tests.evaluate_ll.mdl_path="multirun/2022-09-28/17-25-06/$i")
echo $hmcepsilon
