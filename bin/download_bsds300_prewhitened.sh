#!/usr/bin/env bash
# Downloads the BSDS300 dataset to the current working directory
set -ex

BSDS300_PATH='http://www.rctn.org/bruno/sparsenet/IMAGES.mat'
curl -O $BSDS300_PATH