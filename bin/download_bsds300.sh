#!/usr/bin/env bash
# Downloads the BSDS300 dataset to the current working directory
set -ex

BSDS300_PATH='https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/'\
'bsds/BSDS300-images.tgz'

curl -O $BSDS300_PATH
tar -xzf BSDS300-images.tgz
mv BSDS300 bsds300
rm BSDS300-images.tgz
