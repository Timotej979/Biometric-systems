#!/bin/bash

cd ..

# This script is used to run the test program for the viola-jones-fold,
# Run it in the root folder of Exercise1

echo "Running the test of Viola-jones detector on FDDB dataset..."

python3 viola-jones-fold.py --display True --displayTime 0.5

echo "Evaluating the results..."
docker run --rm -it -v $(pwd):/FDDB housebw/fddb-evaluator

echo "Done!"