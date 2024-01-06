#!/bin/bash

cd ..

# This script is used to run the test program for the viola-jones-fold,
# Run it in the root folder of Exercise1

echo "Running the test of Viola-jones detector on FLICKR dataset..."

python3 viola-jones-custom.py --display True --displayTime 1

echo "Evaluating the results..."
python3 evaluate-custom.py

echo "Done!"