#!/bin/bash

# This script is used to build and run the detector images for the evaluation framework with volumes
# To run in the background use the following command:
#   nohup ./evaluateDetectors.sh > foo.log 2> foo.err < /dev/null &

# Build the docker images for every detector and create the volumes
source buildDetectorImages.sh

# Get the dataset directory
export DATASET_DIR=$(pwd)/datasets
export DETECTOR_OUTPUT_DIR=$(pwd)/detector-output

printenv | grep DATASET_DIR
printenv | grep DETECTOR_OUTPUT_DIR

# Run the docker images for every detector, mounting the volumes created above
# Add the following flags to the docker run command to enable GPU support and increase the shared memory size (After the -it flag):
#   --gpus all --shm-size 16G 
#   -it --gpus all --shm-size 16G

# TESTING:
#docker run -it -v $DATASET_DIR:/datasets:ro -v $DETECTOR_OUTPUT_DIR/oc-fd:/output oc-fd-image /bin/bash
docker run -it -v $DATASET_DIR:/datasets:ro -v $DETECTOR_OUTPUT_DIR/sbi:/output sbi-image /bin/bash

# PRODUCTION:
#docker run --gpus all --shm-size 16G -v $DATASET_DIR:/datasets:ro -v $DETECTOR_OUTPUT_DIR/oc-fd:/output oc-fd-image /bin/bash
#docker run --gpus all --shm-size 64GB -v $DATASET_DIR:/datasets:ro -v $DETECTOR_OUTPUT_DIR/sbi:/output sbi-image /bin/bash
