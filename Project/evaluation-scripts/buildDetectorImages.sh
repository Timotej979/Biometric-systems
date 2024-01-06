#!/bin/bash

cd ..

# This script is used to build the detector images for the evaluation framework via docker compose.

# Execute the global configuration script.
source .env

# Build the docker images for every detector, volumes are mounted on execution
for detector in "${DETECTOR_FOLDERS[@]}"
do
    # Go to correct directory
    cd detectors/$detector

    echo "Building Docker image for $detector..."

    # Build the Docker image, mounting the volumes created above
    docker build -t "$detector-image" .

    # Go back to the project root directory
    cd ../..
done

echo "Current docker images:"
docker images