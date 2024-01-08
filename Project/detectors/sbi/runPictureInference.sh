#!/bin/bash

# Set the CUDA_VISIBLE_DEVICES variable
export CUDA_VISIBLE_DEVICES=*

# Specify the path to the directory containing images
image_directory="/datasets/face-forensics/test"

# Specify the path to the weights file
weights_file="weights/FFraw.tar"

# Find all image files in the specified directory and its subdirectories
find "$image_directory" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -print0 |
while IFS= read -r -d '' image_file; do
    # Execute the inference code for each image
    python3 src/inference/inference_image.py -w "$weights_file" -i "$image_file"
done
