#!/bin/bash

cd ..

# This script is used to run the programs automaticaly,
# Run it in the root folder of Exercise1

echo "Running the Viola-jones detector for all folds with all parameter combinations..."

# Set the environment variable
echo "Setting the environment variable..."
export FDDB_HOME=$(pwd)

# Set the detector input arguments that the program will go through
echo "Setting the detector input arguments that the program will go through..."
cascades=("haarcascade_frontalface_default.xml" "haarcascade_frontalface_alt.xml" "haarcascade_frontalface_alt2.xml" "haarcascade_frontalface_alt_tree.xml")
scaleFactors=(1.05 1.1)
minNeighbors=(7 8 9)
minSizeXs=(10 20)
minSizeYs=(10 20)
maxSizeXs=(200 300)
maxSizeYs=(200 300)


# Calculate number of iterations
numOfIterations=$(( ${#cascades[@]} * ${#scaleFactors[@]} * ${#minNeighbors[@]} * ${#minSizeXs[@]} * ${#minSizeYs[@]} * ${#maxSizeXs[@]} * ${#maxSizeYs[@]} ))
echo "Number of iterations: $numOfIterations"

# Run Viola-jones detector for folds 1 to 10 with all parameter combinations
echo "Running Viola-jones detector for folds 1 to 10 with all parameter combinations..."

for cascade in "${cascades[@]}" 
do
    echo " - Cascade: $cascade"

    for scaleFactor in "${scaleFactors[@]}" 
    do
        echo " - Scale factor: $scaleFactor"

        for minNeighbor in "${minNeighbors[@]}" 
        do
            echo " - Min neighbours: $minNeighbor"
            
            for minSizeX in "${minSizeXs[@]}" 
            do
                echo " - Min size X: $minSizeX"

                for minSizeY in "${minSizeYs[@]}" 
                do
                    echo " - Min size Y: $minSizeY"
                
                    for maxSizeX in "${maxSizeXs[@]}" 
                    do
                        echo " - Max size X: $maxSizeX"
                    
                        for maxSizeY in "${maxSizeYs[@]}" 
                        do
                            echo " - Max size Y: $maxSizeY"

                            # Run the program for all folds
                            for fold_num in 1 3 5 7 9 
                            do

                                nextFold=$((fold_num+1))

                                echo "Running folds $fold_num, $nextFold..."

                                # Currently running 5 concurent programs, add more if you want
                                prog1="python3 viola-jones-fold.py \
                                    --cascade "$cascade" \
                                    --scaleFactor "$scaleFactor" \
                                    --minNeighbor "$minNeighbor" \
                                    --minSizeX "$minSizeX" \
                                    --minSizeY "$minSizeY" \
                                    --maxSizeX "$maxSizeX" \
                                    --maxSizeY "$maxSizeY" \
                                    --fold "$fold_num""

                                prog2="python3 viola-jones-fold.py \
                                    --cascade "$cascade" \
                                    --scaleFactor "$scaleFactor" \
                                    --minNeighbor "$minNeighbor" \
                                    --minSizeX "$minSizeX" \
                                    --minSizeY "$minSizeY" \
                                    --maxSizeX "$maxSizeX" \
                                    --maxSizeY "$maxSizeY" \
                                    --fold "$nextFold""

                                # Run the program
                                (trap 'kill 0' SIGINT; $prog1 & $prog2)

                                # Wait for the program to finish
                                wait

                            done

                            # Run the evaluation through Docker
                            echo "Evaluating the results..."
                            docker run --rm -it -v $(pwd):/FDDB housebw/fddb-evaluator

                            # Move the results to the results folder
                            echo "Moving the result ROC curves to the results folder..."
                            cp "detectionsContROC.png" "results-default/detectionsContROC_${cascade}_${scaleFactor}_${minNeighbor}_${minSizeX}_${minSizeY}_${maxSizeX}_${maxSizeY}.png"
                            cp "detectionsDiscROC.png" "results-default/detectionsDiscROC_${cascade}_${scaleFactor}_${minNeighbor}_${minSizeX}_${minSizeY}_${maxSizeX}_${maxSizeY}.png"
                            cp  "detectionsContROC.txt" "results-default/detectionsContROC_${cascade}_${scaleFactor}_${minNeighbor}_${minSizeX}_${minSizeY}_${maxSizeX}_${maxSizeY}.txt"
                            cp "detectionsDiscROC.txt" "results-default/detectionsDiscROC_${cascade}_${scaleFactor}_${minNeighbor}_${minSizeX}_${minSizeY}_${maxSizeX}_${maxSizeY}.txt"

                            # Calculate the number of iterations left
                            numOfIterations=$((numOfIterations-1))
                            echo "Number of iterations left: $numOfIterations"
                        
                        done
                    done
                done
            done
        done
    done
done