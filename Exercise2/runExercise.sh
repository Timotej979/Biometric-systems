#!/bin/bash

# Create the Text based CLI tool
echo "----------------------------------------------------------------------------------------------------------"
echo "------------------------------------ FACE RECOGNITION EVALUATION TOOL ------------------------------------"
echo "----------------------------------------------------------------------------------------------------------"

echo "Please choose what you want to do: "
echo "  (0)  Exit the tool"

echo "  (1)  Evaluate the LFW dataset OpenBR"

echo "  (2)  Evaluate the LFWA dataset GridSVC"
echo "----------------------------------------------------------------------------------------------------------"
# Read the choice variable
read -p "Enter your choice: " main_choice

# Check if the choice is 0
if [ $main_choice -eq 0 ]; then
    exit 0

elif [ $main_choice -eq 1 ]; then
    echo "----------------------------------------------------------------------------------------------------------"
    echo "Do you want to execute the evaluation sequentially or in parallel:"
    echo "  (0)  Sequentially"
    echo "  (1)  In parallel"
    echo "----------------------------------------------------------------------------------------------------------"

    # Read the choice variable
    read -p "Enter your choice: " sub_choice1

    echo "----------------------------------------------------------------------------------------------------------"
    echo "Do you want to display processing logs:"
    echo "  (0)  No"
    echo "  (1)  Yes"
    echo "----------------------------------------------------------------------------------------------------------"

    # Read the choice variable
    read -p "Enter your choice: " sub_choice2

elif [ $main_choice -eq 2 ]; then
    echo "----------------------------------------------------------------------------------------------------------"
    echo "Which face detection type do you want to use:"
    echo "  (0)  Manual cropping"
    echo "  (1)  OpenCV haarcascade"
    echo "----------------------------------------------------------------------------------------------------------"

    # Read the choice variable
    read -p "Enter your choice: " sub_choice1

    echo "----------------------------------------------------------------------------------------------------------"
    echo "Do you want to substract the HOG features for pairs:"
    echo "  (0)  No"
    echo "  (1)  Yes"
    echo "----------------------------------------------------------------------------------------------------------"

    # Read the choice variable
    read -p "Enter your choice: " sub_choice4

    echo "----------------------------------------------------------------------------------------------------------"
    echo "Which GridSVC kernel do you want to use:"
    echo "  (0)  Linear"
    echo "  (1)  Polynomial"
    echo "  (2)  RBF"
    echo "  (3)  Sigmoid"
    echo "----------------------------------------------------------------------------------------------------------"

    # Read the choice variable
    read -p "Enter your choice: " sub_choice2

    echo "----------------------------------------------------------------------------------------------------------"
    echo "Do you want to display processing logs:"
    echo "  (0)  No"
    echo "  (1)  Yes"
    echo "----------------------------------------------------------------------------------------------------------"

    # Read the choice variable
    read -p "Enter your choice: " sub_choice3

else
    echo "Invalid choice"
    exit 1
fi 

# Handle if choices are not defined
if [ -z "$sub_choice3" ] || [ -z "$sub_choice4" ]; then
    sub_choice3=0
    sub_choice4=0
fi

# If the choice is valid, export the variables
export MAIN_CHOICE=$main_choice
export SUB_CHOICE1=$sub_choice1
export SUB_CHOICE2=$sub_choice2
export SUB_CHOICE3=$sub_choice3
export SUB_CHOICE4=$sub_choice4

# Build the docker compose
docker compose build

# Run the docker compose
docker compose up