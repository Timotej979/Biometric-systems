#!/bin/bash

######## CHOICE PICKER ########
# Get the values of environment variables
# Exit, OpenBR, GridSVC
choice=$CHOICE
# Sequential, Parallel (OpenBR), Face detection type (GridSVC) 
sub_choice1=$SUB_CHOICE1
# Display log (OpenBR), GridSVC kernel (GridSVC)
sub_choice2=$SUB_CHOICE2
# Display log (GridSVC)
sub_choice3=$SUB_CHOICE3
# Substract HOG features (GridSVC)
sub_choice4=$SUB_CHOICE4

# Handle if choices are not defined
if [ -z "$sub_choice3" ] || [ -z "$sub_choice4" ]; then
    sub_choice3=0
    sub_choice4=0
fi

# Print the choices
echo "Choice: $choice"
echo "Sub choice 1: $sub_choice1"
echo "Sub choice 2: $sub_choice2"
echo "Sub choice 3: $sub_choice3"
echo "Sub choice 4: $sub_choice4"

# Exit the tool
if [ "$choice" = "0" ]; then
    echo "Exiting the tool"
    exit 0

############################################################################################################################################################
# OpenBR
elif [ "$choice" = "1" ]; then

    # Sequential
    if [ "$sub_choice1" = "0" ]; then

        # Display log
        if [ "$sub_choice2" = "1" ]; then
            echo "Executing OpenBR sequentially with logging"
            python3 runOpenBR.py --br_path "br" --lfw_dataset_path "lfw-dataset" --pairs_file_path "lfwa-pairs/pairsDevTest.txt" --output_file_path "output-openbr" --display_log
        else
            echo "Executing OpenBR sequentially without logging"
            python3 runOpenBR.py --br_path "br" --lfw_dataset_path "lfw-dataset" --pairs_file_path "lfwa-pairs/pairsDevTest.txt" --output_file_path "output-openbr"
        fi

    # Parallel
    elif [ "$sub_choice1" = "1" ]; then
        # Display log
        if [ "$sub_choice2" = "1" ]; then
            echo "Executing OpenBR in parallel with logging"
            python3 runOpenBR.py --br_path "br" --lfw_dataset_path "lfw-dataset" --pairs_file_path "lfwa-pairs/pairsDevTest.txt" --output_file_path "output-openbr" --parallel --display_log
        # Do not display log
        else
            echo "Executing OpenBR in parallel without logging"
            python3 runOpenBR.py --br_path "br" --lfw_dataset_path "lfw-dataset" --pairs_file_path "lfwa-pairs/pairsDevTest.txt" --output_file_path "output-openbr" --parallel
        fi
    # Invalid choice
    else
        echo "Invalid choice"
        exit 1
    fi

############################################################################################################################################################
# GridSVC
elif [ "$choice" = "2" ]; then

    ############################################################################################################################################################
    # Manual cropping
    if [ "$sub_choice1" = "0" ]; then

        # Linear
        if [ "$sub_choice2" = "0" ]; then

            # Substract HOG features
            if [ "$sub_choice4" = "0" ]; then 

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with manual cropping, linear kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset" --pairs_test_file "lfwa-pairs/pairsDevTest.txt" --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc" --face_detection_type "crop" --svc_kernel "linear" --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with manual cropping and linear kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset" --pairs_test_file "lfwa-pairs/pairsDevTest.txt" --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc" --face_detection_type "crop" --svc_kernel "linear"
                fi
            
            else 

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with manual cropping, linear kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset" --pairs_test_file "lfwa-pairs/pairsDevTest.txt" --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc" --face_detection_type "crop" --svc_kernel "linear" --substract_hog_features --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with manual cropping and linear kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset" --pairs_test_file "lfwa-pairs/pairsDevTest.txt" --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc" --face_detection_type "crop" --svc_kernel "linear" --substract_hog_features
                fi

            fi

        # Polynomial
        elif [ "$sub_choice2" = "1" ]; then

            # Substract HOG features
            if [ "$sub_choice4" = "0" ]; then 

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with manual cropping, polynomial kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset" --pairs_test_file "lfwa-pairs/pairsDevTest.txt" --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc" --face_detection_type "crop" --svc_kernel "poly" --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with manual cropping and polynomial kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset" --pairs_test_file "lfwa-pairs/pairsDevTest.txt" --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc" --face_detection_type "crop" --svc_kernel "poly"
                fi

            else

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with manual cropping, polynomial kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset" --pairs_test_file "lfwa-pairs/pairsDevTest.txt" --pairs_train_file "lfwa-pairs/pairsDevTrain.txt"  --output_file_path "output-gridsvc" --face_detection_type "crop" --svc_kernel "poly" --substract_hog_features --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with manual cropping and polynomial kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset" --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc" --face_detection_type "crop" --svc_kernel "poly" --substract_hog_features
                fi

            fi

        # RBF
        elif [ "$sub_choice2" = "2" ]; then

            # Substract HOG features
            if [ "$sub_choice4" = "0" ]; then 

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with manual cropping, RBF kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset" --pairs_test_file "lfwa-pairs/pairsDevTest.txt" --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc" --face_detection_type "crop" --svc_kernel "rbf" --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with manual cropping and RBF kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset" --pairs_test_file "lfwa-pairs/pairsDevTest.txt" --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc" --face_detection_type "crop" --svc_kernel "rbf"
                fi

            else

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with manual cropping, RBF kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc"  --face_detection_type "crop" --svc_kernel "rbf" --substract_hog_features --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with manual cropping and RBF kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc"  --face_detection_type "crop" --svc_kernel "rbf" --substract_hog_features
                fi

            fi

        # Sigmoid
        elif [ "$sub_choice2" = "3" ]; then

            # Substract HOG features
            if [ "$sub_choice4" = "0" ]; then 

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with manual cropping, sigmoid kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset" --pairs_test_file "lfwa-pairs/pairsDevTest.txt" --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc" --face_detection_type "crop" --svc_kernel "sigmoid" --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with manual cropping and sigmoid kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset" --pairs_test_file "lfwa-pairs/pairsDevTest.txt" --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc" --face_detection_type "crop" --svc_kernel "sigmoid"
                fi

            else

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with manual cropping, sigmoid kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc"  --face_detection_type "crop" --svc_kernel "sigmoid" --substract_hog_features --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with manual cropping and sigmoid kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc"  --face_detection_type "crop" --svc_kernel "sigmoid" --substract_hog_features
                fi

            fi

        # Invalid choice
        else
            echo "Invalid choice"
            exit 1
        fi

    ############################################################################################################################################################
    # OpenCV haarcascade
    elif [ "$sub_choice1" = "1" ]; then

        # Linear
        if [ "$sub_choice2" = "0" ]; then

            # Substract HOG features
            if [ "$sub_choice4" = "0" ]; then 

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with OpenCV haarcascade, linear kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt"  --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "linear" --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with OpenCV haarcascade and linear kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt"  --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "linear"
                fi
            
            else 

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with OpenCV haarcascade, linear kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt"  --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "linear" --substract_hog_features --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with OpenCV haarcascade and linear kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt"  --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "linear" --substract_hog_features
                fi

            fi

        # Polynomial
        elif [ "$sub_choice2" = "1" ]; then

            # Substract HOG features
            if [ "$sub_choice4" = "0" ]; then 

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with OpenCV haarcascade, polynomial kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt"  --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "poly" --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with OpenCV haarcascade and polynomial kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt"  --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "poly"
                fi

            else

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with OpenCV haarcascade, polynomial kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt"  --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "poly" --substract_hog_features --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with OpenCV haarcascade and polynomial kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt"  --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "poly" --substract_hog_features
                fi

            fi

        # RBF
        elif [ "$sub_choice2" = "2" ]; then

            # Substract HOG features
            if [ "$sub_choice4" = "0" ]; then 

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with OpenCV haarcascade, RBF kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt"  --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "rbf" --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with OpenCV haarcascade and RBF kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt"  --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "rbf"
                fi

            else

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with OpenCV haarcascade, RBF kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt"  --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "rbf" --substract_hog_features --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with OpenCV haarcascade and RBF kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt"  --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "rbf" --substract_hog_features
                fi

            fi

        # Sigmoid
        elif [ "$sub_choice2" = "3" ]; then

            # Substract HOG features
            if [ "$sub_choice4" = "0" ]; then 

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with OpenCV haarcascade, sigmoid kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "sigmoid" --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with OpenCV haarcascade and sigmoid kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "sigmoid"
                fi

            else

                # Display log
                if [ "$sub_choice3" = "1" ]; then
                    echo "Executing GridSVC sequentially with OpenCV haarcascade, sigmoid kernel and logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "sigmoid" --substract_hog_features --display_log
                # Do not display log
                else
                    echo "Executing GridSVC sequentially with OpenCV haarcascade and sigmoid kernel without logging"
                    python3 runGridSVC.py --lfwa_dataset_path "lfwa-dataset"  --pairs_test_file "lfwa-pairs/pairsDevTest.txt"  --pairs_train_file "lfwa-pairs/pairsDevTrain.txt" --output_file_path "output-gridsvc"  --face_detection_type "haar" --svc_kernel "sigmoid" --substract_hog_features
                fi

            fi

        # Invalid choice
        else
            echo "Invalid choice"
            exit 1
        fi

    # Invalid choice
    else
        echo "Invalid choice"
        exit 1
    fi

############################################################################################################################################################
# Invalid choice
else
    echo "Invalid choice"
    exit 1
fi