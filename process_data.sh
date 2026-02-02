#!/bin/bash

# Directory containing datasets
DATA_DIR="data/datasets"

# Declare an associative array with datasets and their corresponding Jupyter notebooks
declare -A DATASETS
DATASETS=( ["TwoMoons"]="data/process_data_TwoMoons.ipynb"
           ["EMNIST"]="data/process_data_EMNIST.ipynb"
           ["CIFAR10"]="data/process_data_CIFAR10.ipynb"
           ["ANIMALS10"]="data/process_data_ANIMALS10.ipynb"
           ["tiny-imagenet-200"]="data/process_data_TinyImageNet.ipynb"
           ["MTSD"]="data/process_data_MTSD.ipynb"
            )

# Array to store found datasets
FOUND_DATASETS=("TwoMoons") # add TwoMoons here as a dataset, as the data for it is generated directly in the script.

# Scan for datasets in the data directory
echo "Scanning for datasets in $DATA_DIR..."

echo "Add one of the following datasets to the folder data/datasets: EMNIST, CIFAR10, ANIMALS10, tiny-imagenet-200, MTSD (proprietary)"

for dataset in "${!DATASETS[@]}"; do
    if [ -d "$DATA_DIR/$dataset" ]; then
        FOUND_DATASETS+=("$dataset")
    fi
done

# Check if any datasets were found
if [ ${#FOUND_DATASETS[@]} -eq 0 ]; then
    echo "No datasets found in $DATA_DIR."
    exit 1
fi

# Print found datasets
echo "Found the following datasets (For TwoMoons the data is generated in the script):"
for dataset in "${FOUND_DATASETS[@]}"; do
    echo "- $dataset"
done

echo "Make sure the dimensions are configured as desired in the notebook data/dimensionality_reduction.ipynb"
# Ask the user whether to proceed
# read -p "Do you want to execute the data processing pipeline? (y/n): " RESPONSE

# if [[ "$RESPONSE" != "y" && "$RESPONSE" != "Y" ]]; then
#     echo "Aborting execution."
#     exit 1
# fi


jupyter nbconvert --to notebook --execute --allow-errors --inplace "data/process_data_TwoMoons.ipynb"
jupyter nbconvert --to notebook --execute --allow-errors --inplace "data/dimensionality_reduction.ipynb"
jupyter nbconvert --to notebook --execute --allow-errors --inplace "data/prob_label_generation.ipynb"

echo "Notebook execution completed."
