#!/bin/bash

# Directory containing JSON files
EXPERIMENTS_DIR="experiments"

# Check if the experiments directory exists
if [[ ! -d "$EXPERIMENTS_DIR" ]]; then
  echo "Error: Directory '$EXPERIMENTS_DIR' does not exist."
  exit 1
fi

# List available experiments (JSON files without the .json extension)
AVAILABLE_EXPERIMENTS=($(ls "$EXPERIMENTS_DIR"/*.json 2>/dev/null | xargs -n 1 basename | sed 's/.json$//'))

if [[ ${#AVAILABLE_EXPERIMENTS[@]} -eq 0 ]]; then
  echo "Error: No JSON files found in the directory '$EXPERIMENTS_DIR'."
  exit 1
fi

echo "Available experiments:"
for exp in "${AVAILABLE_EXPERIMENTS[@]}"; do
  echo "  - $exp"
done

# Read input strings
echo "Enter experiment names separated by space (or type 'all' to run all experiments):"
read -ra INPUT_EXPERIMENTS

# Handle "all" shortcut
if [[ "${INPUT_EXPERIMENTS[0]}" == "all" ]]; then
  INPUT_EXPERIMENTS=("${AVAILABLE_EXPERIMENTS[@]}")
fi

# Validate input
for input_exp in "${INPUT_EXPERIMENTS[@]}"; do
  if [[ ! " ${AVAILABLE_EXPERIMENTS[@]} " =~ " $input_exp " ]]; then
    echo "Error: '$input_exp' is not a valid experiment name."
    exit 1
  fi
done

# Execute Python script for each valid experiment
for input_exp in "${INPUT_EXPERIMENTS[@]}"; do
  echo "Running experiment: $input_exp"
  python run_experiment.py "$input_exp"
done

echo "All experiments completed."