#!/bin/bash

NOTEBOOK=$1

# Execute the notebook
jupyter nbconvert --to notebook --execute --allow-errors --inplace "$NOTEBOOK"

echo "Notebook $NOTEBOOK executed."
