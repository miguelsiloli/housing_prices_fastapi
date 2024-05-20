#!/bin/bash

# Step 1: Define paths to the Python scripts
PREPROCESS_SCRIPT="src/data_preprocessing.py"
TRAIN_SCRIPT="src/model_training.py"

# Step 2: Run the preprocessing script
echo "Running preprocessing script..."
python $PREPROCESS_SCRIPT

# Step 3: Run the training script
echo "Running training script..."
python $TRAIN_SCRIPT

echo "Scripts executed successfully."
