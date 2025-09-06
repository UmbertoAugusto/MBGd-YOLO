#!/bin/bash

cat << "EOF"
  __  __ ____   _____ _____  
 |  \/  |  _ \ / ____|  __ \ 
 | \  / | |_) | |  __| |  | |
 | |\/| |  _ <| | |_ | |  | |
 | |  | | |_) | |__| | |__| |
 |_|  |_|____/ \_____|_____/ 
                             
                             
EOF
echo " --- Welcome to MBGd workflow ! ---"
echo " content objective: change Faster-RCNN for YOLO "

# Load configurations from config.yaml
CONFIG_FILE="configs/config.yaml"

# Path to Local yq (linux)
YQ_LOCAL="./yq"

# Verifica se o yq local existe
if [ ! -f "$YQ_LOCAL" ]; then
    echo "yq not found. Downloading yq..."
    wget https://github.com/mikefarah/yq/releases/download/v4.35.2/yq_linux_amd64 -O yq
    chmod +x yq
fi

# Extract values from config.yaml usando yq local
OBJ=$($YQ_LOCAL e '.OBJECT' $CONFIG_FILE)
FOLDS=$($YQ_LOCAL e '.NUM_FOLDS' $CONFIG_FILE)

# START MOSQUITOES WORKFLOW
echo "Starting training workflow for ${OBJ} detection, ${FOLDS} folds."
echo "Current date and time: $(date +"%Y-%m-%d %H:%M:%S")"
echo ""
SECONDS=0

for ((fold=1; fold<=FOLDS; fold++)); do

    echo "Current date and time: $(date +"%Y-%m-%d %H:%M:%S")"
    duration=$SECONDS
    echo "Time spent: $((duration / 3600)) hours, $(((duration / 60) % 60)) minutes and $((duration % 60)) seconds"
    echo ""
    echo "Running train_and_val.py for fold $fold..."
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python codes/train_and_val.py --config-file $CONFIG_FILE --object "${OBJ}" --fold "$fold"
    echo ""
    echo "Current date and time: $(date +"%Y-%m-%d %H:%M:%S")"
    duration=$SECONDS
    echo "Time spent: $((duration / 3600)) hours, $(((duration / 60) % 60)) minutes and $((duration % 60)) seconds"
    echo ""
done

# FINISH MOSQUITOES WORKFLOW
echo ""
echo "Current date and time: $(date +"%Y-%m-%d %H:%M:%S")"
duration=$SECONDS
echo "Time spent: $((duration / 3600)) hours, $(((duration / 60) % 60)) minutes and $((duration % 60)) seconds"
echo ""
echo "Training workflow completed."