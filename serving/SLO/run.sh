#!/bin/bash

# Define the output log directory
LOG_DIR="logs"
mkdir -p $LOG_DIR  # Ensure the log directory exists

# Define the different script versions and their corresponding labels
declare -A SCRIPTS=(
    ["serving_system.py"]="ours"
    # ["serving_system_N.py"]="NIRVANA"
    # ["serving_baseline.py"]="baseline"
)

# Define the request rates
RATES=(7)

# Loop through each script and each request rate
for SCRIPT_NAME in "${!SCRIPTS[@]}"; do
    LABEL="${SCRIPTS[$SCRIPT_NAME]}"  # Get the label associated with the script

    for RATE in "${RATES[@]}"; do
        echo "Running $SCRIPT_NAME with request rate: $RATE"
        
        # Run the script with the specified request rate and log the output
        python $SCRIPT_NAME --rate $RATE | tee "$LOG_DIR/request_rate_${RATE}_${LABEL}.log"
        
        echo "Finished run with request rate: $RATE for $LABEL"
        echo "--------------------------------"
    done

    echo "All runs for $LABEL completed."
done
