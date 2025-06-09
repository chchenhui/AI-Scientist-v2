#!/bin/bash

MODEL_WRITEUP="o4-mini-2025-04-16"
MODEL_CITATION="o4-mini-2025-04-16"
MODEL_AGG_PLOTS="o4-mini-2025-04-16"
NUM_CITE_ROUNDS=20


TASKS=(
    # your tasks here
    "iclr2025_dl4c"
    # "iclr2025_scope"
    # "iclr2025_scsl"
    # "iclr2025_verifai"
    # "iclr2025_wsl"
)


for task_name in "${TASKS[@]}"
do
    echo "Processing task: $task_name"
    python launch_scientist_bfts.py \
        --load_ideas "ai_scientist/ideas/o4-mini/${task_name}.json" \
        --model_writeup "$MODEL_WRITEUP" \
        --model_citation "$MODEL_CITATION" \
        --skip_review \
        --model_agg_plots "$MODEL_AGG_PLOTS" \
        --num_cite_rounds "$NUM_CITE_ROUNDS"

   
    echo "Finished processing task: $task_name"
    echo "--------------------------------------"
done

echo "All tasks processed."