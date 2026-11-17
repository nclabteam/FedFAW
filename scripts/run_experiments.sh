#!/bin/bash

CONFIG_FILE="./configs/base.yaml"  # Path to the configuration file

# Define parameters
# seeds=(0 13 44 123 456)
seeds=(0 13 44)
gpus=(0)            # GPU indices to schedule jobs on, e.g. (0 1 2 3)
alphas=(0.1)        # Dirichlet concentration(s), e.g. (0.05 0.1 0.3)

# Format: "strategy[:fisher_type]" (fisher_type only for Fisher-based methods)
experiments=(
    "fedavg"
    "fedprox"
    "scaffold"
    "otfusion"
    "regmean"
    "pfnm"
    "dense"
    "fisher_merge:diagonal"
    "fedlpa"
    "fedfisher:diagonal"
    "fedfisher:kfac"
    "fedbens"
    "fedfaw:diagonal"
    "fedfaw_nofisher"   # FedFAW w/o Fisher (ablation)
)

datasets=("cifar10")   # e.g. ("fashion_mnist" "cifar10" "cinic10" "svhn" "cifar100")

# Create log directory
LOG_DIR="./out/logs"
mkdir -p "$LOG_DIR"

# ──────────────────────────────────────────────
# Generate all tasks upfront
# ──────────────────────────────────────────────
tasks=()
for dataset in "${datasets[@]}"; do
    for alpha in "${alphas[@]}"; do
        for exp in "${experiments[@]}"; do
            for seed in "${seeds[@]}"; do
                tasks+=("$dataset|$alpha|$exp|$seed")
            done
        done
    done
done

TOTAL_JOBS=${#tasks[@]}

echo "=========================================="
echo "    Parallel GPU Task Queue (Dynamic)"
echo "=========================================="
echo "Datasets   : ${datasets[@]}"
echo "Alphas     : ${alphas[@]}"
echo "Experiments: ${experiments[@]}"
echo "Seeds      : ${seeds[@]}"
echo "GPUs       : ${gpus[@]}"
echo "Total jobs : $TOTAL_JOBS"
echo "Logs dir   : $LOG_DIR"
echo "=========================================="
echo ""

# ──────────────────────────────────────────────
# Task Queue Management
# ──────────────────────────────────────────────
job_count=0
declare -A gpu_status  # Map of GPU_ID -> PID of the running process
declare -A gpu_task    # Map of GPU_ID -> Task description (for logging)

# Function to launch a task on a specific GPU
launch_task() {
    local task_str=$1
    local gpu_id=$2
    
    # Parse task string
    IFS='|' read -r dataset alpha exp seed <<< "$task_str"
    STRATEGY="${exp%%:*}"
    FISHER_TYPE="${exp#*:}"
    
    # Build command
    CMD="python main.py --config $CONFIG_FILE --strategy $STRATEGY --gpu $gpu_id --seed $seed --alpha $alpha --dataset $dataset"
    if [ -n "$FISHER_TYPE" ] && [ "$FISHER_TYPE" != "$STRATEGY" ]; then
        CMD="$CMD --fisher_type $FISHER_TYPE"
    fi
    
    # Log name
    LOG_NAME="${LOG_DIR}/${dataset}_${STRATEGY}_${FISHER_TYPE}_alpha${alpha}_seed${seed}.log"
    
    # Increment counter and log progress
    ((job_count++))
    progress=$(( job_count * 100 / TOTAL_JOBS ))
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] [$job_count/$TOTAL_JOBS - ${progress}%] GPU $gpu_id -> $STRATEGY (seed $seed, alpha $alpha)"
    
    # Run in background
    $CMD > "$LOG_NAME" 2>&1 &
    
    # Store PID
    gpu_status[$gpu_id]=$!
    gpu_task[$gpu_id]="$STRATEGY-$seed-$alpha"
}

# 1. Initially fill all GPUs
for gpu_id in "${gpus[@]}"; do
    if [ ${#tasks[@]} -gt 0 ]; then
        task="${tasks[0]}"
        tasks=("${tasks[@]:1}") # Remove first element
        launch_task "$task" "$gpu_id"
        sleep 2 # Small stagger to prevent simultaneous resource spikes
    fi
done

# 2. As soon as any job finishes, find which GPU it was on and refill it
while [ ${#tasks[@]} -gt 0 ]; do
    # Wait for the next process to finish
    wait -n
    
    # Find which GPU became free
    for gpu_id in "${gpus[@]}"; do
        pid=${gpu_status[$gpu_id]}
        if ! kill -0 "$pid" 2>/dev/null; then
            # This GPU is free!
            task="${tasks[0]}"
            tasks=("${tasks[@]:1}")
            launch_task "$task" "$gpu_id"
            break # Refill one and go back to wait -n
        fi
    done
done

# 3. Final wait for remaining jobs
echo "  >>> All tasks submitted. Waiting for final jobs to complete..."
wait

echo ""
echo "=========================================="
echo "  All $TOTAL_JOBS experiments completed!"
echo "=========================================="
