#!/bin/bash
#
# Same dynamic task queue as run_experiments.sh, but packs multiple jobs
# onto each GPU concurrently instead of one job per GPU. Total concurrent
# capacity = len(gpus) * JOBS_PER_GPU.
#
# Tune JOBS_PER_GPU to how much headroom your GPUs actually have (model
# size, batch size, and BO search all add up) -- start low (2) and
# watch `nvidia-smi` before pushing it higher. Override without editing the
# file via:  JOBS_PER_GPU=3 ./scripts/run_experiments_multijob.sh

CONFIG_FILE="./configs/base.yaml"

# Define parameters
seeds=(0 13 44 123 456)
gpus=(0 1 2 3)            # GPU indices to schedule jobs on, e.g. (0 1 2 3)
alphas=(0.05 0.1 0.3)        # Dirichlet concentration(s), e.g. (0.05 0.1 0.3)
JOBS_PER_GPU=${JOBS_PER_GPU:-1}

# Format: "strategy[:fisher_type]" (fisher_type only for Fisher-based methods)
experiments=(
    "fedavg"
    "otfusion"
    "regmean"
    "dense"
    "fisher_merge:diagonal"
    "fedlpa"
    "fedfisher:diagonal"
    "fedfisher:kfac"
    "fedbens"
    "fedfaw:diagonal"
    "fedfaw_nofisher"   # FedFAW w/o Fisher (ablation)
)

datasets=("fashion_mnist" "cifar10" "cinic10" "svhn" "cifar100")

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

# ──────────────────────────────────────────────
# Build the slot -> GPU mapping. Each GPU gets JOBS_PER_GPU concurrent
# slots, e.g. gpus=(0 1) JOBS_PER_GPU=2 -> slot_gpus=(0 0 1 1)
# ──────────────────────────────────────────────
slot_gpus=()
for gpu_id in "${gpus[@]}"; do
    for ((j = 0; j < JOBS_PER_GPU; j++)); do
        slot_gpus+=("$gpu_id")
    done
done
NUM_SLOTS=${#slot_gpus[@]}

echo "=========================================="
echo "  Parallel GPU Task Queue (Multi-Job/GPU)"
echo "=========================================="
echo "Datasets     : ${datasets[@]}"
echo "Alphas       : ${alphas[@]}"
echo "Experiments  : ${experiments[@]}"
echo "Seeds        : ${seeds[@]}"
echo "GPUs         : ${gpus[@]}"
echo "Jobs per GPU : $JOBS_PER_GPU"
echo "Total slots  : $NUM_SLOTS"
echo "Total jobs   : $TOTAL_JOBS"
echo "Logs dir     : $LOG_DIR"
echo "=========================================="
echo ""

# ──────────────────────────────────────────────
# Task Queue Management
# ──────────────────────────────────────────────
job_count=0
declare -A slot_pid   # Map of slot_id -> PID of the running process
declare -A slot_task  # Map of slot_id -> Task description (for logging)

# Function to launch a task on a specific slot (and thus a specific GPU)
launch_task() {
    local task_str=$1
    local slot_id=$2
    local gpu_id=${slot_gpus[$slot_id]}

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
    echo "  [$(date '+%Y-%m-%d %H:%M:%S')] [$job_count/$TOTAL_JOBS - ${progress}%] GPU $gpu_id (slot $slot_id) -> $STRATEGY (seed $seed, alpha $alpha)"

    # Run in background
    $CMD > "$LOG_NAME" 2>&1 &

    # Store PID
    slot_pid[$slot_id]=$!
    slot_task[$slot_id]="$STRATEGY-$seed-$alpha"
}

# 1. Initially fill all slots
for ((slot_id = 0; slot_id < NUM_SLOTS; slot_id++)); do
    if [ ${#tasks[@]} -gt 0 ]; then
        task="${tasks[0]}"
        tasks=("${tasks[@]:1}") # Remove first element
        launch_task "$task" "$slot_id"
        sleep 1 # Small stagger to prevent simultaneous resource spikes
    fi
done

# 2. As soon as any job finishes, find which slot it was on and refill it
while [ ${#tasks[@]} -gt 0 ]; do
    # Wait for the next process to finish
    wait -n

    # Find which slot became free
    for ((slot_id = 0; slot_id < NUM_SLOTS; slot_id++)); do
        pid=${slot_pid[$slot_id]}
        if ! kill -0 "$pid" 2>/dev/null; then
            # This slot is free!
            task="${tasks[0]}"
            tasks=("${tasks[@]:1}")
            launch_task "$task" "$slot_id"
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
