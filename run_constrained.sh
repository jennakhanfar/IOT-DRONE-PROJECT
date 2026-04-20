#!/bin/bash
# Run benchmarks inside Docker with real RAM constraints.
# Usage:
#   ./run_constrained.sh 256    mobilefacenet   vggface2
#   ./run_constrained.sh 512    arcface_r50     droneface
#   ./run_constrained.sh 256    all             vggface2
#
# Args:
#   $1 = RAM limit in MB (e.g. 256, 512)
#   $2 = model name or "all"
#   $3 = dataset type: vggface2 or droneface

RAM_MB=${1:-256}
MODEL=${2:-mobilefacenet}
DATASET_TYPE=${3:-vggface2}
CPU_FRACTION=${CPU_FRACTION:-0.15}
CPU_QUOTA=$(awk "BEGIN { printf \"%d\", ${CPU_FRACTION} * 100000 }")
RUN_TAG=${RUN_TAG:-docker_${RAM_MB}mb_cpu$(echo "$CPU_FRACTION" | tr '.' 'p')}

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Map dataset type to folder
if [ "$DATASET_TYPE" = "vggface2" ]; then
    DATASET_ROOT="archive (1)"
elif [ "$DATASET_TYPE" = "droneface" ]; then
    DATASET_ROOT="open_data_set"
else
    echo "Unknown dataset type: $DATASET_TYPE (use vggface2 or droneface)"
    exit 1
fi

IMAGE_NAME="bench-constrained"

# Build if needed
echo "=== Building Docker image ==="
docker build -f Dockerfile.bench -t "$IMAGE_NAME" "$PROJECT_DIR"

echo ""
echo "=== Running: $MODEL on $DATASET_TYPE with ${RAM_MB}MB RAM cap and ${CPU_FRACTION} CPU ==="
echo ""

# Pick the right mount path inside the container
if [ "$DATASET_TYPE" = "vggface2" ]; then
    CONTAINER_DATASET="/bench/dataset"
    HOST_DATASET="$PROJECT_DIR/archive (1)"
else
    CONTAINER_DATASET="/bench/dataset"
    HOST_DATASET="$PROJECT_DIR/open_data_set"
fi

DOCKER_ARGS=(
    --rm
    --memory="${RAM_MB}m"
    --memory-swap="${RAM_MB}m"
    --cpu-period=100000
    --cpu-quota="${CPU_QUOTA}"
    -e PYTHONUNBUFFERED=1
    -e BENCHMARK_FORCE_CONSTRAINED=1
    -e BENCHMARK_RAM_MB="${RAM_MB}"
    -e BENCHMARK_CPU_FRACTION="${CPU_FRACTION}"
    -e DOCKER_RAM_MB="${RAM_MB}"
    -e DOCKER_CPU_FRACTION="${CPU_FRACTION}"
    -e DOCKER_CPU_PERIOD=100000
    -e DOCKER_CPU_QUOTA="${CPU_QUOTA}"
    -v "$HOST_DATASET":/bench/dataset:ro
    -v "$PROJECT_DIR/benchmark_results":/bench/benchmark_results
    -v "$HOME/.insightface":/root/.insightface:ro
    "$IMAGE_NAME"
    python benchmark_recognizers.py
    --dataset-root "/bench/dataset"
    --dataset-type "$DATASET_TYPE"
    --constrained
    --run-tag "$RUN_TAG"
)

if [ "$MODEL" = "all" ]; then
    docker run "${DOCKER_ARGS[@]}" --all
else
    docker run "${DOCKER_ARGS[@]}" --model "$MODEL"
fi
