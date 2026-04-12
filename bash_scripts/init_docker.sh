#!/bin/bash
set -euo pipefail

# =============================================================================
# Docker Container Initialization Script
# =============================================================================
# This script accepts all configuration as arguments from run_pipeline.sh
# DO NOT hardcode values here - all config should come from run_pipeline.sh

# Accept parameters from run_pipeline.sh
USER_NAME="$1"
IMAGE_NAME="$2"
CONTAINER_NAME="$3"
CODE_MOUNT="$4"
DATA_MOUNT="$5"
HF_CACHE_HOST="$6"
TRITON_CACHE="$7"
TMP_CACHE="$8"
VLLM_CACHE="$9"
ENVIRONMENT_MOUNT="${10}"
# DEPS_DIR="${11}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Docker Container Initialization"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "User:       $USER_NAME"
echo "Image:      $IMAGE_NAME"
echo "Container:  $CONTAINER_NAME"
echo "Code Mount: $CODE_MOUNT"
echo "HF Cache:   $HF_CACHE_HOST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

docker build --no-cache \
    --build-arg UID="$(id -u ${USER_NAME})" \
    --build-arg GID="$(id -g ${USER_NAME})" \
    --build-arg USER_NAME="${USER_NAME}" \
    -t "$IMAGE_NAME" .

# ---- Build image if not exists ----
# if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
#   echo "Building image $IMAGE_NAME..."
#   docker build \
#     --build-arg UID="$(id -u ${USER_NAME})" \
#     --build-arg GID="$(id -g ${USER_NAME})" \
#     --build-arg USER_NAME="${USER_NAME}" \
#     -t "$IMAGE_NAME" .
# else
#   echo "Image $IMAGE_NAME already exists."
# fi

# ---- Ensure cache dirs exist on host ----
mkdir -p "$HF_CACHE_HOST"/{hub,transformers,xdg,datasets}
mkdir -p "$TRITON_CACHE"
mkdir -p "$TMP_CACHE"
mkdir -p "$VLLM_CACHE"/torch_inductor
# mkdir -p "$DEPS_DIR"

# ---- Create or Start container ----
if docker ps -aq --filter "name=^/${CONTAINER_NAME}$" | grep -q .; then
  echo "Container $CONTAINER_NAME already exists — starting..."
  docker start "$CONTAINER_NAME" >/dev/null
else
  echo "Creating container $CONTAINER_NAME..."
  docker run -dit --gpus all \
    --network=host \
    --shm-size=64g \
    \
    -v "$DATA_MOUNT":"$DATA_MOUNT" \
    -v "$CODE_MOUNT":/code \
    \
    -v "$HF_CACHE_HOST":/hf_cache \
    \
    -v "$TRITON_CACHE":/triton_cache \
    -v "$TMP_CACHE":/tmp \
    -v "$VLLM_CACHE":/vllm_cache \
    -v "$ENVIRONMENT_MOUNT":/Environments \
    -v /fsxvision_new:/fsxvision_new \
    -v /fsx/lavanya.venna/.ssh:/home/lavanya.venna/.ssh \
    \
    -e HF_HOME=/hf_cache \
    -e HF_HUB_CACHE=/hf_cache/hub \
    -e HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    -e TRANSFORMERS_CACHE=/hf_cache/transformers \
    -e XDG_CACHE_HOME=/hf_cache/xdg \
    -e HF_DATASETS_CACHE=/hf_cache/datasets \
    \
    -e TRITON_CACHE_DIR="/triton_cache" \
    -e TMPDIR="/tmp" \
    \
    -e VLLM_CACHE_DIR="/vllm_cache" \
    -e TORCHINDUCTOR_CACHE_DIR="/vllm_cache/torch_inductor" \
    \
    --name "$CONTAINER_NAME" \
    -w /code \
    "$IMAGE_NAME" \
    bash
fi


# ---- Kaggle keys copy ---
# echo "Copying kaggle keys into container...."
# docker exec "$CONTAINER_NAME" mkdir -p "/home/$USER_NAME/.kaggle"
# docker cp "/home/$USER_NAME/.kaggle/kaggle.json" "$CONTAINER_NAME":/home/$USER_NAME/.kaggle/kaggle.json
# docker exec "$CONTAINER_NAME" chown -R $USER_NAME:$USER_NAME "/home/$USER_NAME/.kaggle"
# docker exec "$CONTAINER_NAME" chmod 600 "/home/$USER_NAME/.kaggle/kaggle.json"
# docker exec "$CONTAINER_NAME" bash -c \
# "grep -q 'local/bin' ~/.bashrc || echo 'export PATH=\$PATH:/home/$USER_NAME/.local/bin' >> ~/.bashrc"

# ---- Cleanup bad container-layer caches (CRITICAL) ----
docker exec "$CONTAINER_NAME" bash -lc "
  rm -rf /root/.triton /root/.cache/vllm /projects 2>/dev/null || true
"

echo ""
echo "🚀 READY — attach using:"
echo "docker exec -it $CONTAINER_NAME bash"
echo ""
echo "🔍 Verify inside container:"
echo "docker exec -it $CONTAINER_NAME bash -lc 'echo HF_HOME=\$HF_HOME; echo HF_HUB_CACHE=\$HF_HUB_CACHE; echo TRANSFORMERS_CACHE=\$TRANSFORMERS_CACHE; echo TRITON_CACHE_DIR=\$TRITON_CACHE_DIR; echo VLLM_CACHE_DIR=\$VLLM_CACHE_DIR; echo TMPDIR=\$TMPDIR; df -h / /tmp /fsxvision /hf_cache | tail -n +1'"
echo ""
