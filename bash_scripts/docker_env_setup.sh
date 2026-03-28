#!/bin/bash

# RUN as: bash docker_env_setup.sh

USER_NAME="lavanya.venna"  # Docker user name (must match Dockerfile user)
IMAGE_NAME="visual-grounding-engine"
CONTAINER_NAME="visual-grounding-engine-${USER_NAME}"

# Python Environment Path (HOST path - full path on the server)
# This will be split into: Directory (for Docker mount) and Name (for init_env.sh inside container)
ENV_PATH="/home/${USER_NAME}/Environments/visual-grounding-env"

# Option 2: Shared environment for all users (uncomment to use)
# ENV_PATH="/home/shared_environments/layout_data_engine_env"

# Extract directory and environment name from ENV_PATH
ENVIRONMENT_MOUNT="$(dirname "$ENV_PATH")"  # e.g., /home/user/environments
ENV_NAME="$(basename "$ENV_PATH")"          # e.g., layout_data_engine_env

# Mount Paths
#/home/lavanya.venna/layout_data_engine/Layout-Data-Engine/bash_scripts/docker_env_setup.sh
CODE_MOUNT="/home/${USER_NAME}/visual-grounding-pipeline"
DATA_MOUNT="/home" 

# Dependencies Directory (same path in host and container)
# DEPS_DIR="/home/${USER_NAME}/layout_data_engine/Layout-Data-Engine/dependencies"

# Cache Directories (using fast NVMe storage). We can maybe setup a common cache later.
HF_CACHE_HOST="/opt/dlami/nvme/${USER_NAME}/hf_cache"
TMP_CACHE="/opt/dlami/nvme/${USER_NAME}/tmp"
VLLM_CACHE="/opt/dlami/nvme/${USER_NAME}/vllm_cache"
TRITON_CACHE="/opt/dlami/nvme/${USER_NAME}/triton_cache"



# --- Initialize Docker Container and Environment ---
bash init_docker.sh \
    "$USER_NAME" \
    "$IMAGE_NAME" \
    "$CONTAINER_NAME" \
    "$CODE_MOUNT" \
    "$DATA_MOUNT" \
    "$HF_CACHE_HOST" \
    "$TRITON_CACHE" \
    "$TMP_CACHE" \
    "$VLLM_CACHE" \
    "$ENVIRONMENT_MOUNT" 
    # "$DEPS_DIR"

# --- Initialize Environment inside Docker container ---
docker exec -i $CONTAINER_NAME /bin/bash << EOF
cd /code/bash_scripts
source init_env.sh "$ENV_NAME"
EOF

echo "Docker container and environment setup completed."
