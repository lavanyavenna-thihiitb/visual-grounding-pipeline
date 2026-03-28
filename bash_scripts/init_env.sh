#!/usr/bin/env bash
set -e

# =============================================================================
# Python Environment Initialization Script
# =============================================================================
# This script runs INSIDE the Docker container.
# It receives the environment NAME (not full path) from run_pipeline.sh.
# The /environments directory is mounted from the host.
#
# Usage: init_env.sh <env_name> [deps_dir]
# Example: init_env.sh vision_ingestion_engine_env /fsxvision/user/Layout-Data-Engine/dependencies
#
# This will create/activate: /environments/<env_name>

if [ -z "$1" ]; then
  echo "❌ Error: No environment name provided"
  echo "Usage: $0 <env_name> [deps_dir]"
  echo "Example: $0 vision_ingestion_engine_env /fsxvision/user/Layout-Data-Engine/dependencies"
  exit 1
fi

ENV_NAME="$1"
ENV_PATH="/Environments/${ENV_NAME}"
# DEPS_DIR="${2:-/code/dependencies}"  # Use provided path or default

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Python Environment Initialization (Inside Container)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Environment name: ${ENV_NAME}"
echo "Container path:   ${ENV_PATH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

if [ -d "${ENV_PATH}" ]; then
  echo "🔹 Environment already exists — activating..."
else
  echo "🟢 Environment not found — creating new venv..."
  python3 -m venv "${ENV_PATH}"
  echo "✅ Created virtual environment at ${ENV_PATH}"

  # Optional: install base packages
  "${ENV_PATH}/bin/pip" install --upgrade pip setuptools wheel
  "${ENV_PATH}/bin/pip" install psutil nvitop pydantic ipython
  "${ENV_PATH}/bin/pip" install beautifulsoup4
  "${ENV_PATH}/bin/pip" install surya-ocr
  "${ENV_PATH}/bin/pip" install pillow
  "${ENV_PATH}/bin/pip" install accelerate
  "${ENV_PATH}/bin/pip" install transformers=4.51.0
  "${ENV_PATH}/bin/pip" install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126 
  "${ENV_PATH}/bin/pip" install vllm
  "${ENV_PATH}/bin/pip" install flash-attn --no-build-isolation
  "${ENV_PATH}/bin/pip" install doclayout-yolo shapely
fi

echo
echo "🚀 Activating environment..."
# shellcheck disable=SC1090
source "${ENV_PATH}/bin/activate"


# echo
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo "Installing dependencies from source (editable mode)"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# # echo "Dependencies directory: ${DEPS_DIR}"
# echo

# mkdir -p "$DEPS_DIR"

# Array of repository URLs
# REPOS=(
#   "git@github-tihiitb:sriharib128/StageWeaver.git"
#   "git@github-tihiitb:sriharib128/Vision-Ingestion-Engine.git"
#   "git@github-tihiitb:sriharib128/layout-qwen-zero-shot.git"
# )

# Install each repository
# for REPO_URL in "${REPOS[@]}"; do
#   REPO_NAME=$(basename "$REPO_URL" .git)

#   echo "🔹 Setting up ${REPO_NAME}..."

#   # Step 1: Clone if not exists
#   if [ ! -d "$DEPS_DIR/$REPO_NAME" ]; then
#     echo "  Cloning ${REPO_NAME}..."
#     if ! git clone "$REPO_URL" "$DEPS_DIR/$REPO_NAME"; then
#       echo "❌ Error: Failed to clone ${REPO_NAME}"
#       echo "Please manually clone ${REPO_URL} inside $DEPS_DIR"
#       echo "Then rerun this script."
#       exit 1
#     fi
#   else
#     echo "  ${REPO_NAME} already cloned"
#   fi

#   # Step 2: Check if package is installed, install if not
#   echo "  Checking installation status..."
#   # Check if there's an editable install from this directory
#   if pip list -v 2>/dev/null | grep -q "$DEPS_DIR/$REPO_NAME"; then
#     echo "  ${REPO_NAME} already installed in editable mode — skipping"
#   else
#     echo "  Installing in editable mode..."
#     if ! pip install -e "$DEPS_DIR/$REPO_NAME"; then
#       echo "❌ Error: Installation failed for ${REPO_NAME}"
#       echo "Please either:"
#       echo "  1. Install manually: pip install -e $DEPS_DIR/$REPO_NAME"
#       echo "  2. Fix the error in $DEPS_DIR/$REPO_NAME"
#       echo "     Then rerun this script to clone and install again"
#       exit 1
#     fi
#   fi
# done

# echo "✅ Dependencies installed in editable mode"
# echo

# Add auto-activation to .bashrc if not already present
# BASHRC="$HOME/.bashrc"
# ACTIVATION_LINE="conda deactivate && source ${ENV_PATH}/bin/activate"

if [ -f "$BASHRC" ]; then
  if ! grep -qF "$ACTIVATION_LINE" "$BASHRC"; then
    echo "" >> "$BASHRC"
    echo "# Auto-activate Python environment" >> "$BASHRC"
    echo "$ACTIVATION_LINE" >> "$BASHRC"
    echo "✅ Added auto-activation to ~/.bashrc"
  else
    echo "ℹ️  Auto-activation already in ~/.bashrc"
  fi
else
  echo "# Auto-activate Python environment" > "$BASHRC"
  echo "$ACTIVATION_LINE" >> "$BASHRC"
  echo "✅ Created ~/.bashrc with auto-activation"
fi

echo "👍 Done. Active Python: $(which python)"
echo "   Env path: ${ENV_PATH}"
echo "   Future docker exec sessions will auto-activate this environment"
