FROM nvcr.io/nvidia/pytorch:24.03-py3

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsm6 \
        libxext6 \
        poppler-utils \
        ca-certificates \
        curl \
        git \
        vim \
        tzdata \
        libgl1 \
        software-properties-common \
        libgl1-mesa-glx \
        htop \
        ncdu \
        tmux \
        python3.10-venv \
        sudo \
        tree \
        wget && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install Google Chrome
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && \
    apt-get install -y ./google-chrome-stable_current_amd64.deb && \
    rm google-chrome-stable_current_amd64.deb


# -----------------------------
# User creation (CRITICAL)
# -----------------------------
ARG UID
ARG GID
ARG USER_NAME

RUN if [ -z "$UID" ] || [ -z "$GID" ] || [ -z "$USER_NAME" ]; then \
      echo "UID, GID, USER_NAME must be provided" >&2 && exit 1; \
    fi && \
    groupadd -g ${GID} ${USER_NAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER_NAME} && \
    passwd -d ${USER_NAME} && \
    usermod -aG sudo ${USER_NAME}

# Create sudoers file as a separate step for clarity
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" | tee /etc/sudoers.d/${USER_NAME} && \
    chmod 0440 /etc/sudoers.d/${USER_NAME} && \
    visudo -c -f /etc/sudoers.d/${USER_NAME}

# -----------------------------
# Global python tools (root-only)
# -----------------------------
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install psutil nvitop

# -----------------------------
# Switch to non-root
# -----------------------------
USER ${USER_NAME}
WORKDIR /code

CMD ["/bin/bash"]