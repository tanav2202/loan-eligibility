FROM jupyter/minimal-notebook:latest

USER root

USER ${NB_UID}

# Copy all lock files and environment file
COPY conda-linux-64.lock /tmp/conda-linux-64.lock
COPY conda-linux-aarch64.lock /tmp/conda-linux-aarch64.lock
COPY environment.yml /tmp/environment.yml

# Auto-detect platform and install from correct lock file
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
    echo "Installing for linux-64 (x86_64)..." && \
    mamba update --file /tmp/conda-linux-64.lock --yes; \
    elif [ "$ARCH" = "aarch64" ]; then \
    echo "Installing for linux-aarch64 (ARM)..." && \
    mamba update --file /tmp/conda-linux-aarch64.lock --yes; \
    else \
    echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    conda clean --all -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

USER ${NB_UID}

# Set working directory
WORKDIR /home/${NB_USER}

# Default command
CMD ["start-notebook.sh"]
