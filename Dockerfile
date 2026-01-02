FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

USER root

# Install LaTeX and make in one layer
RUN apt-get update && apt-get install -y \
    lmodern \
    texlive \
    texlive-luatex \
    make && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Fix permissions
RUN fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Switch back to notebook user
USER ${NB_UID}

# Copy project files
COPY --chown=${NB_UID}:${NB_GID} . /home/${NB_USER}/work

# Set working directory
WORKDIR /home/${NB_USER}/work

# Update conda and install environment
RUN conda update -n base -c conda-forge conda && \
    conda env update --name base --file environment.yml && \
    conda clean --all -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8000", "--no-browser", "--allow-root"]