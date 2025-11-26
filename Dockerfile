FROM jupyter/minimal-notebook:latest

COPY conda-linux-64.lock /tmp/conda-linux-64.lock

RUN conda install --quiet --file /tmp/conda-linux-64.lock
RUN conda clean --all -y
RUN fix-permissions "${CONDA_DIR}"
RUN fix-permissions "/home/${NB_USER}"

