ARG ISAAC_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
FROM ${ISAAC_IMAGE}

ARG ISAAC_LAB_REF

SHELL ["/bin/bash", "-lc"]
USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN cd /isaac-sim && \
    test -n "${ISAAC_LAB_REF}" && \
    git clone https://github.com/isaac-sim/IsaacLab.git && \
    cd IsaacLab && \
    git checkout "${ISAAC_LAB_REF}" && \
    ./isaaclab.sh --install

WORKDIR /workspace/isaac-projects
