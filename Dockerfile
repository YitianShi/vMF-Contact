# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

################################################################################
# Create a base stage for the application.
# We use the nvidia/cuda image as a base image for this so we can leverage the GPU for our application.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV TORCH_CUDA_ARCH_LIST="8.9"

################################################################################
# Create a stage for building/compiling the application.
RUN apt-get update && apt-get install -y \
    # Install any build dependencies here.
    python3 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install ninja

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu121/torch_stable.html

# Run the install.sh script
RUN mkdir /vmf
COPY . /vmf/
RUN chmod +x ./vmf/install.sh
RUN cd ./vmf/vmf_contact_main/openpoints/cpp/chamfer_dist && python3 setup.py install
RUN cd ./vmf/vmf_contact_main/openpoints/cpp/emd && python3 setup.py install
RUN cd ./vmf/vmf_contact_main/openpoints/cpp/pointnet2_batch && python3 setup.py install
RUN cd ./vmf/vmf_contact_main/openpoints/cpp/pointops && python3 setup.py install
# KEEPS FAILING
# RUN cd ./vmf/vmf_contact_main/openpoints/cpp/subsampling && python3 setup.py build_ext --inplace

COPY requirements.txt ./
RUN pip3 install -r requirements.txt


# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
# ARG UID=10001
# RUN adduser \
#     --disabled-password \
#     --gecos "" \
#     --home "/nonexistent" \
#     --shell "/sbin/nologin" \
#     --no-create-home \
#     --uid "${UID}" \
#     appuser
# USER appuser

WORKDIR /vmf

# Add a HEALTHCHECK instruction to ensure the container's health is monitored.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl -f http://localhost/ || exit 1
