# Use Nvidia Ubuntu 20 base (includes CUDA if a supported GPU is present)
# https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && usermod -a -G video user \ 
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

# Add the deadsnakes PPA to get the latest Python version
RUN sudo apt-get install -y software-properties-common \
  && sudo add-apt-repository ppa:deadsnakes/ppa

# Install dependencies
RUN sudo apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
  build-essential \
  curl \
  git \
  poppler-utils \
  python3 \
  python3.11 \
  python3-dev \
  python3-distutils \
  python3-venv \
  expat \
  rsync

# Install pip (we need the latest version not the standard Ubuntu version, to
# support modern wheels)
RUN sudo curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py

# Install poetry
RUN sudo curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Set python aliases
RUN sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# User the new user
USER $USERNAME
