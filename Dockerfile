# ================================================================
# DreamerV3 PyTorch - Docker Container
# ================================================================
#
# BUILD:
#   docker build -t dreamerv3-torch:latest .
#
# RUN:
#   docker run -it --rm --gpus all --env-file .env dreamerv3-torch:latest
#
# ================================================================

# Base image with PyTorch and CUDA (includes Python 3.11)
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Additional environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone the repository
RUN git clone https://github.com/Min34r/dreamerv3-torch.git

# Change to repository directory
WORKDIR /workspace/dreamerv3-torch

# Copy .env file (not in git repo)
COPY .env .env

# Install requirements
RUN pip install -r requirements.txt

# Note: Comet ML environment variables should be passed at runtime using --env-file .env
# or -e flags. Do not hardcode secrets in the Dockerfile.

# Default command
CMD ["bash"]