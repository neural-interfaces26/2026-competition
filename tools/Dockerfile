# Step 1: Start from an official Docker image with desired base environment
# Good starting points are the official codalab images or
# pytorch images with CUDA support:
#    - Codalab: codalab/codalab-legacy:py39
#    - Codalab GPU: codalab/codalab-legacy:gpu310
#    - Pytorch: pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime
FROM codalab/codalab-legacy:py39

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Step 2: Install system-level dependencies (if any)
# e.g., git, wget, or common libraries for OpenCV like libgl1
RUN pip install -U pip

# Step 3: Copy and pre-install all Python dependencies
# This 'requirements.txt' file should list pandas, scikit-learn, timm, etc.
# Place it in the same directory as this Dockerfile.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
