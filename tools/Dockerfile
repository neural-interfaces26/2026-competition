# One image for the 4 track competitions and both phases: it carries the
# environment and the shared code, while the benchmark and the phase config
# travel together in the phase bundle mounted on /app/input_data.
#
#     docker build -f tools/Dockerfile -t <registry>/neural-compet:v1 .
#
# The same image serves the Codabench workers and the participants:
#
#     # stage a phase's data into a host folder (organizers: the folder
#     # later bind-mounted as /app/data for the scoring runs)
#     docker run -v <host-data>:/app/data -v <phase-bundle>:/app/input_data \
#         <image> benchopt prepare /app/input_data/benchmark \
#             --config /app/input_data/config.yaml
#
#     # reproduce the platform evaluation on a submission folder
#     docker run --gpus all -v <host-data>:/app/data \
#         -v <phase-bundle>:/app/input_data \
#         -v <submission>:/app/ingested_program -v <results>:/app/output \
#         <image> python3 /compet/ingestion_program/ingestion.py
# Python 3.12 base (neuralset needs >= 3.12; pytorch/pytorch images ship
# 3.11). torch comes from the PyTorch wheel index (see TORCH_CHANNEL below) —
# the wheels bundle the CUDA runtime, so GPU works through the nvidia
# container runtime without a CUDA base image.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# The torch trio from the official PyTorch wheel index with an explicit CUDA
# channel: PyPI's default silently tracks the newest CUDA (cu13 today), which
# needs very recent drivers; cu126 runs on any R525+ driver. All three come
# from the same index so their C++ ops match (a PyPI torchvision against a
# channel torch fails at import with `torchvision::nms does not exist`).
# Installed first so the portable pins in requirements.txt are satisfied.
ARG TORCH_CHANNEL=cu126
RUN pip install -U pip && pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/${TORCH_CHANNEL}

# All Python dependencies a submission may import (this is the participants'
# dependency contract: no install happens at submission time).
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY tools/neuralbench_config.json /compet/neuralbench_config.json

# Shared code only: the benchmark itself travels in the phase bundle, which
# is what makes one image serve the four tracks and both phases.
COPY benchmark_utils /compet/benchmark_utils
COPY codabench/ingestion_program /compet/ingestion_program
COPY codabench/scoring_program /compet/scoring_program

# Data is bind-mounted at /app/data — the path the Codabench compute worker
# mounts (read-only) in every submission container. The phase bundle lands
# on /app/input_data, benchmark included.
ENV BENCHOPT_DATA_HOME=/app/data \
    NEURALBENCH_CONFIG=/compet/neuralbench_config.json

# Hide upstream deprecations (MNE montages, ...) and noisy tqdm
ENV PYTHONWARNINGS=ignore::FutureWarning \
    TQDM_DISABLE=1
VOLUME /app/data

# Fail the build if the data stack is incoherent: at runtime a broken import
# (e.g. a torch/torchvision mismatch) only shows up as a silently
# "not installed" dataset.
RUN python -c "import neuralbench, braindecode, benchopt"
