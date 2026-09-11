# Shared image recipe for the 4 track competitions. The only per-track
# difference is which benchmark is embedded, selected at build time:
#
#     docker build -f tools/Dockerfile --build-arg TRACK=sleep_onset \
#         -t <registry>/neural-compet-sleep_onset:v1 .
#
# The same image serves the Codabench workers and the participants:
#
#     # download a public dataset into a host folder (organizers: the
#     # folder later bind-mounted as /data for the scoring runs)
#     docker run -v <host-data>:/data <image> \
#         benchopt prepare "$COMPET_BENCHMARK_DIR" -d Sleep-EDF
#
#     # reproduce the platform evaluation on a submission folder
#     docker run --gpus all -v <host-data>:/data \
#         -v <submission>:/app/ingested_program -v <results>:/app/output \
#         <image> python3 /compet/ingestion_program/ingestion.py
# Python 3.12 base (neuralset needs >= 3.12; pytorch/pytorch images ship
# 3.11). torch 2.6 comes from pip — the cu124 wheels bundle the CUDA runtime,
# so GPU works through the nvidia container runtime without a CUDA base.
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# All Python dependencies a submission may import (this is the participants'
# dependency contract: no install happens at submission time).
COPY requirements.txt /tmp/requirements.txt
RUN pip install -U pip && pip install --no-cache-dir -r /tmp/requirements.txt

# The embedded track: benchmark + sibling compet_core (found by the
# benchmark's benchmark_utils parent walk) + the shared programs.
ARG TRACK
COPY compet_core /compet/compet_core
COPY tracks/${TRACK} /compet/benchmark
COPY codabench/ingestion_program /compet/ingestion_program
COPY codabench/scoring_program /compet/scoring_program

# Default phase config so ingestion works out of the box (dev phase);
# Codabench mounts the live phase's input_data over /app/input_data.
COPY codabench/phases/dev/${TRACK} /app/input_data

# All phase configs, addressable by name — used to stage /data on the
# workers from the same yaml that drives the run:
#     docker run -v /data:/data <image> sh -c \
#         'benchopt prepare "$COMPET_BENCHMARK_DIR" \
#              --config /compet/phases/dev/$COMPET_TRACK/config.yaml'
COPY codabench/phases /compet/phases

# Data is bind-mounted at /data (host-downloaded via `benchopt prepare`).
ENV COMPET_BENCHMARK_DIR=/compet/benchmark \
    COMPET_TRACK=${TRACK} \
    BENCHOPT_DATA_HOME=/data
VOLUME /data

# Fail the build if the embedded benchmark does not load.
RUN benchopt info "$COMPET_BENCHMARK_DIR"
