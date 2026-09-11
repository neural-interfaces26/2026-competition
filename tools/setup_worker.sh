#!/usr/bin/env bash
# Provision a Codabench compute worker for the 4 track competitions.
#
# Run as root on a GPU VM with /data mounted (staging can take hours — see
# the README: launch with nohup):
#
#     setup_worker.sh [config_file] [track ...]
#
# config_file  root-only environment file (default: /etc/codabench-worker.env)
#   Required: BROKER_URL      the Codabench queue broker.
#   Optional: BROKER_USE_SSL, CODALAB_IGNORE_CLEANUP_STEP, WORKER_IMAGE,
#             WORKER_NAME     worker settings (same as upstream Codabench);
#             REGISTRY, TAG   where the track images live (default:
#                             tommoral / v1, matching tools/build_images.sh);
#             COMPET_PHASE    competition phase to stage /data for. Mandatory
#                             when the images ship several phases; defaults
#                             to the only one otherwise.
# track ...    tracks to stage (default: all 4). Restricting them is handy
#              for testing, e.g. `setup_worker.sh my.env emg_pose`.
#
# The worker is shared across the tracks: the script pulls the selected
# tracks' images (deterministic name: $REGISTRY/neural-compet-<track>:$TAG)
# and stages their datasets in /data by running `benchopt prepare` on the
# phase's config.yaml baked in each image — the same file that drives the
# ingestion runs, so staging and evaluation cannot drift. `benchopt prepare`
# is idempotent: re-running this script only re-validates.
set -exuo pipefail

ALL_TRACKS=(bci_decoding emg_pose image_decoding sleep_onset)

config_file="${1:-/etc/codabench-worker.env}"
shift $(( $# > 0 ? 1 : 0 ))
if [[ $# -gt 0 ]]; then TRACKS=("$@"); else TRACKS=("${ALL_TRACKS[@]}"); fi
for track in "${TRACKS[@]}"; do
  [[ " ${ALL_TRACKS[*]} " == *" $track "* ]] || {
    echo "Unknown track '$track'. Tracks: ${ALL_TRACKS[*]}" >&2; exit 1; }
done

if [[ ! -r "$config_file" ]]; then
  echo "Configuration file not readable: $config_file" >&2
  exit 1
fi

set -a
source "$config_file"
set +a

: "${BROKER_URL:?BROKER_URL must be set in $config_file}"
: "${BROKER_USE_SSL:=true}"
: "${CODALAB_IGNORE_CLEANUP_STEP:=false}"
: "${WORKER_IMAGE:=codalab/codabench-compute-worker:latest}"
: "${WORKER_NAME:=eeg-competition}"
: "${REGISTRY:=tommoral}"
: "${TAG:=v1}"
: "${COMPET_PHASE:=}"

nvidia-smi
systemctl enable --now docker
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

install -d -m 0755 /opt/codabench-worker /codabench /data

# --- Stage the competition data in /data, before the worker goes live -----
image() { echo "$REGISTRY/neural-compet-$1:$TAG"; }

for track in "${TRACKS[@]}"; do
  docker pull "$(image "$track")"
done

# Resolve the phase against the phases shipped in the images (a phase dir
# holding a config for the track). Mandatory when several are available.
mapfile -t phases < <(docker run --rm "$(image "${TRACKS[0]}")" sh -c \
  'for d in /compet/phases/*/; do
     [ -f "$d$COMPET_TRACK/config.yaml" ] && basename "$d"
   done')
if [[ -z "$COMPET_PHASE" ]]; then
  if [[ ${#phases[@]} -eq 1 ]]; then
    COMPET_PHASE=${phases[0]}
  else
    echo "COMPET_PHASE must be set in $config_file." \
         "Available phases: ${phases[*]}" >&2
    exit 1
  fi
elif [[ ! " ${phases[*]} " == *" $COMPET_PHASE "* ]]; then
  echo "Unknown phase '$COMPET_PHASE'. Available phases: ${phases[*]}" >&2
  exit 1
fi

# `--gpus` covers image_decoding's one-time DINOv2 embedding pass; harmless
# for the other tracks.
for track in "${TRACKS[@]}"; do
  docker run --rm --gpus all -v /data:/data -e PHASE="$COMPET_PHASE" \
    "$(image "$track")" sh -c 'benchopt prepare "$COMPET_BENCHMARK_DIR" \
        --config "/compet/phases/$PHASE/$COMPET_TRACK/config.yaml"'
done
# ---------------------------------------------------------------------------

cat >/opt/codabench-worker/.env <<EOF
BROKER_URL=$BROKER_URL
BROKER_USE_SSL=$BROKER_USE_SSL
CODALAB_IGNORE_CLEANUP_STEP=$CODALAB_IGNORE_CLEANUP_STEP
HOST_DIRECTORY=/codabench
USE_GPU=true
EOF

cat >/opt/codabench-worker/compose.yaml <<EOF
services:
  worker:
    image: $WORKER_IMAGE
    container_name: compute_worker
    hostname: $WORKER_NAME
    volumes:
      - /codabench:/codabench
      - /var/run/docker.sock:/var/run/docker.sock
      - /data:/data
    env_file:
      - .env
    restart: unless-stopped
    logging:
      options:
        max-size: 50m
        max-file: 3
EOF

docker compose --project-directory /opt/codabench-worker pull
docker compose --project-directory /opt/codabench-worker up -d

# Nightly cleanup; keep the last 48 h so the track images and their shared
# dependency layer are not re-pulled on every submission.
cat >/etc/cron.daily/codabench-docker-prune <<'EOF'
#!/bin/sh
docker system prune -af --filter "until=48h"
EOF
chmod 0755 /etc/cron.daily/codabench-docker-prune
