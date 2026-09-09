#!/usr/bin/env bash
# Build (and optionally push) the per-track competition Docker images.
# Run on a machine with docker; `docker login` first when pushing.
#
#     tools/build_images.sh                     # build all 4 tracks
#     tools/build_images.sh --push              # build + push to Docker Hub
#     tools/build_images.sh -t v2 sleep_onset   # one track, custom tag
#
# The requirements layer sits before ARG TRACK in the Dockerfile, so the 4
# builds share it (one pip install) and pushes only upload the small
# per-track COPY layers after the first image.
set -euo pipefail

cd "$(dirname "$0")/.."
REGISTRY=${REGISTRY:-tommoral}
TAG=v1 PUSH=0 TRACKS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag) TAG=$2; shift 2 ;;
        --push) PUSH=1; shift ;;
        *) TRACKS+=("$1"); shift ;;
    esac
done
[[ ${#TRACKS[@]} -gt 0 ]] \
    || TRACKS=(bci_decoding emg_pose image_decoding sleep_onset)

for track in "${TRACKS[@]}"; do
    image="$REGISTRY/neural-compet-$track:$TAG"
    echo "=== $image ==="
    docker build -f tools/Dockerfile --build-arg "TRACK=$track" \
        -t "$image" .
    [[ $PUSH == 1 ]] && docker push "$image"
done
