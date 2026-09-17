#!/usr/bin/env bash
# Build (and optionally push) the competition Docker image.
# Run on a machine with docker; `docker login` first when pushing.
#
#     tools/build_image.sh              # build
#     tools/build_image.sh --push       # build + push to Docker Hub
#     tools/build_image.sh -t v2        # custom tag
#
# One image serves the 4 tracks and both phases: the benchmark and the phase
# config travel in the phase bundle, not in the image.
set -euo pipefail

cd "$(dirname "$0")/.."
REGISTRY=${REGISTRY:-tommoral}
TAG=v1 PUSH=0
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag) TAG=$2; shift 2 ;;
        --push) PUSH=1; shift ;;
        *) echo "unexpected argument: $1" >&2; exit 1 ;;
    esac
done

image="$REGISTRY/neural-compet:$TAG"
echo "=== $image ==="
docker build -f tools/Dockerfile -t "$image" .
[[ $PUSH == 1 ]] && docker push "$image"
