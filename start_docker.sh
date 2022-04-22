#!/usr/bin/env bash
set -u

no_proxy="localhost,127.0.0.1,127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
http_proxy="http://proxy.ikim.uk-essen.de:3128"

# Configure the mountpoints so that the shared datadir is visible from the
# user's work directory.
WORKDIR_HOST="/local/work/nvidia-workshop-2022-04/user/$LOGNAME"
DATADIR_HOST="/local/work/nvidia-workshop-2022-04/dataset"
WORKDIR_CONTAINER="/workspace"
DATADIR_CONTAINER="$WORKDIR_CONTAINER/dataset"

# Export the repository working copy into the work directory.
REPODIR="$(dirname "$0")"
mkdir -p "$WORKDIR_HOST/NVIDIA_IKIM_Workshop" \
    && cp -aR "$REPODIR"/* "$WORKDIR_HOST/NVIDIA_IKIM_Workshop/"

# Start the container.
docker run --rm -it \
    --gpus all \
    --shm-size=4g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --user root \
	--workdir="$WORKDIR_CONTAINER" \
	--env WORKDIR="$WORKDIR_CONTAINER" \
	--env DATADIR="$DATADIR_CONTAINER" \
	--env no_proxy="$no_proxy" \
	--env NO_PROXY="$no_proxy" \
	--env http_proxy="$http_proxy" \
	--env https_proxy="$http_proxy" \
	--env HTTP_PROXY="$http_proxy" \
	--env HTTPS_PROXY="$http_proxy" \
    -v "$WORKDIR_HOST":"$WORKDIR_CONTAINER" \
    -v "$DATADIR_HOST":"$DATADIR_CONTAINER":ro \
    --network=host \
    --name "essen_workshop_$LOGNAME" \
    projectmonai/monai:latest
