#!/usr/bin/env bash
set -u

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
    -v "$WORKDIR_HOST":"$WORKDIR_CONTAINER" \
    -v "$DATADIR_HOST":"$DATADIR_CONTAINER":ro \
    --network=host \
    --name "essen_workshop_$LOGNAME" \
    projectmonai/monai:latest
