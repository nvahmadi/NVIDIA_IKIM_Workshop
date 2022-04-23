#!/usr/bin/env bash
set -u -o pipefail

no_proxy="localhost,127.0.0.1,127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
http_proxy="http://proxy.ikim.uk-essen.de:3128"

# Configure the mountpoints so that the shared datadir is visible from the
# user's work directory.
WORKDIR_HOST="/local/work/nvidia-workshop-2022-04/user/$LOGNAME"
DATADIR_HOST="/local/work/nvidia-workshop-2022-04/dataset"
WORKDIR_CONTAINER="/workspace"
DATADIR_CONTAINER="$WORKDIR_CONTAINER/dataset"

# Construct the container name from the current user name.
CONTAINERNAME="nvidia_workshop_$LOGNAME"

# Pick a random GPU id between 0 and 5.
NVIDIA_VISIBLE_DEVICES=$(( $RANDOM % 6 ))

# Export the repository working copy into the work directory.
REPODIR="$(dirname "$0")"
mkdir -p "$WORKDIR_HOST/NVIDIA_IKIM_Workshop" \
    && cp -aR "$REPODIR"/* "$WORKDIR_HOST/NVIDIA_IKIM_Workshop/"

# Start the container.
docker run --rm -d \
    --runtime=nvidia \
    --shm-size=4g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --user root \
    --workdir="$WORKDIR_CONTAINER" \
    --env NVIDIA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES" \
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
    -p 8888 \
    --name "$CONTAINERNAME" \
    projectmonai/monai:latest \
    jupyter lab --no-browser

if [ $? -eq 0 ]; then
    # Wait for JupyterLab to start up and obtain the access token.
    echo "Launching JupyterLab..."
    sleep 3
    token=$(docker exec "$CONTAINERNAME" jupyter notebook list | grep -o -m 1 'token=[[:alnum:]]*' | awk -F '=' '{print $2}')
fi

if [ $? -eq 0 ]; then
    # Obtain the host port that docker picked automatically.
    hostport=$(docker port "$CONTAINERNAME" 8888 | grep -o -m 1 '[[:digit:]]\{4,\}')

    # Display instructions.
    echo "The JupyterLab container was started with the following parameters:"
    echo "    Host port: $hostport"
    echo "    Token: $token"
    echo "    GPU index: $NVIDIA_VISIBLE_DEVICES"
    echo
    echo "To connect to the server, open a terminal on your local machine and establish a tunnel using"
    echo "    ssh $(hostname -s) -N -L $hostport:127.0.0.1:$hostport"
    echo "then point your browser to"
    echo "    http://localhost:$hostport/?token=$token"
fi
