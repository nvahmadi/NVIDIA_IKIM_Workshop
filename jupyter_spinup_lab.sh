#!/usr/bin/env bash
set -u

portnr=${1:-9020}

echo "Starting Jupyter server @ port $portnr."
echo "Command:    jupyter lab --ip 0.0.0.0 --no-browser --port $portnr"
echo "Kill via:   jupyter notebook stop $portnr"

cd "$WORKDIR"
nohup jupyter lab --ip 0.0.0.0 --no-browser --port "$portnr" &
sleep 3.
jupyter notebook list
