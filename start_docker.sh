echo "Mapping all ports."
docker run -it \
    --gpus all \
    --shm-size=4g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --user root \
	--env SPLEEN_TAR_URL=file:////data/Projects/Essen_Workshop/sources/data/Task09_Spleen.tar \
    -v /data:/data \
    --network=host \
    --name essen_workshop \
    projectmonai/monai:latest
