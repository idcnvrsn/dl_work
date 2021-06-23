WORKDIR=${HOME}/docker-shared

nvidia-docker run  \
-it \
 --env http_proxy=${HTTP_PROXY} \
 --env https_proxy=${HTTPS_PROXY} \
 --env NO_PROXY=$NO_PROXY \
 --network=host \
-v ${WORKDIR}:/work/shared \
-v /mnt/mlflow:/mnt/mlflow \
-v ${HOME}/.keras:${HOME}/.keras \
-v /etc/localtime:/etc/localtime:ro \
-w /work/shared/idcnvrsn_clone/dl_work/ray_horovod \
--shm-size=1024m \
ray_horovod:latest
