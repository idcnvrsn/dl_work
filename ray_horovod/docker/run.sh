WORKDIR=${HOME}/docker-shared

nvidia-docker run \
 --env http_proxy=${HTTP_PROXY} \
 --env https_proxy=${HTTPS_PROXY} \
 --env NO_PROXY=$NO_PROXY \
 --hostname $HOSTNAME \
 --net=host \
-it \
-u "$(id -u $(whoami)):$(id -g $(whoami))" \
-v /etc/passwd:/etc/passwd:ro \
-v /etc/group:/etc/group:ro \
-v ${WORKDIR}:/work/shared \
-v /mnt/mlflow:/mnt/mlflow \
-v ${HOME}/.keras:${HOME}/.keras \
-v ${HOME}/data:${HOME}/data \
-v ${HOME}/.cache:${HOME}/.cache \
-v /etc/localtime:/etc/localtime:ro \
-w /work/shared/idcnvrsn_clone/dl_work/ray_horovod \
--shm-size=1024m \
ray_horovod:latest
