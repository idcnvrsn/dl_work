WORKDIR=${HOME}/docker-shared
IMAGENAME=ray_horovod:latest

docker_ps_num=`docker ps | grep $IMAGENAME | wc -l`
if [ $docker_ps_num -gt 0 ]; then
    echo "Error : docker image $IMAGENAME already running"
    exit -1
fi

nvidia-docker run \
 --env http_proxy=${HTTP_PROXY} \
 --env https_proxy=${HTTPS_PROXY} \
 --env NO_PROXY=$NO_PROXY \
-it \
--network=host \
-v ${WORKDIR}:/work/shared \
-v /mnt/mlflow:/mnt/mlflow \
--shm-size=1024m \
${IMAGENAME} \
bash -c "/usr/sbin/sshd -p 12345; sleep infinity"
