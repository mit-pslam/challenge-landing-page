#!/bin/bash

# Get the FlightGoggles path and Docker image name from command line arguments
FLIGHTGOGGLES_PATH=$1
DOCKER_IMAGE_NAME=$2

XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    echo "$XAUTH file does not exist, creating one"
    touch $XAUTH
    xauth_list=$(xauth nlist $DISPLAY | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
	echo "Populating $XAUTH with $xauth_list"
        xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
    else
        echo "ERROR: nothing to populate XAUTH with"
    fi
    chmod a+r $XAUTH
fi

docker run --gpus=all \
    --net=host \
    --env "DISPLAY=$DISPLAY" \
    --env "QT_X11_NO_MITSHM=1" \
    --volume "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env "XAUTHORITY=$XAUTH" \
    --volume "$XAUTH:$XAUTH" \
    --volume $FLIGHTGOGGLES_PATH:/challenge/FlightGoggles/ \
    --privileged \
    -it $DOCKER_IMAGE_NAME bash
