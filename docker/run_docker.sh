#!/bin/bash

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
    --volume ${PWD}/challenge/:/work/challenge \
    --volume /home/triton/Documents/FlightGogglesv3-release:/work/FG/ \
    --privileged \
    -it aiia-challenge bash
