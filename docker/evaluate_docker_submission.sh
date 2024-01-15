#!/bin/bash

# Get the FlightGoggles path and Docker image name from command line arguments
FLIGHTGOGGLES_PATH=$1
DOCKER_IMAGE_NAME=$2
PORT=$3

# Get the number of episodes from command line arguments. If not provided, default to 100
EPISODES=${4:-100}


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

# Create output file name that combines "output" with the current date and time
DATE=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT="/challenge/output/$DATE.output.yaml"
STDOUT="/challenge/output/$DATE.stdout.log"
STDERR="/challenge/output/$DATE.stderr.log"


docker run --gpus=all \
    --net=host \
    --env "DISPLAY=$DISPLAY" \
    --env "QT_X11_NO_MITSHM=1" \
    --volume "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env "XAUTHORITY=$XAUTH" \
    --volume "$XAUTH:$XAUTH" \
    --volume $FLIGHTGOGGLES_PATH:/challenge/FlightGoggles/ \
    --volume $PWD:/challenge/output/ \
    --privileged \
    -it $DOCKER_IMAGE_NAME \
    /bin/bash -c "python /challenge/challenge-landing-page/evaluate-agent.py \
        --agent-config /challenge/challenge-landing-page/configs/eval-submission-agent.yaml \
        --episodes $EPISODES \
        --flight-goggles-path /challenge/FlightGoggles/FlightGogglesv3.x86_64 \
        --base-port $PORT \
        --output-file $OUTPUT \
        --disable \
        > >(tee $STDOUT) 2> >(tee $STDERR >&2)"  # record stdout and stderr to files
