# Docker Submission Overview

This page gives an overview of our Docker image.
The default [Dockerfile](../docker/Dockerfile) that we provide defaults to running the random agent.
For the challenge, you will need to modify (or create your own) Dockerfile to run your trained agent.

The remainder of this page describes
1. how to build the example docker image,
1. how to evaluate the example docker image,
1. advice for preparing your own image,
1. evaluating your own image for validation purposes, and
1. steps to submit your image.

We assume you have Docker locally installed.
If not, see [here](https://docs.docker.com/desktop/install/linux-install/) for install instructions on linux.

## Build the Example Docker Image

We provide a an example [Dockerfile](../docker/Dockerfile) that installs competition software dependencies.
This includes various system dependencies, python and various packages, plus competition software.

While not overly significant, we do not install FlightGoggles.
Instead, we mount a host machine folder that contains FlightGoggles at runtime using the `--volume` argument.
The primary purpose of this is to minimize the size of the resulting Docker image.

From the root of this repository, run the following.
```sh
docker build --tag=aia-challenge -f docker/Dockerfile .
```

This will create a docker image called `aia-challenge`.
Note that you may need to provide your own unique tag if you are using shared compute.


## Evaluate the Example Docker Image

This repository contains a script for running evaluations of Docker solutions.
See [evaluate_docker_submission.sh](../docker/evaluate_docker_submission.sh).
Essentially, this script is a simple utility to set up a `docker run` command to run [evaluate-agent.py](../evaluate-agent.py).
It ensure all the necessary flags are set and appropriate arguments passed to `evaluate-agent.py`

To run it, use the following. Note that you'll need to pass the proper path to the FlightGoggles folder and choose a port.
Recall that port numbers are between 1 to 65535, and typically set to something above 1024 (e.g., 8000).
```sh
./docker/evaluate_docker_submission.sh <PATH/TO/FLIGHTGOGGLES/FOLDER> aia-challenge <PORT> 10
```

You will be prompted to enter a seed number within a few seconds.
Afterwards, you'll see content printed to your terminal.

Note the last argument.
That is the number of episodes.
For this example, we're just using 10 in interest of time.
Complete evaluations are for 100 episodes, which is the default value for the fourth argument is not provided.

At completion, there should be three new files in your current folder.
They all begin with a date string (format `%Y-%m-%d_%H-%M-%S`).
The files are as follows.
1. *%Y-%m-%d_%H-%M-%S.output.yaml* is a yaml that contains the results of the evaluation (i.e., episode outcomes).
1. *%Y-%m-%d_%H-%M-%S.stdout.log* contains everything outputted to `stdout` by `evaluate-agent.py`, which can be useful for debugging/analysis.
1. *%Y-%m-%d_%H-%M-%S.stderr.log* contains everything outputted to `stderr` by `evaluate-agent.py`, which can be useful for debugging/analysis.


## Prepare Your Own Docker Image

In order to create your own submission image, you'll need to either 
1. locally modify the provided [Dockerfile](../docker/Dockerfile) or
2. create your own file using our example as a template.

Typically, you should only need to modify the last few lines of the Dockerfile that are copying local files into the Docker image.
If you are adding new software dependencies, then of course you will be responsible for installing those into the Docker image.

Whatever course you choose, we recommend using version control to manage your changes (e.g., fork this repository).

Assuming you have modified our Dockerfile, then you can build your image as follows.
From the root of this repository, run the following from the root of this repository.
```sh
docker build --build-arg GITHUB_TOKEN=<YOUR/GITHUB/TOKEN> --tag=<SOLUTION/NAME> -f docker/Dockerfile .
```
We recommend picking a different tag name to keep track of your images.

### Debugging

It may be helpful to start a `bash` session in your terminal in order to debug things.
In order to facilitate this, there is another shell script available in this repository.
```sh
./docker/run_docker.sh <PATH/TO/FLIGHTGOGGLES/FOLDER> <SOLUTION/NAME>
```

## Evaluate Your Own Docker Image

Evaluation should be exactly the same as before.
```sh
./docker/evaluate_docker_submission.sh <PATH/TO/FLIGHTGOGGLES/FOLDER> <SOLUTION/NAME> <PORT> 10
```
Anyone that has your docker image and FlightGoggles can evaluate your solution now.
Note that we're again setting the number of episodes here to 10 for expediency.

## Submit Your Docker Image

Point us to your docker image and we'll evaluate it.
1. If we know what machine it is on, we can log in and run it.
1. Push the image to a [docker repository](https://docs.docker.com/registry/deploying/) and we can pull it down to run.
1. [Export](https://docs.docker.com/engine/reference/commandline/save/) an image as a tar file and share that file.

We'll give you back the `output.yaml`, the `stdout.log`, and `stderr.log` files.
In the case that `output.yaml` isn't created (note it gets populated and the end of the evaluation), 
you may need to debug the image with the contents of the two log files.
