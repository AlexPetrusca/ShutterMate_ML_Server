#!/usr/bin/env bash

# kill all running docker containers
docker kill $(docker ps -q)

# delete all images and stopped containers
docker rm $(docker ps -a -q)
docker rmi shuttermate --force

# build and run docker process
docker build -t shuttermate .

#run docker image
docker run -d -p 80:80 shuttermate

# show running containers
docker ps
