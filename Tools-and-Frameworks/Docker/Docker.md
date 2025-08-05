# Docker 


Labs : 

How to start the first hello-world container ?

Run the command `docker run hello-world:latest` , docker will find the hello-world docker image present locally ? if not then it will try to pull the hello-world docker image from the library/hello-world of docker hub or nvidia nsg after docker image is downloaded then it will run the image as docker container on the docker engine which is running onto or local system.

Pulling of container from docker hub happens layers by layers

Note : each docker container has a shared OS Kernel

What containers are running on our systems ? 

`docker ps` to check the list of running docker containers.

How many docker images are cached there in my system locally ? : `docker images`

How many containers run, stored and running ? status of dockers command : `docker ps -a`

`docker pull ubuntu:18.04` and `docker pull ubuntu:0.4` # used to run the ubuntu OS as docker container

How to check someone has tampered or changed the docker image from the docker hub which we have just pulled ?

`docker inspect docker_id` 

How to run the docker image that we have pulled from the docker hub ?

-d is used for running docker container in deamon mode i.e in the backgroud : `docker run -d repos-name` and `docker run -d redis:latest`

We are not only wanted to just run the docker in the deamon mode but run it in the intractive mode such that we can use it

`docker run -i -t -d --name user_name_of_docker_container docker_image_name:latest` and `docker run -i -t -d --name redis_name redis:latest`

i : intractive, t : terminal, d: deamon

Now our container is running let's move inside the container

`docker exec -it  redis_name /bin/bash`

running the redis_name docker container's bash shell

we make some changes like like create a file having text 

`echo "Hello from redis" > hello-world.txt` 

then we exit the container by `exit` command ok

then to see the output of the file hello-world.txt we again run the fine along with container as 

`docker exec -it redis_name cat hello-world.txt`

? To attach some command with the docker container which is running something  is called as docker attach. docker attack command attaches any primary process inside our container /bin/bash file to the container instance

please check : `docker run -itd -e "USER=panoramic" -e "PASSWARD=trekking" postgress:12`

`docker exec -it docker_id psql --username panoramic --passward`


# NVIDIA NGC 

accessing NGC softwares

it's like docker hub having docker containers which are optimized for running onto the NVIDIA GPUs of your devices, cluster, workstation etc.

1. Sign-in to the ngc plateform
2. Go to the dashboard, select the containers and search for the TensorFlow Container
3. Just copy the command by clicking on the pull tag.
4. New we can use it to the our project
5. To access the containers which are running, we needed to go to the our profile and generate a ngc api key for accessing the container

docke login nvcr.io

username: $oauthtoken

passward: 

output: login succeded then we can access any docker container from the ngc hub with our account.

Let's access the our TensorFlow Container

check

docker images

docker ps -a

using tensorflow and jupyter notebook how we can run some workload onto the DGX or the cluster. via clied server 

docker start jupyter_demo

docker inspect jupyter_demo

docker exec -it docker_id /bin/bash

it will ask for authentical with passward

To run the jupyter notebook into the our container : jupyter notebook --allow-root --notebook-dir=/workspace/chps_demo --ip 0.0.0.0 --port 8962 --no-browser

Now jupyter notebook is running onto the DGX server

and try to open it onto the our local client side system using same port  and ssh and id for authentication

Port forwarding from the local host to the IP address of DGX along with user name

ssh -N -f -L localhost:port_number:IP_Adress_local_system:port_number account_id uday@10.2.0.7

then copy the DGX server running notebook linke /token and run into the browser to see the jupternotebook running into the our local server.

## Jupyter Notebook

Our docker container running onto the DGX-GPU , it's given links to the jupyter notebook but don't have a GUI
