CONTAINER_NAME=test_exaslct_init_script

start_container (){
  docker rm -f $CONTAINER_NAME
  docker run -d --name $CONTAINER_NAME $1 sleep infinity
}

docker build -t exaslct_test_container .

# start_container
# docker exec -it --workdir /test  $CONTAINER_NAME ./exaslct


# start_container
# docker exec -it --workdir /test  $CONTAINER_NAME apt update &> /dev/null
# docker exec -it --workdir /test  $CONTAINER_NAME apt install -y python3-pip git &> /dev/null
# docker exec -it --workdir /test  $CONTAINER_NAME bash -c "export LC_ALL=C.UTF-8 && export LANG=C.UTF-8 && echo yes | ./exaslct"

start_container exaslct_test_container
docker exec -it --workdir /test  $CONTAINER_NAME virtualenv venv
docker exec -it --workdir /test  $CONTAINER_NAME bash -c "source /venv/bin/activate && export LC_ALL=C.UTF-8 && export LANG=C.UTF-8 && echo yes | ./exaslct"

docker rm -f $CONTAINER_NAME
