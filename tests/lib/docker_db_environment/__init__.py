import os

def is_docker_available():
    path = "/var/run/docker.sock"
    return os.path.exists(path) and os.access(path, os.W_OK)

def get_docker_client():
    if is_docker_available():
        import docker
        client = docker.DockerClient(base_url='unix://var/run/docker.sock')
        return client
    else:
        raise Exception("Docker not available")

def get_environment_type():
    return os.environ["TEST_ENVIRONMENT_TYPE"]

def get_environment_name():
    return os.environ["TEST_ENVIRONMENT_NAME"]

def get_db_container_name():
    return os.environ["TEST_DOCKER_DB_CONTAINER_NAME"]

def get_docker_network_name():
    return os.environ["TEST_DOCKER_NETWORK_NAME"]

def is_available():
    return  is_docker_available() and \
            get_environment_type() == "docker_db" and \
            get_db_container_name is not None and \
            get_docker_network_name() is not None

class DockerDBEnvironment:
    
    def __init__(self, test_name):
        self._client = get_docker_client()
        self._test_name=test_name
        self._started_containers = []

    def __del__(self):
        self.close()

    def close(self):
        self._client.close()
        self.remove_all_started_containers()

    def get_client(self):
        return self._client

    def remove_all_started_containers(self):
        for container in self._started_containers:
            try:
                container.remove(v=True,force=True)
            except:
                pass
        self._started_containers = []

    def remove_started_container(self, container):
        try:
            container.remove(v=True,force=True)
        except:
            pass
        self._started_containers.remove(containers)

    def list_started_containers(self):
        return list(self._started_containers)

    def get_docker_db_container(self):
        container = self._client.containers.get(get_db_container_name()) # TODO protect that it doesn't get killed
        container.reload()
        return container

    def get_container_name_prefix(self):
        return ("%s_%s_"%(get_environment_name(),self._test_name)).replace("_","-").replace(".","-").replace("-","").lower()[:10]

    def run(self, name, image, command=None,**kwargs):
        kwargs["name"]="%s%s"%(self.get_container_name_prefix(),name)
        labels={"test_environment_name":get_environment_name(),"container_type":"test_case_container","test_case":self._test_name}
        if "labels" in kwargs:
            kwargs["labels"].update(labels)
        else:
            kwargs["labels"]=labels
        kwargs["network"]=get_docker_network_name()
        kwargs["detach"]=True
        kwargs["image"]=image
        kwargs["command"]=command
        try:
            container=self._client.containers.get(kwargs["name"])
            container.remove(v=True,force=True)
        except:
            pass
        container=self._client.containers.run(**kwargs)
        container.reload()
        self._started_containers.append(container)
        return container

    def get_ip_address_of_container(self, container):
        if container in self._started_containers or container.name == self.get_docker_db_container().name:
            return container.attrs['NetworkSettings']['Networks'][get_docker_network_name()]['IPAddress']
        else:
            raise Exception("Container not found in started containers")
