import docker
import luigi

from exaslct_src.lib.config.docker_config import docker_client_config


class DockerImageTarget(luigi.Target):
    def __init__(self, image_name: str, image_tag: str):
        self.image_name = image_name
        self.image_tag = image_tag

    def get_complete_name(self):
        return f"{self.image_name}:{self.image_tag}"

    def exists(self) -> bool:
        client = docker_client_config().get_client()
        try:
            image = client.images.get(self.get_complete_name())
            return True
        except docker.errors.DockerException as e:
            return False
        finally:
            client.close()
