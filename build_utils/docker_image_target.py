import docker
import luigi


class DockerImageTarget(luigi.Target):
    def __init__(self, image_name: str, image_tag: str):
        self._image_name = image_name
        self._image_tag = image_tag
        self._client = docker.from_env()

    def get_complete_name(self):
        return f"{self._image_name}:{self._image_tag}"

    def exists(self) -> bool:
        try:
            image = self._client.images.get(self.get_complete_name())
            return True
        except docker.errors.ImageNotFound as e:
            return False

    def __del__(self):
        self._client.close()
