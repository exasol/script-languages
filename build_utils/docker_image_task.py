import docker
import luigi


class DockerImageTask(luigi.Task):
    image_name = luigi.Parameter()
    image_tag = luigi.Parameter("latest")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = docker.from_env()

    def get_complete_name(self):
        complete_name = f"{self.image_name}:{self.image_tag}"
        return complete_name

    def __del__(self):
        self._client.close()
