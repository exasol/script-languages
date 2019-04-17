import docker
import luigi
from luigi.parameter import ParameterVisibility


class docker_config(luigi.Config):
    base_url = luigi.Parameter("unix:///var/run/docker.sock")
    repository_name = luigi.Parameter("exasol/script-language-container")
    username = luigi.OptionalParameter(None, significant=False, visibility=ParameterVisibility.PRIVATE)
    password = luigi.OptionalParameter(None, significant=False, visibility=ParameterVisibility.PRIVATE)
    timeout = luigi.Parameter(300, significant=False, visibility=ParameterVisibility.PRIVATE)

    def get_client(self):
        return docker.DockerClient(base_url=self.base_url, timeout=self.timeout)

    def get_low_level_client(self):
        return docker.APIClient(base_url=self.base_url, timeout=self.timeout)