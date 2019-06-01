import docker
import luigi
from luigi.parameter import ParameterVisibility


class docker_client_config(luigi.Config):
    timeout = luigi.IntParameter(1000, significant=False, visibility=ParameterVisibility.PRIVATE)

    def get_client(self):
        return docker.from_env(timeout=self.timeout, assert_hostname=False)

    def get_low_level_client(self):
        return self.get_client().api


class source_docker_repository_config(luigi.Config):
    repository_name = luigi.Parameter("exasol/script-language-container")
    tag_prefix = luigi.Parameter("")
    username = luigi.OptionalParameter(None, significant=False, visibility=ParameterVisibility.PRIVATE)
    password = luigi.OptionalParameter(None, significant=False, visibility=ParameterVisibility.PRIVATE)


class target_docker_repository_config(luigi.Config):
    repository_name = luigi.Parameter("exasol/script-language-container")
    tag_prefix = luigi.Parameter("")
    username = luigi.OptionalParameter(None, significant=False, visibility=ParameterVisibility.PRIVATE)
    password = luigi.OptionalParameter(None, significant=False, visibility=ParameterVisibility.PRIVATE)
