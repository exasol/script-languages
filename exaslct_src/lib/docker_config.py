import luigi
from luigi.parameter import ParameterVisibility


class docker_config(luigi.Config):
    base_url = luigi.Parameter("unix:///var/run/docker.sock")
    repository_name = luigi.Parameter("exasol/script-language-container")
    username = luigi.OptionalParameter(None, significant=False, visibility=ParameterVisibility.PRIVATE)
    password = luigi.OptionalParameter(None, significant=False, visibility=ParameterVisibility.PRIVATE)