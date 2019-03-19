import luigi
from luigi.parameter import ParameterVisibility


class docker_config(luigi.Config):
    base_url = luigi.Parameter("unix:///var/run/docker.sock")
    repository_user = luigi.Parameter("exasol")
    repository_name = luigi.Parameter("script-language-container")
    username = luigi.OptionalParameter(None, significant=False, visibility=ParameterVisibility.PRIVATE)
    password = luigi.OptionalParameter(None, significant=False, visibility=ParameterVisibility.PRIVATE)