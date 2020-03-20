from exaslct_src.exaslct.lib.config.docker_config import docker_client_config
from exaslct_src.test_environment.base.dependency_logger_base_task import DependencyLoggerBaseTask


class DockerBaseTask(DependencyLoggerBaseTask):

    def __init__(self, *args, **kwargs):
        self._init()
        super().__init__(*args, **kwargs)

    def _init(self):
        self._client = docker_client_config().get_client()
        self._low_level_client = docker_client_config().get_low_level_client()

    def __del__(self):
        if "_client" in self.__dict__:
            self._client.close()
        if "_low_level_client" in self.__dict__:
            self._low_level_client.close()

    def __getstate__(self):
        new_dict=super().__getstate__()
        del new_dict["_client"]
        del new_dict["_low_level_client"]
        return new_dict

    def __setstate__(self, new_dict):
        super().__setstate__(new_dict)
        self._init()
