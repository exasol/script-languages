import luigi

from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask
from exaslct_src.lib.base.json_pickle_parameter import JsonPickleParameter
from exaslct_src.lib.data.image_info import ImageInfo
from exaslct_src.lib.docker_config import docker_client_config


class DockerImageCreatorBaseTask(DependencyLoggerBaseTask):
    image_name = luigi.Parameter()
    # ParameterVisibility needs to be hidden instead of private, because otherwise a MissingParameter gets thrown
    image_info = JsonPickleParameter(ImageInfo,
                                     visibility=luigi.parameter.ParameterVisibility.HIDDEN,
                                     significant=True) #type:ImageInfo

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = docker_client_config().get_client()

    def __del__(self):
        self.client.close()
