import luigi

from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask
from exaslct_src.lib.base.json_pickle_parameter import JsonPickleParameter
from exaslct_src.lib.data.image_info import ImageInfo
from exaslct_src.lib.base.docker_base_task import DockerBaseTask


class DockerImageCreatorBaseTask(DockerBaseTask):
    image_name = luigi.Parameter()
    # ParameterVisibility needs to be hidden instead of private, because otherwise a MissingParameter gets thrown
    image_info = JsonPickleParameter(ImageInfo,
                                     visibility=luigi.parameter.ParameterVisibility.HIDDEN,
                                     significant=True) #type:ImageInfo

