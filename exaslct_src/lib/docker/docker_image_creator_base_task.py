import logging

import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.image_info import ImageInfo
from exaslct_src.lib.docker.docker_image_target import DockerImageTarget
from exaslct_src.lib.docker_config import docker_client_config
from exaslct_src.lib.stoppable_task import StoppableTask


class DockerImageCreatorBaseTask(StoppableTask):
    logger = logging.getLogger('luigi-interface')
    image_name = luigi.Parameter()
    # ParameterVisibility needs to be hidden instead of private, because otherwise a MissingParameter gets thrown
    image_info_json = luigi.Parameter(visibility=luigi.parameter.ParameterVisibility.HIDDEN,
                                      significant=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = docker_client_config().get_client()
        self.image_info = ImageInfo.from_json(self.image_info_json)
        self.image_target = DockerImageTarget(self.image_info.target_repository_name,
                                              self.image_info.get_target_complete_tag())
        self.remove_image()

    def __del__(self):
        self.client.close()

    def remove_image(self):
        if self.image_target.exists():
            self.client.images.remove(image=self.image_target.get_complete_name(), force=True)
            self.logger.warning("Task %s: Removed docker images %s",
                                self.task_id, self.image_target.get_complete_name())

    def output(self):
        return self.image_target