import logging
import pathlib
from datetime import datetime

import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector
from exaslct_src.lib.data.image_info import ImageInfo, ImageState
from exaslct_src.lib.docker.push_log_handler import PushLogHandler
from exaslct_src.lib.docker_config import docker_client_config, target_docker_repository_config
from exaslct_src.lib.log_config import log_config
from exaslct_src.lib.still_running_logger import StillRunningLogger
from exaslct_src.lib.stoppable_task import StoppableTask

class DockerPushImageBaseTask(StoppableTask):
    image_name = luigi.Parameter()
    logger = logging.getLogger('luigi-interface')
    force_push = luigi.BoolParameter(False, visibility=luigi.parameter.ParameterVisibility.HIDDEN)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self._client = docker_client_config().get_client()
        self._log_config = log_config()
        self.docker_image_task = self.get_docker_image_task()
        self._prepare_outputs()

    def __del__(self):
        self._client.close()

    def _prepare_outputs(self):
        self._push_info_target = luigi.LocalTarget(
            "%s/info/push/%s"
            % (build_config().output_directory, self.task_id))
        if self._push_info_target.exists():
            self._push_info_target.remove()

    def output(self):
        return self._push_info_target

    def requires_tasks(self):
        return self.docker_image_task

    def get_docker_image_task(self):
        pass

    def run_task(self):
        image_info = DependencyImageInfoCollector().get_from_sinlge_input(self.input())
        was_build = image_info.image_state == ImageState.WAS_BUILD.name
        if was_build or self.force_push:
            auth_config = {
                "username": target_docker_repository_config().username,
                "password": target_docker_repository_config().password
            }
            generator = self._client.images.push(repository=image_info.get_target_complete_name(),
                                                 tag=image_info.get_target_complete_tag(),
                                                 auth_config=auth_config,
                                                 stream=True)
            self._handle_output(generator, image_info)
        self.write_output(image_info)

    def write_output(self, image_info:ImageInfo):
        with self._push_info_target.open("w") as file:
            file.write(image_info.get_target_complete_name())

    def _handle_output(self, output_generator, image_info: ImageInfo):
        log_file_path = self.prepate_log_file_path(image_info)
        with PushLogHandler(log_file_path, self.logger, self.__repr__(), image_info) as log_hanlder:
            still_running_logger = StillRunningLogger(
                self.logger, self.__repr__(), "push image %s" % image_info.get_target_complete_name())
            for log_line in output_generator:
                still_running_logger.log()
                log_hanlder.handle_log_line(log_line)

    def prepate_log_file_path(self, image_info: ImageInfo):
        log_file_path = pathlib.Path("%s/logs/docker-push/%s/%s/%s"
                                     % (build_config().output_directory,
                                        image_info.target_repository_name, image_info.target_tag,
                                        datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
        log_file_path = luigi.LocalTarget(str(log_file_path))
        if log_file_path.exists():
            log_file_path.remove()
        return log_file_path
