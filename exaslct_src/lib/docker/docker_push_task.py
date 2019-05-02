import logging
from datetime import datetime
import pathlib

import docker
import luigi

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector
from exaslct_src.lib.data.image_info import ImageInfo
from exaslct_src.lib.docker.push_log_handler import PushLogHandler
from exaslct_src.lib.docker_config import docker_config
from exaslct_src.lib.log_config import log_config
from exaslct_src.lib.still_running_logger import StillRunningLogger
from exaslct_src.stoppable_task import StoppableTask

# TODO don't push if image was pulled
# TODO discover tree of dependencies by following requires results
# TODO or model dependencies in data tasks only and build from there the build and push dependencies
class DockerPushImageTask(StoppableTask):
    logger = logging.getLogger('luigi-interface')
    force_push = luigi.BoolParameter(False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()
        self._build_config = build_config()
        self._client = docker_config().get_client()
        self._log_config = log_config()
        self._prepare_outputs()

    def __del__(self):
        self._client.close()

    def _prepare_outputs(self):
        self._push_info_target = luigi.LocalTarget(
            "%s/info/push/%s"
            % (self._build_config.output_directory, self.task_id))
        if self._push_info_target.exists():
            self._push_info_target.remove()

    def output(self):
        return self._push_info_target

    def requires_tasks(self):
        return self.get_docker_image_task()

    def get_docker_image_task(self):
        pass

    def run_task(self):
        image_info = DependencyImageInfoCollector().get_from_sinlge_input(self.input())
        if image_info.was_build or image_info.was_pulled or self.force_push:
            auth_config = {
                "username": self._docker_config.username,
                "password": self._docker_config.password
            }
            generator = self._client.images.push(repository=image_info.name,
                                                 tag=image_info.tag + "_" + image_info.hash,
                                                 auth_config=auth_config,
                                                 stream=True)
            self._handle_output(generator, image_info)
        self.write_output(image_info)

    def write_output(self, image_info):
        with self._push_info_target.open("w") as file:
            file.write(image_info.complete_name)

    def _handle_output(self, output_generator, image_info: ImageInfo):
        log_file_path = self.prepate_log_file_path(image_info)
        with PushLogHandler(log_file_path, self.logger, self.task_id, image_info) as log_hanlder:
            still_running_logger = StillRunningLogger(
                self.logger, self.task_id, "push image %s" % image_info.complete_name)
            for log_line in output_generator:
                still_running_logger.log()
                log_hanlder.handle_log_line(log_line)

    def prepate_log_file_path(self, image_info: ImageInfo):
        log_file_path = pathlib.Path("%s/logs/docker-push/%s/%s/%s"
                                     % (self._build_config.output_directory,
                                        image_info.name, image_info.tag,
                                        datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
        log_file_path = luigi.LocalTarget(str(log_file_path))
        if log_file_path.exists():
            log_file_path.remove()
        return log_file_path
