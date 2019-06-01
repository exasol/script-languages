import datetime
import logging
import pathlib
import shutil
import tempfile

import luigi
from docker import APIClient

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.image_info import ImageInfo
from exaslct_src.lib.docker.build_context_creator import BuildContextCreator
from exaslct_src.lib.docker.build_log_handler import BuildLogHandler
from exaslct_src.lib.docker_config import docker_client_config, target_docker_repository_config
from exaslct_src.lib.log_config import log_config
from exaslct_src.lib.still_running_logger import StillRunningLogger


class DockerImageBuilder:
    logger = logging.getLogger('luigi-interface')

    def __init__(self, task_id: str):


        self._log_config = log_config()
        self._low_level_client = docker_client_config().get_low_level_client()
        self._task_id = task_id

    def __del__(self):
        self._low_level_client.close()

    def build(self, image_info: ImageInfo):
        log_file_path = self.prepare_log_file_path(image_info)
        self.logger.info("Task %s: Build docker image %s, log file can be found here %s",
                         self._task_id, image_info.get_target_complete_name(), log_file_path)
        temp_directory = tempfile.mkdtemp(prefix="script_langauge_container_tmp_dir",
                                          dir=build_config().temporary_base_directory)
        try:
            image_description = image_info.image_description
            build_context_creator = BuildContextCreator(temp_directory,
                                                        image_info,
                                                        log_file_path)
            build_context_creator.prepare_build_context_to_temp_dir()
            output_generator = \
                self._low_level_client.build(path=temp_directory,
                                             tag=image_info.get_target_complete_name(),
                                             rm=True,
                                             **image_description.transparent_build_arguments,
                                             **image_description.image_changing_build_arguments)
            self._handle_output(output_generator, image_info, log_file_path)
        finally:
            shutil.rmtree(temp_directory)

    def _handle_output(self, output_generator,
                       image_info: ImageInfo,
                       log_file_path: luigi.LocalTarget):
        log_file_path = log_file_path.joinpath("docker-build.log")
        with BuildLogHandler(log_file_path, self.logger, self._task_id, image_info) as log_hanlder:
            still_running_logger = StillRunningLogger(
                self.logger, self._task_id, "build image %s" % image_info.get_target_complete_name())
            for log_line in output_generator:
                still_running_logger.log()
                log_hanlder.handle_log_line(log_line)

    def prepare_log_file_path(self, image_info: ImageInfo):
        log_file_path = pathlib.Path("%s/logs/docker-build/%s/%s/%s_%s"
                                     % (build_config().output_directory,
                                        image_info.target_repository_name, image_info.target_tag,
                                        datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                                        image_info.hash
                                        ))
        if log_file_path.exists():
            shutil.rmtree(log_file_path)
        log_file_path.mkdir(parents=True)
        return log_file_path


