import logging
from datetime import datetime
import json
import pathlib

import docker
import luigi

from build_utils.lib.build_config import build_config
from build_utils.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector
from build_utils.lib.data.image_info import ImageInfo
from build_utils.lib.abstract_log_handler import AbstractLogHandler
from build_utils.lib.docker_config import docker_config
from build_utils.lib.log_config import log_config, WriteLogFilesToConsole
from build_utils.lib.still_running_logger import StillRunningLogger


class DockerPushImageTask(luigi.Task):
    logger = logging.getLogger('luigi-interface')
    flavor_path = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._docker_config = docker_config()
        self._build_config = build_config()
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._log_config = log_config()
        self._prepare_outputs()

    def __del__(self):
        self._client.close()

    def _prepare_outputs(self):
        self._push_info_target = luigi.LocalTarget(
            "%s/push_info/%s"
            % (self._build_config.ouput_directory, self.task_id))
        if self._push_info_target.exists():
            self._push_info_target.remove()

    def output(self):
        return self._push_info_target

    def requires(self):
        return self.get_docker_image_task(self.flavor_path)

    def get_docker_image_task(self, flavor_path):
        pass

    def run(self):
        image_info = DependencyImageInfoCollector().get_from_sinlge_input(self.input())
        generator = self._client.images.push(repository=image_info.name, tag=image_info.tag + "_" + image_info.hash,
                                             auth_config={
                                                 "username": self._docker_config.username,
                                                 "password": self._docker_config.password
                                             },
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
                                     % (self._build_config.ouput_directory,
                                        image_info.name, image_info.tag,
                                        datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
        log_file_path = luigi.LocalTarget(str(log_file_path))
        if log_file_path.exists():
            log_file_path.remove()
        return log_file_path


class PushLogHandler(AbstractLogHandler):

    def __init__(self, log_file_path, logger, task_id, image_info: ImageInfo):
        super().__init__(log_file_path, logger, task_id)
        self._image_info = image_info

    def handle_log_line(self, log_line, error:bool=False):
        log_line = log_line.decode("utf-8")
        log_line = log_line.strip('\r\n')
        json_output = json.loads(log_line)
        if "status" in json_output and json_output["status"] != "Pushing":
            self._complete_log.append(log_line)
            self._log_file.write(log_line)
            self._log_file.write("\n")
        if 'errorDetail' in json_output:
            self._error_message = json_output["errorDetail"]["message"]

    def finish(self):
        if self._log_config.write_log_files_to_console==WriteLogFilesToConsole.all:
            self._logger.info("Task %s: Push Log of image %s\n%s",
                              self._task_id,
                              self._image_info.complete_name,
                              "\n".join(self._complete_log))
        if self._error_message is not None:
            if self._log_config.write_log_files_to_console == WriteLogFilesToConsole.only_error:
                self._logger.error("Task %s: Push of image %s failed\nPush Log:\n%s",
                                  self._task_id,
                                  self._image_info.complete_name,
                                  "\n".join(self._complete_log))
            raise Exception(
                "Error occured during the push of the image %s. Received error \"%s\" ."
                "The whole log can be found in %s"
                % (self._image_info.complete_name, self._error_message, self._log_file_path.path))
