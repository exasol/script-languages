import datetime
import json
import logging
import os
import pathlib
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any

import docker
import luigi
from docker import APIClient
from jinja2 import Template

from build_utils.lib.build_config import build_config
from build_utils.lib.data.image_info import ImageInfo
from build_utils.lib.abstract_log_handler import AbstractLogHandler
from build_utils.lib.docker_config import docker_config
from build_utils.lib.log_config import log_config, WriteLogFilesToConsole
from build_utils.lib.still_running_logger import StillRunningLogger


class DockerImageBuilder:
    logger = logging.getLogger('luigi-interface')

    def __init__(self, task_id: str,
                 build_directories_mapping: Dict[str, str],
                 dockerfile: str,
                 additional_docker_build_options: Dict[str, Any]):
        self._additional_docker_build_options = additional_docker_build_options
        self._docker_config = docker_config()
        self._build_config = build_config()
        self._log_config = log_config()
        self._low_level_client = APIClient(base_url=self._docker_config.base_url)
        self._task_id = task_id
        self._build_directories_mapping = build_directories_mapping
        self._dockerfile = dockerfile

    def __del__(self):
        self._low_level_client.close()

    def build(self, image_info: ImageInfo,
              image_info_of_dependencies: Dict[str, ImageInfo]):
        log_file_path = self.prepate_log_file_path(image_info)
        self.logger.info("Task %s: Build docker image %s, config file can be found here %s",
                         self._task_id, image_info.complete_name, log_file_path)
        try:
            temp_directory = tempfile.mkdtemp(prefix="script_langauge_container_tmp_dir",
                                              dir=self._build_config.temporary_base_directory)
            self._prepare_build_context_to_temp_dir(temp_directory, image_info_of_dependencies, log_file_path)

            output_generator = \
                self._low_level_client.build(path=temp_directory,
                                             tag=image_info.complete_name,
                                             rm=True,
                                             **self._additional_docker_build_options)
            self._handle_output(output_generator, image_info, log_file_path)
        finally:
            shutil.rmtree(temp_directory)

    def _handle_output(self, output_generator,
                       image_info: ImageInfo,
                       log_file_path: luigi.LocalTarget):
        log_file_path = log_file_path.joinpath("docker-build.log")
        with BuildLogHandler(log_file_path, self.logger, self._task_id, image_info) as log_hanlder:
            still_running_logger = StillRunningLogger(
                self.logger, self._task_id, "build image %s" % image_info.complete_name)
            for log_line in output_generator:
                still_running_logger.log()
                log_hanlder.handle_log_line(log_line)

    def prepate_log_file_path(self, image_info: ImageInfo):
        log_file_path = pathlib.Path("%s/logs/docker-build/%s/%s/%s_%s"
                                     % (self._build_config.ouput_directory,
                                        image_info.name, image_info.tag,
                                        datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                                        image_info.hash
                                        ))
        if log_file_path.exists():
            shutil.rmtree(log_file_path)
        log_file_path.mkdir(parents=True)
        return log_file_path

    def _prepare_build_context_to_temp_dir(
            self, temp_directory, image_info_of_dependencies: Dict[str, ImageInfo], log_file_path: Path):
        self._copy_build_directories(temp_directory)
        self._prepare_dockerfile(temp_directory, image_info_of_dependencies)
        self._log_build_context(temp_directory, log_file_path)

    def _prepare_dockerfile(self, temp_directory, image_info_of_dependencies: Dict[str, ImageInfo]):
        with open(self._dockerfile, "rt") as file:
            dockerfile_content = file.read()
        template = Template(dockerfile_content)
        image_names_of_dependencies = \
            {key: image_info.complete_name for key, image_info in image_info_of_dependencies.items()}
        final_dockerfile = template.render(**image_names_of_dependencies)
        with open(temp_directory + "/Dockerfile", "wt") as file:
            file.write(final_dockerfile)

    def _copy_build_directories(self, temp_directory):
        for dest, src in self._build_directories_mapping.items():
            shutil.copytree(src, temp_directory + "/" + dest)

    def _log_build_context(self, temp_directory, log_file_path: Path):
        if self._build_config.log_build_context_content:
            build_context_log_file = log_file_path.joinpath("docker-build-context.log")
            with build_context_log_file.open("wt") as log_file:
                for file in self._get_files_in_build_context(temp_directory):
                    log_file.write(file)
                    log_file.write("\n")
            dockerfile_log_file = log_file_path.joinpath("Dockerfile.generated")
            shutil.copy(temp_directory + "/Dockerfile", str(dockerfile_log_file))

    def _get_files_in_build_context(self, temp_directory):
        return [os.path.join(r, file) for r, d, f in os.walk(temp_directory) for file in f]


class BuildLogHandler(AbstractLogHandler):

    def __init__(self, log_file_path, logger, task_id, image_info: ImageInfo):
        super().__init__(log_file_path, logger, task_id)
        self._image_info = image_info

    def handle_log_line(self, log_line, error: bool = False):
        log_line = log_line.decode("utf-8")
        self._log_file.write(log_line)
        log_line = log_line.strip('\r\n')
        self._complete_log.append(log_line)
        json_output = json.loads(log_line)
        if 'errorDetail' in json_output:
            self._error_message = json_output["errorDetail"]["message"]

    def finish(self):
        if self._log_config.write_log_files_to_console == WriteLogFilesToConsole.all:
            self._logger.info("Task %s: Build Log of image %s\n%s",
                              self._task_id,
                              self._image_info.complete_name,
                              "\n".join(self._complete_log))
        if self._error_message is not None:
            if self._log_config.write_log_files_to_console == WriteLogFilesToConsole.only_error:
                self._logger.error("Task %s: Build of image %s failed\nBuild Log:\n%s",
                                   self._task_id,
                                   self._image_info.complete_name,
                                   "\n".join(self._complete_log))
            raise docker.errors.BuildError(
                "Error occured during the build of the image %s. Received error \"%s\" ."
                "The whole log can be found in %s"
                % (self._image_info.complete_name,
                   self._error_message,
                   self._log_file_path.absolute()),
                self._log_file_path.absolute())
