import json
import logging
import os
import pathlib
import shutil
import tempfile
from pathlib import Path
from typing import Dict

import docker
from docker import APIClient
from jinja2 import Template

from build_utils.build_config import build_config
from build_utils.docker_config import docker_config
from build_utils.docker_image_target import DockerImageTarget


class DockerImageBuilder:
    logger = logging.getLogger('luigi-interface')

    def __init__(self, task_id: str,
                 build_config:build_config,
                 docker_condig:docker_config,
                 build_directories_mapping:Dict[str,str],
                 dockerfile:str):
        self._docker_condig = docker_condig
        self._build_config = build_config
        self._low_level_client = APIClient(base_url=self._docker_condig.docker_base_url)
        self._task_id = task_id
        self._build_directories_mapping = build_directories_mapping
        self._dockerfile = dockerfile


    def __del__(self):
        self._low_level_client.close()

    def build(self, image_target: DockerImageTarget, images_names_of_dependencies: Dict[str, str]):
        log_file_path = self.prepate_log_file_path(image_target)
        self.logger.info("Task %s: Execute build of image %s, config file can be found here %s",
                         self._task_id, image_target.get_complete_name(), log_file_path)
        try:
            temp_directory = tempfile.mkdtemp(prefix="script_langauge_container_tmp_dir",
                                              dir=self._build_config.build_context_base_directory)
            self._prepare_build_context_to_temp_dir(temp_directory, images_names_of_dependencies, log_file_path)

            output_generator = \
                self._low_level_client.build(path=temp_directory,
                                             tag=image_target.get_complete_name(),
                                             rm=True)
            self._handle_output(output_generator, image_target, log_file_path)
        finally:
            shutil.rmtree(temp_directory)

    def _handle_output(self, output_generator, image_target, log_file_path):
        log_file_path=log_file_path.joinpath("docker-build.log")
        with log_file_path.open("wb") as log_file:
            error = False
            error_message = None
            for log_line in output_generator:
                log_file.write(log_line)
                log_line = log_line.decode("utf-8")
                log_line = log_line.strip('\r\n')
                json_output = json.loads(log_line)
                if 'errorDetail' in json_output:
                    error = True
                    error_message = json_output["errorDetail"]["message"]
        if error:
            raise docker.errors.BuildError(
                "Error occured during the build of the image %s. Received error \"%s\" ."
                "The whole log can be found in %s"
                % (image_target.get_complete_name(),
                   error_message,
                   log_file_path.absolute()),
                log_file_path.absolute())

    def prepate_log_file_path(self, image_target):
        log_file_path = pathlib.Path("%s/logs/docker-build/%s/"
                                     % (self._build_config.ouput_directory,
                                        image_target.get_complete_name()))
        if log_file_path.exists():
            shutil.rmtree(log_file_path)
        log_file_path.mkdir(parents=True)
        return log_file_path

    def _prepare_build_context_to_temp_dir(
            self, temp_directory, images_names_of_dependencies: Dict[str, str], log_file_path:Path):
        self._copy_build_directories(temp_directory)
        self._prepare_dockerfile(temp_directory, images_names_of_dependencies)
        self._log_build_context(temp_directory, log_file_path)

    def _prepare_dockerfile(self, temp_directory, images_names_of_dependencies: Dict[str, str]):
        with open(self._dockerfile, "rt") as file:
            dockerfile_content = file.read()
        template = Template(dockerfile_content)
        final_dockerfile = template.render(**images_names_of_dependencies)
        with open(temp_directory + "/Dockerfile", "wt") as file:
            file.write(final_dockerfile)

    def _copy_build_directories(self, temp_directory):
        for dest, src in self._build_directories_mapping.items():
            shutil.copytree(src, temp_directory + "/" + dest)

    def _log_build_context(self, temp_directory, log_file_path:Path):
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
